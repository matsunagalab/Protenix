# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch

from protenix.model.utils import centre_random_augmentation


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list

def get_ca_coord_for_residue(x_l, input_feature_dict, residue_i):
    """
    残基iのCα原子の座標を取り出す関数
    
    Args:
        x_l: 原子座標テンソル [..., N_sample, N_atom, 3]
        input_feature_dict: 入力特徴量辞書
        residue_i: 残基番号（0-indexベース）
    
    Returns:
        ca_coords: 指定された残基のCα原子の座標 [..., N_sample, 3]
    """
    # 残基iに属する原子のマスクを作成
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    residue_mask = (atom_to_token_idx == residue_i)
    
    # 原子名からCαを特定
    atom_names = input_feature_dict["ref_atom_name_chars"]  # [N_atom, 4, 64]
    # 'C'と'A'のエンコードされた位置: ord(c) - 32
    c_pos = ord('C') - 32  # 35
    a_pos = ord('A') - 32  # 33
    
    # 最初の文字が'C'で2番目が'A'の原子を探す
    is_c = (atom_names[:, 0, c_pos] == 1)
    is_a = (atom_names[:, 1, a_pos] == 1)
    ca_mask = is_c & is_a
    
    # 残基iのCα原子のインデックスを取得
    matching_atoms = torch.where(residue_mask & ca_mask)[0]
    if len(matching_atoms) == 0:
        raise ValueError(f"No CA atom found for residue {residue_i}")
    atom_idx = matching_atoms[0]
    
    # 座標を取り出す
    ca_coords = x_l[..., atom_idx, :]
    
    return ca_coords

def calc_ca_distance_loss(res_i, res_j, x_coords, input_feature_dict):
    """
    残基iと残基jのCα原子間距離が1.0からどれだけ離れているかの二乗損失を計算
    
    Args:
        res_i: 最初の残基のインデックス
        res_j: 2番目の残基のインデックス
        x_coords: 座標テンソル [..., N_sample, N_atom, 3] (requires_grad=True が必要)
        input_feature_dict: 入力特徴量辞書
    
    Returns:
        loss: 損失値（スカラー）
    """
    # 残基のCα原子の座標を取得
    ca_i = get_ca_coord_for_residue(x_coords, input_feature_dict, res_i)  # [..., N_sample, 3]
    ca_j = get_ca_coord_for_residue(x_coords, input_feature_dict, res_j)  # [..., N_sample, 3]
    
    # Cα原子間の距離を計算
    ca_dist = torch.sqrt(torch.sum((ca_j - ca_i) ** 2, dim=-1))  # [..., N_sample]
    print("ca_dist:", ca_dist.item())
    
    # 距離1.0からの差分の二乗を計算
    loss = torch.mean((ca_dist - 15.0) ** 2)  # スカラー
    
    return loss

def calc_com_distance_loss(res_i, res_j, x_coords, input_feature_dict):
    """
    残基グループi と残基グループj の重心間距離が0.5からどれだけ離れているかの二乗損失を計算
    
    Args:
        res_i: 最初の残基グループのインデックスのリスト
        res_j: 2番目の残基グループのインデックスのリスト
        x_coords: 座標テンソル [..., N_sample, N_atom, 3] (requires_grad=True が必要)
        input_feature_dict: 入力特徴量辞書
    
    Returns:
        loss: 損失値（スカラー）
    """
    # 残基グループiの重心を計算
    centroid_i = torch.zeros_like(x_coords[..., 0, :])  # [..., N_sample, 3]
    count = 0
    for idx in res_i:
        # 残基idxに属する原子のマスクを作成
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        residue_mask = (atom_to_token_idx == idx)
        
        # マスクに該当する原子の座標を取得
        matching_atoms = torch.where(residue_mask)[0]
        if len(matching_atoms) == 0:
            raise ValueError(f"No atoms found for residue {idx}")
        
        # 残基内の全原子の座標を合計
        for atom_idx in matching_atoms:
            centroid_i += x_coords[..., atom_idx, :]
        
        # 残基内の原子数でカウント
        count += len(matching_atoms)

    centroid_i = centroid_i / count

    # 残基グループjの重心を計算
    centroid_j = torch.zeros_like(x_coords[..., 0, :])  # [..., N_sample, 3]
    count = 0
    for idx in res_j:
        # 残基idxに属する原子のマスクを作成
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        residue_mask = (atom_to_token_idx == idx)
        
        # マスクに該当する原子の座標を取得
        matching_atoms = torch.where(residue_mask)[0]
        if len(matching_atoms) == 0:
            raise ValueError(f"No atoms found for residue {idx}")
        
        # 残基内の全原子の座標を合計
        for atom_idx in matching_atoms:
            centroid_j += x_coords[..., atom_idx, :]
        
        # 残基内の原子数でカウント
        count += len(matching_atoms)
    
    centroid_j = centroid_j / count

    # 重心間の距離を計算
    centroid_dist = torch.sqrt(torch.sum((centroid_j - centroid_i) ** 2, dim=-1))  # [..., N_sample]
    
    # 距離0.5からの差分の二乗を計算
    loss = torch.mean((centroid_dist - 10.0) ** 2)  # スカラー
    
    return loss

def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        for _, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )

            with torch.enable_grad():
                x_coords = x_noisy.clone().detach().requires_grad_(True)
    
                # 距離制約の損失を計算
                #loss = calc_ca_distance_loss(10, 129, x_coords, input_feature_dict)
                loss = calc_com_distance_loss([127, 128, 129, 130, 131], [7, 8, 9, 10, 11], x_coords, input_feature_dict)

                # 勾配を計算
                loss.backward()
    
                # 勾配を取得
                gradient = x_coords.grad
    
                print("Loss:", loss.item())
                #print("Gradient shape:", gradient.shape)
                #print("Gradient max:", gradient.abs().max().item())
            
            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.

            g = gradient
            g = g * (torch.norm(delta.reshape(-1)) / torch.norm(g.reshape(-1)))
            delta = delta + 0.2 * g

            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
