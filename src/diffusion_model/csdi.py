from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusion_model.inference_utils import fourier_loss, total_variation_loss


def reshape_for_broadcast(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
	return values.view(-1, *([1] * (like.dim() - 1)))


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
	half_dim = dim // 2
	if half_dim == 0:
		return values.unsqueeze(-1)

	values = values.float()
	freq_exponent = torch.arange(half_dim, device=values.device, dtype=values.dtype)
	freq_exponent = freq_exponent / max(half_dim - 1, 1)
	freq = torch.exp(-math.log(max_period) * freq_exponent)
	angles = values.unsqueeze(-1) * freq
	embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

	if dim % 2 == 1:
		embedding = F.pad(embedding, (0, 1))

	return embedding


def to_b1l1(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
	if x.dim() == 4:
		return x

	if x.dim() == 1:
		return x.unsqueeze(0).unsqueeze(1).unsqueeze(-1)

	if x.dim() == 3:
		if x.shape[1] == 1:
			return x.unsqueeze(-1)
		if x.shape[-1] == 1:
			return x.unsqueeze(1)

	if x.dim() == 2:
		return x.unsqueeze(1).unsqueeze(-1)

	raise ValueError(f"{name} must have 1, 2, 3, or 4 dimensions, got shape={tuple(x.shape)}")



def restore_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
	if reference.dim() == 4:
		return x
	if reference.dim() == 1:
		return x.squeeze(0).squeeze(0).squeeze(-1)
	if reference.dim() == 3:
		if reference.shape[1] == 1:
			return x.squeeze(-1)
		if reference.shape[-1] == 1:
			return x.squeeze(1)
	if reference.dim() == 2:
		return x.squeeze(1).squeeze(-1)
	raise ValueError(f"reference must have 1, 2, 3, or 4 dimensions, got shape={tuple(reference.shape)}")


class Conv1x1LastDim(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.permute(0, 3, 1, 2)
		x = self.conv(x)
		return x.permute(0, 2, 3, 1)


class CSDIResidualLayer(nn.Module):
	def __init__(
		self,
		hidden_channels: int,
		side_channels: int,
		nheads: int,
		dropout: float,
	) -> None:
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_channels,
			nhead=nheads,
			dim_feedforward=hidden_channels * 4,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
		self.channel_expand = Conv1x1LastDim(hidden_channels, hidden_channels * 2)
		self.side_projection = Conv1x1LastDim(side_channels, hidden_channels * 2)
		self.residual_projection = Conv1x1LastDim(hidden_channels, hidden_channels)
		self.skip_projection = Conv1x1LastDim(hidden_channels, hidden_channels)

	def forward(self, x: torch.Tensor, side_information: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		h = x.squeeze(1)
		h = self.temporal_transformer(h)
		h = h.unsqueeze(1)

		h = self.channel_expand(h)
		h = h + self.side_projection(side_information)

		h_tanh, h_sigmoid = torch.chunk(h, chunks=2, dim=-1)
		gated = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)

		residual = self.residual_projection(gated)
		skip = self.skip_projection(gated)
		return x + residual, skip


class CSDIDenoiser(nn.Module):
	def __init__(
		self,
		hidden_channels: int = 64,
		n_residual_layers: int = 4,
		nheads: int = 8,
		dropout: float = 0.0,
		diffusion_embedding_dim: int = 128,
		time_embedding_dim: int = 128,
	) -> None:
		super().__init__()

		self.hidden_channels = hidden_channels
		self.diffusion_embedding_dim = diffusion_embedding_dim
		self.time_embedding_dim = time_embedding_dim

		self.main_projection = Conv1x1LastDim(2, hidden_channels)
		self.diffusion_embedding_projection = nn.Sequential(
			nn.Linear(diffusion_embedding_dim, diffusion_embedding_dim),
			nn.SiLU(),
		)
		self.diffusion_to_hidden = Conv1x1LastDim(diffusion_embedding_dim, hidden_channels)

		side_channels = time_embedding_dim + 1
		self.residual_layers = nn.ModuleList(
			[
				CSDIResidualLayer(
					hidden_channels=hidden_channels,
					side_channels=side_channels,
					nheads=nheads,
					dropout=dropout,
				)
				for _ in range(n_residual_layers)
			]
		)

		self.output_projection_1 = Conv1x1LastDim(hidden_channels, hidden_channels)
		self.output_projection_2 = Conv1x1LastDim(hidden_channels, 1)

	def prepare_condition(
		self,
		x_t: torch.Tensor,
		condition: dict[str, torch.Tensor] | torch.Tensor | None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
		if condition is None:
			x_co = torch.zeros_like(x_t)
			m_co = torch.zeros_like(x_t)
			time_index = None
		elif isinstance(condition, dict):
			x_co_raw = condition.get("x_co")
			m_co_raw = condition.get("m_co")
			time_index = condition.get("time_index")

			x_co = to_b1l1(x_co_raw, name="condition['x_co']") if isinstance(x_co_raw, torch.Tensor) else torch.zeros_like(x_t)
			m_co = to_b1l1(m_co_raw, name="condition['m_co']") if isinstance(m_co_raw, torch.Tensor) else torch.ones_like(x_t)
		else:
			raise TypeError("condition must be None, a tensor, or a dict")

		return x_co, m_co, time_index

	def build_time_context(
		self,
		batch_size: int,
		seq_len: int,
		device: torch.device,
		dtype: torch.dtype,
		time_index: torch.Tensor | None,
	) -> torch.Tensor:
		if time_index is None:
			time_index = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
		elif time_index.dim() == 1:
			time_index = time_index.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
		elif time_index.dim() == 2:
			time_index = time_index.to(device=device, dtype=dtype)

		time_embedding = sinusoidal_embedding(time_index, self.time_embedding_dim)
		return time_embedding.unsqueeze(1)

	def forward(
		self,
		x_t: torch.Tensor,
		t: torch.Tensor,
		condition: dict[str, torch.Tensor] | torch.Tensor | None = None,
	) -> torch.Tensor:
		x_reference = x_t
		x_t = to_b1l1(x_t, name="x_t")
		batch_size, _, seq_len, _ = x_t.shape

		x_co, m_co, time_index = self.prepare_condition(x_t, condition)

		main_input = torch.cat([x_t, x_co], dim=-1)
		h = F.relu(self.main_projection(main_input))

		t_embedding = sinusoidal_embedding(t, self.diffusion_embedding_dim).to(device=x_t.device, dtype=x_t.dtype)
		t_embedding = self.diffusion_embedding_projection(t_embedding)
		t_embedding = t_embedding.view(batch_size, 1, 1, self.diffusion_embedding_dim)
		h = h + self.diffusion_to_hidden(t_embedding)

		time_context = self.build_time_context(
			batch_size=batch_size,
			seq_len=seq_len,
			device=x_t.device,
			dtype=x_t.dtype,
			time_index=time_index,
		)
		side_information = torch.cat([time_context, m_co], dim=-1)

		skip_connections = []
		for layer in self.residual_layers:
			h, skip = layer(h, side_information)
			skip_connections.append(skip)

		h = torch.stack(skip_connections, dim=0).sum(dim=0)
		h = F.relu(self.output_projection_1(h))
		h = self.output_projection_2(h)

		flat_mask = m_co.view(batch_size, -1)
		is_all_zero = flat_mask.eq(0).all(dim=1)
		is_all_one = flat_mask.eq(1).all(dim=1)
		should_apply_mask = ~(is_all_zero | is_all_one)
		if bool(should_apply_mask.any()):
			apply_mask = should_apply_mask.view(batch_size, 1, 1, 1)
			h = torch.where(apply_mask, h * (1.0 - m_co), h)

		return restore_like(h, x_reference)


class CosineScheduler(nn.Module):
	def __init__(
		self,
		timesteps: int,
		s: float = 0.008,
		beta_min: float = 1e-5,
		beta_max: float = 0.999,
		dtype: torch.dtype = torch.float32,
	) -> None:
		super().__init__()

		self.timesteps = timesteps

		x = torch.linspace(0, timesteps, timesteps + 1, dtype=dtype)
		alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
		alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

		betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
		betas = betas.clamp(min=beta_min, max=beta_max)

		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas, dim=0)
		alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

		sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
		sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
		sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

		posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
		posterior_variance = posterior_variance.clamp(min=1e-20)
		posterior_log_variance = torch.log(posterior_variance)

		posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
		posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

		self.register_buffer("betas", betas)
		self.register_buffer("alphas", alphas)
		self.register_buffer("alphas_cumprod", alphas_cumprod)
		self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
		self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
		self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
		self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
		self.register_buffer("posterior_variance", posterior_variance)
		self.register_buffer("posterior_log_variance", posterior_log_variance)
		self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
		self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

	def coefficients(self, t: torch.Tensor, like: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
		t = t.long()
		coeffs = {
			"beta": self.betas.gather(0, t),
			"alpha": self.alphas.gather(0, t),
			"alpha_cumprod": self.alphas_cumprod.gather(0, t),
			"alpha_cumprod_prev": self.alphas_cumprod_prev.gather(0, t),
			"sqrt_alpha_cumprod": self.sqrt_alphas_cumprod.gather(0, t),
			"sqrt_one_minus_alpha_cumprod": self.sqrt_one_minus_alphas_cumprod.gather(0, t),
			"sqrt_recip_alpha": self.sqrt_recip_alphas.gather(0, t),
			"posterior_variance": self.posterior_variance.gather(0, t),
			"posterior_log_variance": self.posterior_log_variance.gather(0, t),
			"posterior_mean_coef1": self.posterior_mean_coef1.gather(0, t),
			"posterior_mean_coef2": self.posterior_mean_coef2.gather(0, t),
		}
		if like is None:
			return coeffs
		return {k: reshape_for_broadcast(v, like) for k, v in coeffs.items()}



class DiffusionModel(nn.Module):
	def __init__(
		self,
		timesteps: int,
		denoiser: nn.Module | None = None,
		cond_drop_prob: float = 0.2,
	) -> None:
		super().__init__()
		self.scheduler = CosineScheduler(timesteps=timesteps)
		self.denoiser = denoiser if denoiser is not None else CSDIDenoiser()
		self.cond_drop_prob = cond_drop_prob

	@property
	def timesteps(self) -> int:
		return self.scheduler.timesteps

	@timesteps.setter
	def timesteps(self, new_steps: int) -> None:
		self.scheduler.timesteps = new_steps

	def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
		return torch.randint(0, self.timesteps, (batch_size,), device=device)

	def add_noise(
		self,
		x0: torch.Tensor,
		t: torch.Tensor,
		noise: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if noise is None:
			noise = torch.randn_like(x0)
		coeffs = self.scheduler.coefficients(t, like=x0)
		x_t = coeffs["sqrt_alpha_cumprod"] * x0 + coeffs["sqrt_one_minus_alpha_cumprod"] * noise
		return x_t, noise

	def add_noise_trajectory(self, x0: torch.Tensor) -> list[torch.Tensor]:
		trajectory = [x0]
		x_t = x0
		for step in range(self.timesteps):
			alpha_t = self.scheduler.alphas[step]
			beta_t = self.scheduler.betas[step]
			eps = torch.randn_like(x_t)
			x_t = torch.sqrt(alpha_t) * x_t + torch.sqrt(beta_t) * eps
			trajectory.append(x_t)
		return trajectory

	def predict_noise(
		self,
		x_t: torch.Tensor,
		t: torch.Tensor,
		condition: dict[str, torch.Tensor] | torch.Tensor | None,
		guidance_scale: float,
	) -> torch.Tensor:
		if condition is None or guidance_scale == 0.0:
			return self.denoiser(x_t, t, condition)

		eps_uncond = self.denoiser(x_t, t, None)
		eps_cond = self.denoiser(x_t, t, condition)
		return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

	def p_mean_variance(
		self,
		x_t: torch.Tensor,
		t: torch.Tensor,
		condition: dict[str, torch.Tensor] | torch.Tensor | None,
		guidance_scale: float,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		coeffs = self.scheduler.coefficients(t, like=x_t)
		eps_theta = self.predict_noise(x_t, t, condition=condition, guidance_scale=guidance_scale)

		x0_pred = (x_t - coeffs["sqrt_one_minus_alpha_cumprod"] * eps_theta) / coeffs["sqrt_alpha_cumprod"]
		model_mean = coeffs["posterior_mean_coef1"] * x0_pred + coeffs["posterior_mean_coef2"] * x_t
		return model_mean, coeffs["posterior_variance"], coeffs["posterior_log_variance"]

	@torch.no_grad()
	def denoise_step(
		self,
		x_t: torch.Tensor,
		t: torch.Tensor,
		condition: dict[str, torch.Tensor] | torch.Tensor | None = None,
		guidance_scale: float = 0.0,
	) -> torch.Tensor:
		mean, _, log_variance = self.p_mean_variance(
			x_t,
			t,
			condition=condition,
			guidance_scale=guidance_scale,
		)
		noise = torch.randn_like(x_t)
		nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
		return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

	@torch.no_grad()
	def sample(
		self,
		shape,
		condition: dict[str, torch.Tensor] | torch.Tensor | None = None,
		guidance_scale: float = 0.0,
		device: torch.device | None = None,
	) -> torch.Tensor:
		if device is None:
			device = next(self.parameters()).device
		x_t = torch.randn(*shape, device=device)
		batch_size = x_t.shape[0]
		for step in reversed(range(self.timesteps)):
			t = torch.full((batch_size,), step, device=device, dtype=torch.long)
			x_t = self.denoise_step(x_t, t, condition=condition, guidance_scale=guidance_scale)
		return x_t

	def drop_condition(self, condition: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
		mask = drop_mask.view(-1, *([1] * (condition.dim() - 1)))
		return torch.where(mask, torch.zeros_like(condition), condition)

	def drop_condition_any(
		self,
		condition: dict[str, torch.Tensor] | torch.Tensor,
		drop_mask: torch.Tensor,
	) -> dict[str, torch.Tensor] | torch.Tensor:
		x_co = condition.get("x_co")
		m_co = condition.get("m_co")
		if isinstance(x_co, torch.Tensor) and not isinstance(m_co, torch.Tensor):
			condition["m_co"] = torch.ones_like(x_co)

		dropped = {}
		for key, value in condition.items():
			if key in {"x_co", "m_co"} and isinstance(value, torch.Tensor):
				mask = drop_mask.view(-1, *([1] * (value.dim() - 1)))
				dropped[key] = torch.where(mask, torch.zeros_like(value), value)
			elif isinstance(value, torch.Tensor):
				dropped[key] = value
		return dropped

	def move_condition_to_device(
		self,
		condition: dict[str, torch.Tensor] | torch.Tensor,
		device: torch.device,
	) -> dict[str, torch.Tensor] | torch.Tensor:
		moved = {}
		for key, value in condition.items():
			if isinstance(value, torch.Tensor):
				moved[key] = value.to(device)
		return moved

	def masked_ddpm_loss(
		self,
		noise_pred: torch.Tensor,
		noise_target: torch.Tensor,
		condition: dict[str, torch.Tensor] | torch.Tensor | None,
	) -> torch.Tensor:
		del condition
		return F.mse_loss(noise_pred, noise_target)

	def train_ddpm(
		self,
		train_dataloader,
		val_dataloader,
		model_save_path: Path,
		optimizer: torch.optim.Optimizer,
		device: torch.device,
		epochs: int = 1,
	) -> list[float]:
		self.to(device)
		self.train()

		best_loss = float("inf")

		total_steps = epochs * len(train_dataloader)
		progress = tqdm(total=total_steps, desc="DDPM training", unit="batch")

		epoch_losses = []
		for epoch_idx in range(epochs):
			total_loss = 0.0
			batches = 0
			for batch in train_dataloader:
				x0 = batch.get("x0")
				condition = batch.get("condition")

				x0 = x0.to(device)
				condition = self.move_condition_to_device(condition, device)

				t = self.sample_timesteps(x0.shape[0], device=device)
				x_t, noise = self.add_noise(x0, t)

				drop_mask = torch.rand(x0.shape[0], device=device) < self.cond_drop_prob
				condition = self.drop_condition_any(condition, drop_mask)

				noise_pred = self.denoiser(x_t, t, condition)
				loss = self.masked_ddpm_loss(noise_pred, noise, condition)

				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()

				total_loss += float(loss.item())
				batches += 1
				progress.update(1)
				progress.set_postfix(
					loss=f"{loss.item():.6f}",
					epoch=f"{epoch_idx + 1}/{epochs}",
				)

			epoch_train_loss = total_loss / max(1, batches)

			val_loss_text = "n/a"
			epoch_val_loss = None
			if val_dataloader is not None:
				self.eval()
				val_total_loss = 0.0
				val_batches = 0
				with torch.no_grad():
					for batch in val_dataloader:
						x0 = batch.get("x0")
						condition = batch.get("condition")

						x0 = x0.to(device)
						condition = self.move_condition_to_device(condition, device)

						t = self.sample_timesteps(x0.shape[0], device=device)
						x_t, noise = self.add_noise(x0, t)
						noise_pred = self.denoiser(x_t, t, condition)
						val_loss = self.masked_ddpm_loss(noise_pred, noise, condition)

						val_total_loss += float(val_loss.item())
						val_batches += 1

				epoch_val_loss = val_total_loss / max(1, val_batches)
				val_loss_text = f"{epoch_val_loss:.6f}"
				self.train()

			comparison_loss = epoch_val_loss if epoch_val_loss is not None else epoch_train_loss
			if comparison_loss < best_loss:
				torch.save(self.state_dict(), model_save_path)
				best_loss = comparison_loss

			epoch_losses.append(epoch_train_loss)
			progress.set_postfix(
				train_loss=f"{epoch_train_loss:.6f}",
				val_loss=val_loss_text,
				epoch=f"{epoch_idx + 1}/{epochs}",
			)

		progress.close()
		return epoch_losses

	def financial_time_series_inference(
		self: DiffusionModel,
		sample: dict[str , torch.Tensor  | dict[str, torch.Tensor]],
		device,
		N: int,
		T_prime: int,
		corrector_steps: int,
		f_cutoff: float,
		eta: float = 0.01,
		s: int = 5,
		guidance_scale: float = 3.0,
		langevin_snr: float = 0.16,
		seq_dim: int = -1
    ) -> torch.Tensor:
		x_list = []
		K = int(N * T_prime / self.timesteps)

		for _ in range(s):
			x0 = sample.get("x0")
			condition = sample.get("condition")
			if not isinstance(x0, torch.Tensor):
				raise ValueError("sample['x0'] must be a torch.Tensor")
			if not isinstance(condition, dict) or "x_co" not in condition:
				raise ValueError("sample['condition'] must be a dict containing key 'x_co'")

			x0 = x0.to(device)
			condition = self.move_condition_to_device(condition, device)
			batch_size = x0.shape[0]

			T_prime_vector = torch.full((1,), T_prime, device=device, dtype=torch.long)

			x_K, _ = self.add_noise(x0, T_prime_vector)
			x_i = x_K

			for i in range(K-1, 0, -1):
				t_i = torch.full((1,), (i+1) * T_prime / N, device=device, dtype=torch.long)
				x_i = self.denoise_step(x_i, t_i, condition=condition, guidance_scale=guidance_scale)

				for j in range(1, corrector_steps):
					coeffs = self.scheduler.coefficients(t_i, x_i)
					sigma_j = coeffs["sqrt_one_minus_alpha_cumprod"]

					eps_theta = self.predict_noise(x_i, t_i, condition=condition, guidance_scale=guidance_scale)

					alpha_j = (langevin_snr * sigma_j) ** 2

					z = torch.rand_like(x_i)

					x_i = x_i - (alpha_j / (2 * sigma_j)) * eps_theta + torch.sqrt(alpha_j) * z

				with torch.enable_grad():
					x_in = x_i.detach().requires_grad_(True)
					
					loss_tv = total_variation_loss(x_in, seq_dim=seq_dim)
					loss_f = fourier_loss(x_in, condition["x_co"], f=f_cutoff, seq_dim=seq_dim)
					
					total_loss = (eta * loss_tv) + (eta * loss_f)
					
					grad_x = torch.autograd.grad(total_loss, x_in)[0]

				x_i = x_i - grad_x

			x_list.append(x_i)

		return torch.stack(x_list, dim=0).mean(dim=0)
     
					
       
      



     

