import torch
import warnings

def total_variation_loss(x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
    diff = torch.diff(x, dim=seq_dim)
    
    tv_loss = torch.sum(torch.abs(diff))
    
    return tv_loss



def fourier_loss(x_t: torch.Tensor, x: torch.Tensor, f: float, seq_dim: int = -1) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="An output with one or more elements was resized")
        fft_x_t = torch.fft.rfft(x_t, dim=seq_dim)
        fft_x = torch.fft.rfft(x, dim=seq_dim)
    
    amplitude = torch.abs(fft_x)

    filtered_fft_x = fft_x.clone()

    low_amplitude_mask = amplitude < f
    
    filtered_fft_x[low_amplitude_mask] = 0.0 + 0.0j
    
    diff = torch.abs(fft_x_t - filtered_fft_x)
    fourier_loss = torch.sum(diff ** 2)
    
    return fourier_loss