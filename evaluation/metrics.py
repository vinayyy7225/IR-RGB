import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def calculate_psnr(gen_rgb, gt_rgb):
    """
    Calculate PSNR supporting 2D, 3D (H,W,C / C,H,W), and 4D (B,C,H,W) tensors or numpy arrays.
    gen_rgb: Generated RGB image tensor/numpy array in range [0, 1]
    gt_rgb: Ground truth RGB image tensor/numpy array in range [0, 1]
    """
    if isinstance(gen_rgb, torch.Tensor):
        gen_rgb = gen_rgb.detach().cpu().numpy()
    if isinstance(gt_rgb, torch.Tensor):
        gt_rgb = gt_rgb.detach().cpu().numpy()
        
    gen_rgb = np.clip(gen_rgb, 0.0, 1.0)
    gt_rgb = np.clip(gt_rgb, 0.0, 1.0)
    
    # For 4D batch tensors (B, C, H, W), compute mean PSNR over the batch
    if gen_rgb.ndim == 4:
        psnrs = [calculate_psnr(gen_rgb[i], gt_rgb[i]) for i in range(len(gen_rgb))]
        return float(np.mean(psnrs))
        
    # Standardize 3D to (H, W, 3)
    if gen_rgb.ndim == 3 and gen_rgb.shape[0] in [1, 3] and gen_rgb.shape[2] not in [1, 3]:
        gen_rgb = gen_rgb.transpose(1, 2, 0)
    if gt_rgb.ndim == 3 and gt_rgb.shape[0] in [1, 3] and gt_rgb.shape[2] not in [1, 3]:
        gt_rgb = gt_rgb.transpose(1, 2, 0)
        
    return float(psnr_func(gt_rgb, gen_rgb, data_range=1.0))

def calculate_ssim(gen_rgb, gt_rgb):
    """
    Calculate SSIM supporting 2D, 3D (H,W,C / C,H,W), and 4D (B,C,H,W) tensors or numpy arrays.
    gen_rgb: Generated RGB image, range [0, 1]
    gt_rgb: Ground truth RGB image, range [0, 1]
    """
    if isinstance(gen_rgb, torch.Tensor):
        gen_rgb = gen_rgb.detach().cpu().numpy()
    if isinstance(gt_rgb, torch.Tensor):
        gt_rgb = gt_rgb.detach().cpu().numpy()
        
    gen_rgb = np.clip(gen_rgb, 0.0, 1.0)
    gt_rgb = np.clip(gt_rgb, 0.0, 1.0)
    
    # For 4D batch tensors (B, C, H, W)
    if gen_rgb.ndim == 4:
        ssims = [calculate_ssim(gen_rgb[i], gt_rgb[i]) for i in range(len(gen_rgb))]
        return float(np.mean(ssims))
        
    # For 2D unstructured arrays (e.g. flat masked pixels)
    if gen_rgb.ndim == 2:
        rmse = np.sqrt(np.mean((gen_rgb - gt_rgb) ** 2))
        return float(max(0.0, 1.0 - rmse))
        
    # Standardize 3D to (H, W, 3)
    if gen_rgb.ndim == 3 and gen_rgb.shape[0] in [1, 3] and gen_rgb.shape[2] not in [1, 3]:
        gen_rgb = gen_rgb.transpose(1, 2, 0)
    if gt_rgb.ndim == 3 and gt_rgb.shape[0] in [1, 3] and gt_rgb.shape[2] not in [1, 3]:
        gt_rgb = gt_rgb.transpose(1, 2, 0)
        
    channel_axis = 2 if (gen_rgb.ndim == 3 and gen_rgb.shape[2] in [1, 3]) else None
    try:
        if channel_axis is not None:
            return float(ssim_func(gt_rgb, gen_rgb, channel_axis=channel_axis, data_range=1.0))
        else:
            return float(ssim_func(gt_rgb, gen_rgb, data_range=1.0))
    except TypeError:
        # Fallback for older skimage versions
        if gen_rgb.ndim == 3:
            return float(np.mean([
                ssim_func(gt_rgb[:, :, i], gen_rgb[:, :, i], data_range=1.0)
                for i in range(gen_rgb.shape[2])
            ]))
        return float(ssim_func(gt_rgb, gen_rgb, data_range=1.0))

def calculate_fid_from_features(mu1, sigma1, mu2, sigma2):
    """
    Calculate Fréchet Inception Distance between two Gaussian distributions defined by (mu1, sigma1) and (mu2, sigma2).
    """
    import scipy.linalg
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def main():
    print("Testing robust metrics calculation...")
    dummy_gen = torch.rand(3, 256, 256)
    dummy_gt = torch.clamp(dummy_gen + torch.randn(3, 256, 256) * 0.05, 0.0, 1.0)
    
    psnr_val = calculate_psnr(dummy_gen, dummy_gt)
    ssim_val = calculate_ssim(dummy_gen, dummy_gt)
    
    print(f"3D Test PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")
    
    # 4D Batch Test
    dummy_gen_4d = torch.rand(4, 3, 256, 256)
    dummy_gt_4d = torch.clamp(dummy_gen_4d + torch.randn(4, 3, 256, 256) * 0.05, 0.0, 1.0)
    print(f"4D Batch Test PSNR: {calculate_psnr(dummy_gen_4d, dummy_gt_4d):.4f} dB")
    
    # 2D Masked Test
    dummy_gen_2d = torch.rand(500, 3)
    dummy_gt_2d = torch.clamp(dummy_gen_2d + torch.randn(500, 3) * 0.05, 0.0, 1.0)
    print(f"2D Masked Test PSNR: {calculate_psnr(dummy_gen_2d, dummy_gt_2d):.4f} dB")

if __name__ == "__main__":
    main()
