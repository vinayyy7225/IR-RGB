import os
import sys
import glob
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pix2pix.model import UNetGenerator, PatchGANDiscriminator
from models.semantic.semantic_loss import SemanticIndexLoss
from utils.dataset import LandsatDataset
from evaluation.metrics import calculate_psnr, calculate_ssim

def load_config(config_path="configs/config.yaml"):
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        config_file = Path(config_path)
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device, False
        
    if "COLAB_TPU_ADDR" in os.environ or "TPU_NAME" in os.environ:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Using TPU device: {device}")
            return device, True
        except Exception as e:
            print(f"TPU environment detected but initialization failed ({e}). Falling back.")
            
    device = torch.device("cpu")
    print("Using CPU device")
    return device, False

def save_sample_grid(real_ir, fake_rgb, real_rgb, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    b = min(4, real_ir.size(0))
    rows = []
    for i in range(b):
        t10 = real_ir[i, 0].detach().cpu().numpy()
        t10_vis = np.stack([t10, t10, t10], axis=-1)
        t10_vis = np.clip(t10_vis * 255.0, 0, 255).astype(np.uint8)
        
        gen = (fake_rgb[i].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5
        gen_vis = np.clip(gen * 255.0, 0, 255).astype(np.uint8)
        
        gt = (real_rgb[i].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5
        gt_vis = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
        
        row = np.concatenate([t10_vis, gen_vis, gt_vis], axis=1)
        rows.append(row)
        
    full_grid = np.concatenate(rows, axis=0)
    Image.fromarray(full_grid).save(save_path)

def main():
    config = load_config()
    device, is_tpu = get_device()
    use_cuda = (device.type == "cuda")
    
    # Paths
    processed_dir = config["data"]["processed_dir"]
    if not os.path.isabs(processed_dir):
        processed_dir = str(PROJECT_ROOT / processed_dir)
        
    checkpoints_dir = config["training"]["checkpoints_dir"]
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = str(PROJECT_ROOT / checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Checkpoints directory (saving directly to Drive): {checkpoints_dir}")
    
    outputs_dir = config["training"].get("outputs_dir", "outputs")
    if not os.path.isabs(outputs_dir):
        outputs_dir = str(PROJECT_ROOT / outputs_dir)
    samples_dir = os.path.join(outputs_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Dataset and Dataloader
    scale = config["models"]["realesrgan"]["scale"]
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    
    train_dataset = LandsatDataset(train_dir, is_train=True, scale_factor=scale)
    val_dataset = LandsatDataset(val_dir, is_train=False, scale_factor=scale)
    
    if len(train_dataset) == 0:
        print(f"Error: Train dataset is empty at {train_dir}. Please run preprocessing first.")
        return
        
    batch_size = config["training"]["batch_size"]
    num_workers = 2 if (use_cuda and os.name != "nt") else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    print(f"Loaded {len(train_dataset)} train patches and {len(val_dataset)} val patches.")
    
    # Models
    net_g = UNetGenerator(input_nc=2, output_nc=3, num_downs=8, ngf=64).to(device)
    net_d = PatchGANDiscriminator(input_nc=5, ndf=64).to(device)
    
    # Loss functions
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_semantic = SemanticIndexLoss().to(device)
    
    # Optimizers
    lr = config["training"]["lr"]
    beta1 = config["training"]["beta1"]
    beta2 = config["training"]["beta2"]
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, beta2))
    
    lambda_l1 = config["models"]["pix2pix"]["lambda_l1"]
    lambda_semantic = config["models"]["pix2pix"]["lambda_semantic"]
    epochs = config["training"]["epochs_pix2pix"]
    save_freq = config["training"]["save_epoch_freq"]
    
    def lr_lambda(epoch):
        decay_start = epochs // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / float(epochs - decay_start + 1)
        
    scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_lambda)
    scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda)
    
    use_amp = use_cuda and hasattr(torch, "amp")
    scaler_g = torch.amp.GradScaler("cuda") if use_amp else None
    scaler_d = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Automatic Mixed Precision (AMP - FP16) active.")
    
    start_epoch = 1
    # Auto-resume logic: supports both combined and separate gen/disc files
    combined_files = glob.glob(os.path.join(checkpoints_dir, "pix2pix_checkpoint_epoch_*.pth"))
    gen_files = glob.glob(os.path.join(checkpoints_dir, "pix2pix_gen_epoch_*.pth"))
    
    if combined_files:
        try:
            combined_files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))
            latest = combined_files[-1]
            checkpoint = torch.load(latest, map_location=device)
            net_g.load_state_dict(checkpoint['generator'])
            net_d.load_state_dict(checkpoint['discriminator'])
            if 'g_optimizer' in checkpoint: optimizer_g.load_state_dict(checkpoint['g_optimizer'])
            if 'd_optimizer' in checkpoint: optimizer_d.load_state_dict(checkpoint['d_optimizer'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ Found combined checkpoint: {latest}. Resuming Pix2Pix from epoch {start_epoch}...")
        except Exception as e:
            print(f"Notice: Could not load {combined_files[-1]}: {e}")
            
    elif gen_files:
        try:
            gen_files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))
            latest_gen = gen_files[-1]
            epoch_num = int(latest_gen.split("_epoch_")[-1].split(".pth")[0])
            latest_disc = latest_gen.replace("pix2pix_gen_epoch_", "pix2pix_disc_epoch_")
            
            # Load generator weights
            g_weights = torch.load(latest_gen, map_location=device)
            if isinstance(g_weights, dict) and "generator" in g_weights:
                net_g.load_state_dict(g_weights["generator"])
            else:
                net_g.load_state_dict(g_weights)
                
            # Load discriminator weights if available
            if os.path.exists(latest_disc):
                d_weights = torch.load(latest_disc, map_location=device)
                if isinstance(d_weights, dict) and "discriminator" in d_weights:
                    net_d.load_state_dict(d_weights["discriminator"])
                else:
                    net_d.load_state_dict(d_weights)
                    
            start_epoch = epoch_num + 1
            print(f"✅ Found uploaded weights: {latest_gen} (Epoch {epoch_num}). Resuming Pix2Pix from epoch {start_epoch}...")
        except Exception as e:
            print(f"Notice: Could not load {gen_files[-1]}: {e}")
            
    if start_epoch > epochs:
        print(f"🎉 Pix2Pix Colorization has ALREADY completed all {epochs} epochs (Latest epoch was {start_epoch - 1})!")
        print(f"   Model weights are ready at: {os.path.join(checkpoints_dir, 'pix2pix_gen_best.pth')}")
        print("   If you want to train further, increase 'epochs_pix2pix' in configs/config.yaml.")
        return
        
    print(f"Starting Pix2Pix Colorization training from epoch {start_epoch} to {epochs}...")
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, epochs + 1):
        net_g.train()
        net_d.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            real_ir = batch["ir_hr"].to(device, non_blocking=True)
            real_rgb = batch["rgb"].to(device, non_blocking=True)
            nir = batch["nir"].to(device, non_blocking=True)
            
            # --- 1. Train Discriminator ---
            optimizer_d.zero_grad()
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    fake_rgb = net_g(real_ir)
                    real_pair = torch.cat((real_ir, real_rgb), dim=1)
                    pred_real = net_d(real_pair)
                    loss_d_real = criterion_gan(pred_real, torch.full_like(pred_real, 0.9))
                    
                    fake_pair = torch.cat((real_ir, fake_rgb.detach()), dim=1)
                    pred_fake = net_d(fake_pair)
                    loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                    
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                fake_rgb = net_g(real_ir)
                real_pair = torch.cat((real_ir, real_rgb), dim=1)
                pred_real = net_d(real_pair)
                loss_d_real = criterion_gan(pred_real, torch.full_like(pred_real, 0.9))
                
                fake_pair = torch.cat((real_ir, fake_rgb.detach()), dim=1)
                pred_fake = net_d(fake_pair)
                loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer_d, barrier=True)
                else:
                    optimizer_d.step()
            
            # --- 2. Train Generator ---
            optimizer_g.zero_grad()
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    fake_pair_for_g = torch.cat((real_ir, fake_rgb), dim=1)
                    pred_fake_for_g = net_d(fake_pair_for_g)
                    loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
                    loss_g_l1 = criterion_l1(fake_rgb, real_rgb)
                    loss_g_semantic = criterion_semantic(fake_rgb, real_rgb, nir)
                    loss_g = loss_g_gan + (lambda_l1 * loss_g_l1) + (lambda_semantic * loss_g_semantic)
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                fake_pair_for_g = torch.cat((real_ir, fake_rgb), dim=1)
                pred_fake_for_g = net_d(fake_pair_for_g)
                loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
                loss_g_l1 = criterion_l1(fake_rgb, real_rgb)
                loss_g_semantic = criterion_semantic(fake_rgb, real_rgb, nir)
                loss_g = loss_g_gan + (lambda_l1 * loss_g_l1) + (lambda_semantic * loss_g_semantic)
                loss_g.backward()
                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer_g, barrier=True)
                else:
                    optimizer_g.step()
            
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            loop.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}", L1=f"{loss_g_l1.item():.4f}")
            
        scheduler_g.step()
        scheduler_d.step()
        
        avg_train_g_loss = epoch_g_loss / len(train_loader)
        avg_train_d_loss = epoch_d_loss / len(train_loader)
        
        # --- 3. Validation Loop ---
        net_g.eval()
        val_l1_total = 0.0
        val_psnr_total = 0.0
        sample_saved = False
        
        with torch.no_grad():
            for val_batch in val_loader:
                v_ir = val_batch["ir_hr"].to(device, non_blocking=True)
                v_rgb = val_batch["rgb"].to(device, non_blocking=True)
                v_rgb_01 = val_batch["rgb_01"].to(device, non_blocking=True)
                
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        v_fake = net_g(v_ir)
                else:
                    v_fake = net_g(v_ir)
                    
                v_l1 = criterion_l1(v_fake, v_rgb)
                val_l1_total += v_l1.item()
                
                v_fake_01 = torch.clamp((v_fake + 1.0) * 0.5, 0.0, 1.0)
                val_psnr_total += calculate_psnr(v_fake_01, v_rgb_01)
                
                if not sample_saved:
                    sample_path = os.path.join(samples_dir, f"epoch_{epoch}_sample.png")
                    save_sample_grid(v_ir, v_fake, v_rgb, sample_path)
                    sample_saved = True
                    
        num_val = max(1, len(val_loader))
        avg_val_l1 = val_l1_total / num_val
        avg_val_psnr = val_psnr_total / num_val
        
        print(f"Epoch {epoch} finished - Train G Loss: {avg_train_g_loss:.4f} - Val L1: {avg_val_l1:.4f} - Val PSNR: {avg_val_psnr:.2f} dB")
        
        # Save checkpoints directly to Drive
        if epoch % save_freq == 0 or epoch == epochs:
            # 1. Separate Generator and Discriminator state dicts (matching existing files)
            gen_path = os.path.join(checkpoints_dir, f"pix2pix_gen_epoch_{epoch}.pth")
            disc_path = os.path.join(checkpoints_dir, f"pix2pix_disc_epoch_{epoch}.pth")
            torch.save(net_g.state_dict(), gen_path)
            torch.save(net_d.state_dict(), disc_path)
            
            # 2. Combined full dictionary checkpoint
            combined_path = os.path.join(checkpoints_dir, f"pix2pix_checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'generator': net_g.state_dict(),
                'discriminator': net_d.state_dict(),
                'g_optimizer': optimizer_g.state_dict(),
                'd_optimizer': optimizer_d.state_dict(),
                'val_l1': avg_val_l1,
                'val_psnr': avg_val_psnr,
            }, combined_path)
            print(f"💾 Saved checkpoints directly to Drive: {gen_path}")
            
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            best_checkpoint_path = os.path.join(checkpoints_dir, "pix2pix_gen_best.pth")
            torch.save(net_g.state_dict(), best_checkpoint_path)
            print(f"🌟 Saved new best Generator directly to Drive: {best_checkpoint_path} (Val L1: {best_val_loss:.4f}, PSNR: {avg_val_psnr:.2f} dB)")
            
    print("Pix2Pix training finished successfully!")

if __name__ == "__main__":
    main()
