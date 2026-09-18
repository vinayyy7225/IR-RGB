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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.realesrgan.model import RRDBNet
from utils.dataset import LandsatDataset

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
    
    # Model
    nf = config["models"]["realesrgan"]["num_filters"]
    nb = config["models"]["realesrgan"]["num_blocks"]
    model = RRDBNet(in_nc=2, out_nc=2, nf=nf, nb=nb, gc=32, upscale=scale).to(device)
    
    # Optimizer and Loss
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.999))
    
    epochs = config["training"]["epochs_sr"]
    save_freq = config["training"]["save_epoch_freq"]
    
    # Learning rate scheduler (Cosine Decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Automatic Mixed Precision (AMP)
    use_amp = use_cuda and hasattr(torch, "amp")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Automatic Mixed Precision (AMP - FP16) active.")
    
    start_epoch = 1
    # Auto-resume logic: supports both realesrgan_epoch_*.pth and realesrgan_checkpoint_epoch_*.pth
    combined_files = glob.glob(os.path.join(checkpoints_dir, "realesrgan_checkpoint_epoch_*.pth"))
    epoch_files = glob.glob(os.path.join(checkpoints_dir, "realesrgan_epoch_*.pth"))
    
    if combined_files:
        try:
            combined_files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))
            latest = combined_files[-1]
            checkpoint = torch.load(latest, map_location=device)
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ Found combined checkpoint: {latest}. Resuming Real-ESRGAN from epoch {start_epoch}...")
        except Exception as e:
            print(f"Notice: Could not load {combined_files[-1]}: {e}")
            
    elif epoch_files:
        try:
            epoch_files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))
            latest = epoch_files[-1]
            epoch_num = int(latest.split("_epoch_")[-1].split(".pth")[0])
            weights = torch.load(latest, map_location=device)
            if isinstance(weights, dict) and "model" in weights:
                model.load_state_dict(weights["model"])
            else:
                model.load_state_dict(weights)
            start_epoch = epoch_num + 1
            print(f"✅ Found weights checkpoint: {latest}. Resuming Real-ESRGAN from epoch {start_epoch}...")
        except Exception as e:
            print(f"Notice: Could not load {epoch_files[-1]}: {e}")
            
    if start_epoch > epochs:
        print(f"🎉 Real-ESRGAN has ALREADY completed all {epochs} epochs (Latest epoch was {start_epoch - 1})!")
        print(f"   Model weights are ready at: {os.path.join(checkpoints_dir, 'realesrgan_best.pth')}")
        print("   If you want to train further, increase 'epochs_sr' in configs/config.yaml.")
        return
        
    print(f"Starting Real-ESRGAN training from epoch {start_epoch} to {epochs}...")
    best_loss = float("inf")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            lr_ir = batch["ir_lr"].to(device, non_blocking=True)
            hr_ir = batch["ir_hr"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    sr_ir = model(lr_ir)
                    loss = criterion(sr_ir, hr_ir)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr_ir = model(lr_ir)
                loss = criterion(sr_ir, hr_ir)
                loss.backward()
                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.5f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        avg_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr_ir = batch["ir_lr"].to(device, non_blocking=True)
                hr_ir = batch["ir_hr"].to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        sr_ir = model(lr_ir)
                        loss = criterion(sr_ir, hr_ir)
                else:
                    sr_ir = model(lr_ir)
                    loss = criterion(sr_ir, hr_ir)
                avg_val_loss += loss.item()
                
        if len(val_loader) > 0:
            avg_val_loss = avg_val_loss / len(val_loader)
        else:
            avg_val_loss = avg_train_loss
            
        print(f"Epoch {epoch} finished - Train Loss: {avg_train_loss:.5f} - Val Loss: {avg_val_loss:.5f}")
        
        # Save checkpoints directly to Drive
        if epoch % save_freq == 0 or epoch == epochs:
            # 1. State dict format (matches your uploaded format)
            state_dict_path = os.path.join(checkpoints_dir, f"realesrgan_epoch_{epoch}.pth")
            torch.save(model.state_dict(), state_dict_path)
            
            # 2. Resume dictionary format
            full_checkpoint_path = os.path.join(checkpoints_dir, f"realesrgan_checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, full_checkpoint_path)
            print(f"💾 Saved checkpoint directly to Drive: {state_dict_path}")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_checkpoint_path = os.path.join(checkpoints_dir, "realesrgan_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"🌟 Saved new best model directly to Drive: {best_checkpoint_path} (Val Loss: {best_loss:.5f})")
            
    print("Real-ESRGAN training finished!")

if __name__ == "__main__":
    main()
