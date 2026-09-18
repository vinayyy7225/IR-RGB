import os
import sys
import glob
import yaml
from pathlib import Path
import torch
import numpy as np
import rasterio
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.realesrgan.model import RRDBNet
from models.pix2pix.model import UNetGenerator
from evaluation.metrics import calculate_psnr, calculate_ssim

def load_config(config_path="configs/config.yaml"):
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        config_file = Path(config_path)
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def find_best_checkpoint(checkpoints_dir, prefix):
    """Find best checkpoint, or latest epoch checkpoint, or fallback."""
    best_chk = os.path.join(checkpoints_dir, f"{prefix}_best.pth")
    if os.path.exists(best_chk):
        return best_chk
        
    pattern = os.path.join(checkpoints_dir, f"{prefix}*.pth")
    all_files = glob.glob(pattern)
    if not all_files:
        return None
        
    def extract_epoch(fname):
        try:
            if "_epoch_" in fname:
                return int(fname.split("_epoch_")[-1].split(".pth")[0])
        except Exception:
            pass
        return 0
        
    all_files.sort(key=extract_epoch)
    return all_files[-1]

def pad_image(arr, patch_size):
    """Pad image array so its dimensions are multiples of patch_size."""
    h, w = arr.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if arr.ndim == 3:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='edge')
    return padded, pad_h, pad_w

def get_percentile_min_max(band_arr, p_min=2, p_max=98):
    valid_pixels = band_arr[band_arr > 0]
    if len(valid_pixels) == 0:
        return 0.0, 1.0
    return np.percentile(valid_pixels, p_min), np.percentile(valid_pixels, p_max)

def normalize_band(band_arr, b_min, b_max):
    if b_max - b_min < 1e-5:
        return np.zeros_like(band_arr, dtype=np.float32)
    normalized = (band_arr.astype(np.float32) - b_min) / (b_max - b_min)
    return np.clip(normalized, 0.0, 1.0)

def generate_zoomable_viewer_html(image_filename, output_html_path, title="ISRO Ultra HD Satellite Colorization Viewer"):
    """
    Generate a standalone, zero-dependency interactive Pan-and-Zoom HTML viewer
    that loads the Ultra HD output image and lets the user zoom down to pixel level.
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background-color: #0b0d14;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        #header {{
            position: absolute;
            top: 15px;
            left: 20px;
            z-index: 100;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 12px 20px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        }}
        #header h1 {{
            font-size: 1.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        #header p {{
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 2px;
        }}
        #controls {{
            position: absolute;
            bottom: 25px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 16px;
            border-radius: 30px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        }}
        button {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #ffffff;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            font-size: 1.2rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }}
        button:hover {{
            background: #00f2fe;
            color: #0b0d14;
            transform: scale(1.08);
        }}
        #zoomLevel {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #38bdf8;
            min-width: 60px;
            text-align: center;
        }}
        #viewport {{
            width: 100%;
            height: 100%;
            cursor: grab;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        #viewport:active {{
            cursor: grabbing;
        }}
        #targetImage {{
            transform-origin: center center;
            user-select: none;
            -webkit-user-drag: none;
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            image-rendering: auto;
            transition: transform 0.05s ease-out;
            box-shadow: 0 20px 50px rgba(0,0,0,0.8);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🛰️ ISRO Ultra HD Deep Zoom Viewer</h1>
        <p>Scroll or pinch to Zoom • Click & Drag to Pan • Full Pixel Fidelity</p>
    </div>
    
    <div id="viewport">
        <img id="targetImage" src="{image_filename}" alt="Ultra HD Colorized Satellite Image">
    </div>

    <div id="controls">
        <button id="zoomOut" title="Zoom Out">−</button>
        <span id="zoomLevel">100%</span>
        <button id="zoomIn" title="Zoom In">+</button>
        <button id="resetView" style="width: auto; padding: 0 14px; border-radius: 20px; font-size: 0.8rem;" title="Reset View">Reset</button>
        <button id="pixelCrisp" style="width: auto; padding: 0 14px; border-radius: 20px; font-size: 0.8rem;" title="Toggle Crisp Pixels">Crisp</button>
    </div>

    <script>
        const img = document.getElementById('targetImage');
        const viewport = document.getElementById('viewport');
        const zoomLevelText = document.getElementById('zoomLevel');
        
        let scale = 1;
        let panning = false;
        let pointX = 0;
        let pointY = 0;
        let startX = 0;
        let startY = 0;
        let isPixelated = false;

        function updateTransform() {{
            img.style.transform = `translate(${{pointX}}px, ${{pointY}}px) scale(${{scale}})`;
            zoomLevelText.textContent = `${{Math.round(scale * 100)}}%`;
        }}

        viewport.onmousedown = function (e) {{
            e.preventDefault();
            startX = e.clientX - pointX;
            startY = e.clientY - pointY;
            panning = true;
        }};

        document.onmouseup = function () {{
            panning = false;
        }};

        document.onmousemove = function (e) {{
            e.preventDefault();
            if (!panning) return;
            pointX = e.clientX - startX;
            pointY = e.clientY - startY;
            updateTransform();
        }};

        viewport.onwheel = function (e) {{
            e.preventDefault();
            const xs = (e.clientX - pointX) / scale;
            const ys = (e.clientY - pointY) / scale;
            const delta = -e.deltaY;
            
            if (delta > 0) {{
                scale *= 1.15;
            }} else {{
                scale /= 1.15;
            }}
            scale = Math.min(Math.max(0.1, scale), 30.0); // up to 3000% zoom!
            
            pointX = e.clientX - xs * scale;
            pointY = e.clientY - ys * scale;
            updateTransform();
        }};

        document.getElementById('zoomIn').onclick = () => {{
            scale = Math.min(scale * 1.3, 30.0);
            updateTransform();
        }};

        document.getElementById('zoomOut').onclick = () => {{
            scale = Math.max(scale / 1.3, 0.1);
            updateTransform();
        }};

        document.getElementById('resetView').onclick = () => {{
            scale = 1;
            pointX = 0;
            pointY = 0;
            updateTransform();
        }};

        document.getElementById('pixelCrisp').onclick = () => {{
            isPixelated = !isPixelated;
            img.style.imageRendering = isPixelated ? 'pixelated' : 'auto';
            document.getElementById('pixelCrisp').style.background = isPixelated ? '#00f2fe' : 'rgba(255,255,255,0.08)';
            document.getElementById('pixelCrisp').style.color = isPixelated ? '#0b0d14' : '#ffffff';
        }};
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

@torch.no_grad()
def run_inference(scene_dir, model_sr_path=None, model_pix2pix_path=None, output_path=None, config=None):
    if config is None:
        config = load_config()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = config["data"]["patch_size"]
    scale = config["models"]["realesrgan"]["scale"]
    checkpoints_dir = config["training"]["checkpoints_dir"]
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = str(PROJECT_ROOT / checkpoints_dir)
        
    # Auto-resolve checkpoint paths if not explicitly provided
    if model_sr_path is None or not os.path.exists(model_sr_path):
        model_sr_path = find_best_checkpoint(checkpoints_dir, "realesrgan")
        
    if model_pix2pix_path is None or not os.path.exists(model_pix2pix_path):
        model_pix2pix_path = find_best_checkpoint(checkpoints_dir, "pix2pix_gen")
        if model_pix2pix_path is None:
            model_pix2pix_path = find_best_checkpoint(checkpoints_dir, "pix2pix")
            
    # Default output path
    if output_path is None:
        outputs_dir = config["training"].get("outputs_dir", "outputs")
        if not os.path.isabs(outputs_dir):
            outputs_dir = str(PROJECT_ROOT / outputs_dir)
        os.makedirs(outputs_dir, exist_ok=True)
        scene_base = os.path.basename(scene_dir.rstrip("/\\"))
        output_path = os.path.join(outputs_dir, f"{scene_base}_ultra_hd_rgb.TIF")
        
    print(f"Using SR checkpoint: {model_sr_path}")
    print(f"Using Pix2Pix checkpoint: {model_pix2pix_path}")
    
    # Identify bands in the scene
    b2_files = glob.glob(os.path.join(scene_dir, "*_B2.TIF"))
    if not b2_files:
        print(f"Error: B2 band not found in scene {scene_dir}")
        return None
    base_path = b2_files[0].replace("_B2.TIF", "")
    
    paths = {
        "B2": f"{base_path}_B2.TIF",
        "B3": f"{base_path}_B3.TIF",
        "B4": f"{base_path}_B4.TIF",
        "B10": f"{base_path}_B10.TIF",
        "B11": f"{base_path}_B11.TIF",
    }
    
    # Load and normalize full scene bands
    bands_data = {}
    original_profile = None
    for b, p in paths.items():
        if not os.path.exists(p):
            print(f"Error: Band file missing: {p}")
            return None
        with rasterio.open(p) as src:
            bands_data[b] = src.read(1)
            if b == "B2":
                original_profile = src.profile.copy()
                
    h_orig, w_orig = bands_data["B2"].shape
    print(f"Processing Scene dimensions: {w_orig} x {h_orig} pixels (Ultra HD)")
    
    # Normalize RGB (for ground truth comparison)
    gt_rgb = []
    for b in ["B4", "B3", "B2"]:
        b_min, b_max = get_percentile_min_max(bands_data[b])
        gt_rgb.append(normalize_band(bands_data[b], b_min, b_max))
    gt_rgb = np.stack(gt_rgb, axis=-1) # (H, W, 3) range [0, 1]
    
    # Normalize Thermals
    norm_thermals = []
    for b in ["B10", "B11"]:
        b_min, b_max = get_percentile_min_max(bands_data[b])
        norm_thermals.append(normalize_band(bands_data[b], b_min, b_max))
    norm_thermals = np.stack(norm_thermals, axis=-1) # (H, W, 2) range [0, 1]
    
    # Pad images for grid processing
    padded_thermals, pad_h, pad_w = pad_image(norm_thermals, patch_size)
    h_pad, w_pad, _ = padded_thermals.shape
    
    # Instantiate models
    nf_sr = config["models"]["realesrgan"]["num_filters"]
    nb_sr = config["models"]["realesrgan"]["num_blocks"]
    net_sr = RRDBNet(in_nc=2, out_nc=2, nf=nf_sr, nb=nb_sr, gc=32, upscale=scale).to(device)
    if model_sr_path and os.path.exists(model_sr_path):
        weights = torch.load(model_sr_path, map_location=device)
        if "model" in weights:
            weights = weights["model"]
        net_sr.load_state_dict(weights)
        print(f"Loaded Real-ESRGAN weights from {model_sr_path}")
    else:
        print("Warning: Real-ESRGAN weights not found! Running with initial weights.")
    net_sr.eval()
    
    net_g = UNetGenerator(input_nc=2, output_nc=3, num_downs=8, ngf=64).to(device)
    if model_pix2pix_path and os.path.exists(model_pix2pix_path):
        weights_g = torch.load(model_pix2pix_path, map_location=device)
        if "generator" in weights_g:
            weights_g = weights_g["generator"]
        net_g.load_state_dict(weights_g)
        print(f"Loaded Pix2Pix weights from {model_pix2pix_path}")
    else:
        print("Warning: Pix2Pix weights not found! Running with initial weights.")
    net_g.eval()
    
    # Overlap parameters: 50% stride for seamless blending
    stride = patch_size // 2
    
    # Create 2D Hann window for feather blending
    window_1d = np.hanning(patch_size)
    window_2d = np.outer(window_1d, window_1d).astype(np.float32)
    window_rgb = np.expand_dims(window_2d, axis=-1)
    
    stitched_rgb = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    weight_sum = np.zeros((h_pad, w_pad, 1), dtype=np.float32)
    
    print("Running overlapping patch-wise inference with Hann feathering...")
    
    y_steps = range(0, h_pad - patch_size + 1, stride)
    x_steps = range(0, w_pad - patch_size + 1, stride)
    total_patches = len(y_steps) * len(x_steps)
    
    count = 0
    for y in y_steps:
        for x in x_steps:
            patch_ir = padded_thermals[y : y + patch_size, x : x + patch_size]
            patch_ir_tensor = torch.from_numpy(patch_ir).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # Super-resolution
            lr_ir_tensor = torch.nn.functional.interpolate(
                patch_ir_tensor, 
                size=(patch_size // scale, patch_size // scale), 
                mode='bilinear', 
                align_corners=False
            )
            sr_ir_tensor = torch.clamp(net_sr(lr_ir_tensor), 0.0, 1.0)
            
            # Colorization
            fake_rgb_tensor = net_g(sr_ir_tensor)
            fake_rgb_tensor = torch.clamp((fake_rgb_tensor + 1.0) * 0.5, 0.0, 1.0)
            
            rgb_out = fake_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            stitched_rgb[y : y + patch_size, x : x + patch_size] += rgb_out * window_rgb
            weight_sum[y : y + patch_size, x : x + patch_size] += window_rgb
            
            count += 1
            if count % 100 == 0 or count == total_patches:
                print(f"Processed {count}/{total_patches} patches ({(count/total_patches)*100:.1f}%)")
                
    weight_sum[weight_sum == 0] = 1e-8
    stitched_rgb /= weight_sum
    
    final_rgb = stitched_rgb[:h_orig, :w_orig]
    
    # Save Ultra HD GeoTIFF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_profile.update(
        count=3,
        dtype=rasterio.uint8,
        nodata=0
    )
    
    final_rgb_uint8 = np.clip(final_rgb * 255.0, 0, 255).astype(np.uint8)
    bg_mask = (bands_data["B2"] == 0)
    final_rgb_uint8[bg_mask] = 0
    
    with rasterio.open(output_path, "w", **original_profile) as dst:
        dst.write(final_rgb_uint8[:, :, 0], 1)
        dst.write(final_rgb_uint8[:, :, 1], 2)
        dst.write(final_rgb_uint8[:, :, 2], 3)
    print(f"Stitched Ultra HD GeoTIFF saved to {output_path}")
    
    # Save high-resolution PNG for web/local viewer
    png_path = output_path.replace(".TIF", ".png").replace(".tif", ".png")
    Image.fromarray(final_rgb_uint8).save(png_path, quality=95)
    print(f"Stitched Ultra HD PNG saved to {png_path}")
    
    # Generate interactive zoomable HTML viewer
    viewer_html_path = output_path.replace(".TIF", "_viewer.html").replace(".tif", "_viewer.html")
    generate_zoomable_viewer_html(os.path.basename(png_path), viewer_html_path)
    print(f"Interactive Ultra HD Zoomable Viewer created at {viewer_html_path}")
    
    # Metrics computation
    valid_mask = ~bg_mask
    if np.sum(valid_mask) > 0:
        psnr_val = calculate_psnr(final_rgb[valid_mask], gt_rgb[valid_mask])
        ssim_val = calculate_ssim(final_rgb, gt_rgb)
        print(f"Full Scene Metrics -> PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f}")
    else:
        psnr_val, ssim_val = 0.0, 0.0
        
    return {
        "output_tif": output_path,
        "output_png": png_path,
        "viewer_html": viewer_html_path,
        "psnr": psnr_val,
        "ssim": ssim_val
    }

if __name__ == "__main__":
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    if not os.path.isabs(raw_dir):
        raw_dir = str(PROJECT_ROOT / raw_dir)
        
    scene_folders = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and d != "processed"])
    if scene_folders:
        first_scene = os.path.join(raw_dir, scene_folders[0])
        print(f"Running inference on {first_scene}")
        run_inference(first_scene, config=config)
    else:
        print("No raw scene found in FILES directory to test inference.")
