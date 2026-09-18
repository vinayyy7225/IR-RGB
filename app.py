import os
import sys
import yaml
import glob
import base64
import io
from pathlib import Path
import torch
import numpy as np
import rasterio
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.realesrgan.model import RRDBNet
from models.pix2pix.model import UNetGenerator
from evaluation.metrics import calculate_psnr, calculate_ssim

st.set_page_config(
    page_title="ISRO Satellite IR Colorization & Ultra HD Enhancement",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0b0d14;
        color: #ffffff;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 1.8rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #00f2fe;
        margin-top: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #0b0d14;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.6);
        transform: scale(1.02);
    }
    
    section[data-testid="stSidebar"] {
        background-color: #080a10;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path="configs/config.yaml"):
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        config_file = Path(config_path)
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

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

def render_zoomable_viewer(img_array, height=520):
    """
    Renders an interactive Ultra-HD Pan & Zoom viewer component in Streamlit.
    """
    pil_img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    viewer_code = f"""
    <div style="position: relative; width: 100%; height: {height}px; background: #05070d; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); cursor: grab;" id="panContainer">
        <div style="position: absolute; top: 12px; left: 15px; z-index: 10; background: rgba(15,23,42,0.85); backdrop-filter: blur(8px); padding: 6px 14px; border-radius: 20px; font-family: sans-serif; font-size: 12px; color: #38bdf8; border: 1px solid rgba(255,255,255,0.1);">
            🔍 Scroll to Zoom • Drag to Pan (<span id="zoomDisplay">100%</span>)
        </div>
        <div style="position: absolute; bottom: 12px; right: 15px; z-index: 10; display: flex; gap: 8px;">
            <button id="btnZoomIn" style="background: rgba(15,23,42,0.85); color: #fff; border: 1px solid rgba(255,255,255,0.2); width: 32px; height: 32px; border-radius: 50%; font-size: 16px; cursor: pointer;">+</button>
            <button id="btnZoomOut" style="background: rgba(15,23,42,0.85); color: #fff; border: 1px solid rgba(255,255,255,0.2); width: 32px; height: 32px; border-radius: 50%; font-size: 16px; cursor: pointer;">−</button>
            <button id="btnReset" style="background: rgba(15,23,42,0.85); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 0 12px; border-radius: 16px; font-size: 12px; cursor: pointer;">Reset</button>
        </div>
        <div id="wrapper" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
            <img id="zoomImg" src="data:image/png;base64,{img_b64}" style="max-width: 95%; max-height: 95%; object-fit: contain; transform-origin: center center; user-select: none; -webkit-user-drag: none;" />
        </div>
    </div>
    <script>
        const container = document.getElementById('panContainer');
        const img = document.getElementById('zoomImg');
        const zoomDisplay = document.getElementById('zoomDisplay');
        let scale = 1;
        let panning = false;
        let pointX = 0, pointY = 0, startX = 0, startY = 0;

        function update() {{
            img.style.transform = `translate(${{pointX}}px, ${{pointY}}px) scale(${{scale}})`;
            zoomDisplay.textContent = `${{Math.round(scale * 100)}}%`;
        }}

        container.onmousedown = (e) => {{
            e.preventDefault();
            startX = e.clientX - pointX;
            startY = e.clientY - pointY;
            panning = true;
            container.style.cursor = 'grabbing';
        }};

        window.onmouseup = () => {{
            panning = false;
            container.style.cursor = 'grab';
        }};

        window.onmousemove = (e) => {{
            if (!panning) return;
            pointX = e.clientX - startX;
            pointY = e.clientY - startY;
            update();
        }};

        container.onwheel = (e) => {{
            e.preventDefault();
            const xs = (e.clientX - pointX) / scale;
            const ys = (e.clientY - pointY) / scale;
            const delta = -e.deltaY;
            scale = delta > 0 ? scale * 1.15 : scale / 1.15;
            scale = Math.min(Math.max(0.2, scale), 25.0);
            pointX = e.clientX - xs * scale;
            pointY = e.clientY - ys * scale;
            update();
        }};

        document.getElementById('btnZoomIn').onclick = () => {{ scale = Math.min(scale * 1.3, 25.0); update(); }};
        document.getElementById('btnZoomOut').onclick = () => {{ scale = Math.max(scale / 1.3, 0.2); update(); }};
        document.getElementById('btnReset').onclick = () => {{ scale = 1; pointX = 0; pointY = 0; update(); }};
    </script>
    """
    components.html(viewer_code, height=height + 10)

def load_scene_crop(scene_dir, crop_size=1024):
    b2_files = glob.glob(os.path.join(scene_dir, "*_B2.TIF"))
    if not b2_files:
        return None
    base_path = b2_files[0].replace("_B2.TIF", "")
    
    paths = {
        "B2": f"{base_path}_B2.TIF",
        "B3": f"{base_path}_B3.TIF",
        "B4": f"{base_path}_B4.TIF",
        "B5": f"{base_path}_B5.TIF",
        "B10": f"{base_path}_B10.TIF",
        "B11": f"{base_path}_B11.TIF",
    }
    
    bands_data = {}
    for b, p in paths.items():
        if not os.path.exists(p):
            return None
        with rasterio.open(p) as src:
            h, w = src.shape
            cy, cx = h // 2, w // 2
            window = rasterio.windows.Window(max(0, cx - crop_size//2), max(0, cy - crop_size//2), crop_size, crop_size)
            bands_data[b] = src.read(1, window=window)
            
    return bands_data

def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    if not os.path.isabs(raw_dir):
        raw_dir = str(PROJECT_ROOT / raw_dir)
        
    checkpoints_dir = config["training"]["checkpoints_dir"]
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = str(PROJECT_ROOT / checkpoints_dir)
        
    st.markdown('<div class="title-gradient">ISRO Bharatiya Antariksh Hackathon</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Thermal Infrared (TIRS) to Ultra-HD True-Color RGB Synthesis Pipeline</div>', unsafe_allow_html=True)
    
    scene_folders = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and d != "processed"]) if os.path.exists(raw_dir) else []
    
    st.sidebar.markdown("### 🛰️ Landsat Scene Selection")
    if not scene_folders:
        st.sidebar.error("No scene directories found in FILES folder.")
        return
        
    selected_scene = st.sidebar.selectbox("Select Target Scene", scene_folders)
    selected_scene_dir = os.path.join(raw_dir, selected_scene)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Checkpoint Models")
    
    all_chks = os.listdir(checkpoints_dir) if os.path.exists(checkpoints_dir) else []
    sr_chks = sorted([f for f in all_chks if f.startswith("realesrgan") and f.endswith(".pth")])
    pix_chks = sorted([f for f in all_chks if (f.startswith("pix2pix") or f.startswith("pix2pix_gen")) and f.endswith(".pth")])
    
    sr_chk = os.path.join(checkpoints_dir, sr_chks[0]) if sr_chks else None
    pix_chk = os.path.join(checkpoints_dir, pix_chks[0]) if pix_chks else None
    
    if sr_chks:
        st.sidebar.success(f"✔️ Real-ESRGAN: {sr_chks[0]}")
    else:
        st.sidebar.warning("⚠️ No Real-ESRGAN checkpoint found.")
        
    if pix_chks:
        st.sidebar.success(f"✔️ Pix2Pix: {pix_chks[0]}")
    else:
        st.sidebar.warning("⚠️ No Pix2Pix checkpoint found.")
        
    # Crop size selector
    crop_size = st.sidebar.select_slider("Inference Viewport Resolution", options=[512, 1024, 1536, 2048], value=1024)
    
    with st.spinner("Loading high-resolution satellite bands..."):
        data = load_scene_crop(selected_scene_dir, crop_size=crop_size)
        
    if data is None:
        st.error("Could not load scene bands. Ensure B2, B3, B4, B5, B10, and B11 TIFF files exist.")
        return
        
    # Prepare normalized data
    rgb_min_max = {b: get_percentile_min_max(data[b]) for b in ["B4", "B3", "B2"]}
    gt_rgb = np.stack([
        normalize_band(data["B4"], *rgb_min_max["B4"]),
        normalize_band(data["B3"], *rgb_min_max["B3"]),
        normalize_band(data["B2"], *rgb_min_max["B2"])
    ], axis=-1)
    
    t10_min, t10_max = get_percentile_min_max(data["B10"])
    t11_min, t11_max = get_percentile_min_max(data["B11"])
    norm_t10 = normalize_band(data["B10"], t10_min, t10_max)
    norm_t11 = normalize_band(data["B11"], t11_min, t11_max)
    ir_input = np.stack([norm_t10, norm_t11], axis=-1)
    
    tab_enhance, tab_bands, tab_ndvi = st.tabs(["🚀 Ultra HD Colorization", "🔍 Spectral Bands", "🌱 Physical NDVI Consistency"])
    
    with tab_enhance:
        st.markdown("### Interactive Ultra-HD Colorization & Deep Zoom")
        st.write("Generate high-fidelity true color from thermal bands and zoom in smoothly without quality loss.")
        
        if st.button("✨ Run Ultra HD Synthesis"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scale = config["models"]["realesrgan"]["scale"]
            patch_size = config["data"]["patch_size"]
            
            # Load models
            net_sr = RRDBNet(in_nc=2, out_nc=2, nf=config["models"]["realesrgan"]["num_filters"], nb=config["models"]["realesrgan"]["num_blocks"], gc=32, upscale=scale).to(device)
            if sr_chk and os.path.exists(sr_chk):
                weights_sr = torch.load(sr_chk, map_location=device)
                if "model" in weights_sr: weights_sr = weights_sr["model"]
                net_sr.load_state_dict(weights_sr)
            net_sr.eval()
            
            net_g = UNetGenerator(input_nc=2, output_nc=3, num_downs=8, ngf=64).to(device)
            if pix_chk and os.path.exists(pix_chk):
                weights_g = torch.load(pix_chk, map_location=device)
                if "generator" in weights_g: weights_g = weights_g["generator"]
                net_g.load_state_dict(weights_g)
            net_g.eval()
            
            # Stitching with 50% overlap and Hann window
            stride = patch_size // 2
            h_crop, w_crop, _ = ir_input.shape
            
            # Pad
            pad_h = (patch_size - h_crop % patch_size) % patch_size
            pad_w = (patch_size - w_crop % patch_size) % patch_size
            padded_ir = np.pad(ir_input, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            h_p, w_p, _ = padded_ir.shape
            
            stitched_rgb = np.zeros((h_p, w_p, 3), dtype=np.float32)
            weight_sum = np.zeros((h_p, w_p, 1), dtype=np.float32)
            
            w_1d = np.hanning(patch_size)
            w_2d = np.outer(w_1d, w_1d)[:, :, None].astype(np.float32)
            
            with torch.no_grad():
                for y in range(0, h_p - patch_size + 1, stride):
                    for x in range(0, w_p - patch_size + 1, stride):
                        p_ir = padded_ir[y : y + patch_size, x : x + patch_size]
                        t_ir = torch.from_numpy(p_ir).permute(2, 0, 1).unsqueeze(0).float().to(device)
                        
                        lr_ir = torch.nn.functional.interpolate(t_ir, size=(patch_size // scale, patch_size // scale), mode='bilinear', align_corners=False)
                        sr_ir = torch.clamp(net_sr(lr_ir), 0.0, 1.0)
                        
                        fake_rgb = net_g(sr_ir)
                        fake_rgb = torch.clamp((fake_rgb + 1.0) * 0.5, 0.0, 1.0)
                        
                        rgb_out = fake_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        stitched_rgb[y : y + patch_size, x : x + patch_size] += rgb_out * w_2d
                        weight_sum[y : y + patch_size, x : x + patch_size] += w_2d
                        
            weight_sum[weight_sum == 0] = 1e-8
            stitched_rgb /= weight_sum
            final_gen = np.clip(stitched_rgb[:h_crop, :w_crop], 0.0, 1.0)
            
            # Metrics
            psnr_score = calculate_psnr(final_gen, gt_rgb)
            ssim_score = calculate_ssim(final_gen, gt_rgb)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Signal-to-Noise Ratio (PSNR)</div>
                    <div class="metric-value">{psnr_score:.2f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Structural Similarity Index (SSIM)</div>
                    <div class="metric-value">{ssim_score:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("#### 🔍 Ultra-HD Deep Zoom Viewer (Generated Output)")
            st.caption("Interact below: Scroll wheel zooms down to individual pixels. Drag mouse to pan across landscape.")
            gen_uint8 = (final_gen * 255.0).astype(np.uint8)
            render_zoomable_viewer(gen_uint8, height=550)
            
            st.markdown("---")
            st.markdown("#### Side-by-Side Visual Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.image(norm_t10, caption="Input Thermal Infrared (Band 10)", use_container_width=True)
            with c2:
                st.image(gt_rgb, caption="Ground Truth Optical RGB Composite", use_container_width=True)
                
    with tab_bands:
        st.markdown("### Raw Satellite Spectral Bands")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(norm_t10, caption="Band 10: Thermal Infrared 1", use_container_width=True)
        with col2:
            st.image(norm_t11, caption="Band 11: Thermal Infrared 2", use_container_width=True)
        with col3:
            nir_min, nir_max = get_percentile_min_max(data["B5"])
            norm_nir = normalize_band(data["B5"], nir_min, nir_max)
            st.image(norm_nir, caption="Band 5: Near-Infrared (NIR)", use_container_width=True)
            
    with tab_ndvi:
        st.markdown("### Physical Spectral Consistency (NDVI)")
        st.write("Verifies that the generated optical vegetation reflectance obeys physical earth observation laws.")
        nir_min, nir_max = get_percentile_min_max(data["B5"])
        norm_nir = normalize_band(data["B5"], nir_min, nir_max)
        gt_red = gt_rgb[:, :, 0]
        
        gt_ndvi = (norm_nir - gt_red) / np.clip(norm_nir + gt_red, 1e-3, None)
        gt_ndvi_disp = np.clip((gt_ndvi + 1.0) * 0.5, 0.0, 1.0)
        
        st.image(gt_ndvi_disp, caption="Ground Truth Normalized Difference Vegetation Index", use_container_width=True)

if __name__ == "__main__":
    main()
