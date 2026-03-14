import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
import torch
import plotly.graph_objects as go
from scipy import ndimage
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
)

# ============================================================
# 0. Streamlit Configuration
# ============================================================
st.set_page_config(
    page_title="Medical Image Segmentation Pro",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Medical Image Segmentation & Post-processing")
st.markdown("NIfTI 파일을 업로드하고 AI 기반 세그멘테이션과 후처리 결과를 실시간으로 비교 분석합니다.")

# ============================================================
# 1. Utility Functions
# ============================================================

def calculate_dice(y_true, y_pred):
    if y_true is None: return 0.0
    y_true_f = (y_true > 0).astype(np.float32).flatten()
    y_pred_f = (y_pred > 0).astype(np.float32).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)

def post_process(mask, lcc=False, small_obj_removal=False, min_size=100, binary_closing=False, closing_iterations=1):
    processed = mask.copy()
    if lcc:
        labels, num_features = ndimage.label(processed)
        if num_features > 0:
            sizes = ndimage.sum(np.ones_like(processed), labels, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            processed = (labels == largest_label).astype(np.uint8)
    if small_obj_removal:
        labels, num_features = ndimage.label(processed)
        if num_features > 0:
            sizes = ndimage.sum(np.ones_like(processed), labels, range(1, num_features + 1))
            mask_size = sizes < min_size
            remove_pixel = mask_size[labels - 1]
            processed[remove_pixel] = 0
            processed = (processed > 0).astype(np.uint8)
    if binary_closing:
        processed = ndimage.binary_closing(processed, iterations=closing_iterations).astype(np.uint8)
    return processed

def visualize_3d_mask(mask_np):
    try:
        step = 2
        sub_mask = mask_np[::step, ::step, ::step]
        if sub_mask.sum() == 0: return None
        z, y, x = np.where(sub_mask > 0)
        fig = go.Figure(data=[go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=np.ones_like(x).flatten(),
            isomin=0.5, isomax=1.5, opacity=0.7,
            colorscale='Reds', caps=dict(x_show=False, y_show=False)
        )])
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=0), height=500
        )
        return fig
    except Exception: return None

@st.cache_resource
def load_model_cached(model_path: str, device_name: str):
    device = torch.device(device_name)
    model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def run_inference(image_path, model, device, roi_size=(160, 160, 160)):
    transforms = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]), 
        Orientationd(keys=["image"], axcodes="RAS"), 
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear",)), 
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=["image"], source_key="image"), EnsureTyped(keys=["image"])
    ])
    data = transforms({"image": image_path})
    image = data["image"]
    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(inputs=input_tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
    return {"image_np": image[0].cpu().numpy(), "raw_pred_np": torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)}

# ============================================================
# 2. Sidebar Implementation
# ============================================================

with st.sidebar:
    st.header("📂 데이터 업로드")
    img_file = st.file_uploader("이미지 업로드 (NIfTI)", type=["nii", "gz"])
    lbl_file = st.file_uploader("라벨 업로드 (NIfTI - 선택)", type=["nii", "gz"])
    
    st.header("📐 시각화 설정")
    axis_choice = st.selectbox("축 선택", options=["Axial", "Coronal", "Sagittal"], index=0)
    axis_map = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
    ax_idx = axis_map[axis_choice]
    
    st.header("🛠️ 후처리 옵션")
    do_lcc = st.checkbox("Largest Connected Component", value=True)
    do_sor = st.checkbox("Small Object Removal", value=True)
    min_size = st.number_input("Min Object Size", value=100, min_value=1)
    do_bc = st.checkbox("Binary Closing")
    closing_iters = st.number_input("Closing Iterations", value=1, min_value=1)
    
    apply_post = st.button("🔥 후처리 적용", type="primary", use_container_width=True)

# ============================================================
# 3. Model & Main Logic
# ============================================================

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model_cached("./best_metric_model.pth", device)

if img_file:
    # Use session state to keep data across reruns
    if "current_img_name" not in st.session_state or st.session_state.current_img_name != img_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=img_file.name) as tmp_img:
            tmp_img.write(img_file.getbuffer())
            img_path = tmp_img.name
        
        with st.spinner("AI 추론 중..."):
            st.session_state.result = run_inference(img_path, model, device)
            st.session_state.current_img_name = img_file.name
            # Reset label and post-processing when new image is uploaded
            if "label_np" in st.session_state: del st.session_state.label_np
            if "post_pred" in st.session_state: del st.session_state.post_pred

    res = st.session_state.result
    image_np, raw_pred = res["image_np"], res["raw_pred_np"]
    
    # Label Processing
    if lbl_file:
        if "current_lbl_name" not in st.session_state or st.session_state.current_lbl_name != lbl_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=lbl_file.name) as tmp_lbl:
                tmp_lbl.write(lbl_file.getbuffer())
                label_data = nib.load(tmp_lbl.name).get_fdata().astype(np.uint8)
                # Alignment check and resize (simplified for demo)
                new_label = np.zeros(image_np.shape, dtype=np.uint8)
                ms = [min(s1, s2) for s1, s2 in zip(label_data.shape, image_np.shape)]
                new_label[:ms[0], :ms[1], :ms[2]] = label_data[:ms[0], :ms[1], :ms[2]]
                st.session_state.label_np = new_label
                st.session_state.current_lbl_name = lbl_file.name
    else:
        if "label_np" in st.session_state: del st.session_state.label_np

    # Post-processing Logic
    if apply_post or "post_pred" not in st.session_state:
        st.session_state.post_pred = post_process(raw_pred, do_lcc, do_sor, min_size, do_bc, closing_iters)
    
    label_np = st.session_state.get("label_np")
    post_pred = st.session_state.post_pred
    
    # Slice Slider
    max_slices = image_np.shape[ax_idx]
    slice_idx = st.sidebar.slider("슬라이스 선택", 0, max_slices - 1, max_slices // 2)
    
    # 1. Dice Score Comparison
    if label_np is not None:
        d1 = calculate_dice(label_np, raw_pred)
        d2 = calculate_dice(label_np, post_pred)
        c1, c2 = st.columns(2)
        c1.metric("Dice Score (후처리 전)", f"{d1:.4f}")
        c2.metric("Dice Score (후처리 후)", f"{d2:.4f}", delta=f"{d2-d1:.4f}")
        st.markdown("---")

    # 2. Main Display (2x3 Grid)
    def get_sl(v, a, i): return v[i,:,:] if a==0 else v[:,i,:] if a==1 else v[:,:,i]
    
    img_sl = get_sl(image_np, ax_idx, slice_idx)
    raw_sl = get_sl(raw_pred, ax_idx, slice_idx)
    post_sl = get_sl(post_pred, ax_idx, slice_idx)
    lbl_sl = get_sl(label_np, ax_idx, slice_idx) if label_np is not None else np.zeros_like(img_sl)

    tab_2d, tab_3d = st.tabs(["🖼️ 2D 슬라이스 비교", "🧊 3D 입체 시각화"])
    
    with tab_2d:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### **원본 (Original)**")
            fig, ax = plt.subplots(); ax.imshow(img_sl, cmap="gray"); ax.axis("off"); st.pyplot(fig)
            
            st.markdown("### **오버레이 (후처리 전)**")
            fig, ax = plt.subplots(); ax.imshow(img_sl, cmap="gray"); ax.imshow(raw_sl, cmap="Reds", alpha=0.4); ax.axis("off"); st.pyplot(fig)
            
        with col2:
            st.markdown("### **라벨 (Label)**")
            fig, ax = plt.subplots(); ax.imshow(lbl_sl, cmap="gray"); ax.axis("off"); st.pyplot(fig)
            
            st.markdown("### **예측 (후처리 후)**")
            fig, ax = plt.subplots(); ax.imshow(post_sl, cmap="gray"); ax.axis("off"); st.pyplot(fig)
            
        with col3:
            st.markdown("### **예측 (후처리 전)**")
            fig, ax = plt.subplots(); ax.imshow(raw_sl, cmap="gray"); ax.axis("off"); st.pyplot(fig)
            
            st.markdown("### **오버레이 (후처리 후)**")
            fig, ax = plt.subplots(); ax.imshow(img_sl, cmap="gray"); ax.imshow(post_sl, cmap="Reds", alpha=0.4); ax.axis("off"); st.pyplot(fig)

    with tab_3d:
        with st.spinner("3D 렌더링 중..."):
            fig_3d = visualize_3d_mask(post_pred)
            if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)
            else: st.warning("시각화할 영역이 없습니다.")

else:
    st.info("시작하려면 사이드바에서 NIfTI 이미지 파일을 업로드하세요.")
