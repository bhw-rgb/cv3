import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
import torch
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
    page_title="Medical Image Segmentation & Post-processing",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Medical Image Segmentation & Post-processing")

# ============================================================
# 1. Utility Functions
# ============================================================

def calculate_dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
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

def build_model() -> UNet:
    return UNet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=2, 
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2), 
        num_res_units=2, 
        norm=Norm.BATCH
    )

@st.cache_resource
def load_model_cached(model_path: str, device_name: str):
    device = torch.device(device_name)
    model = build_model()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def run_inference(image_path, model, device_name, roi_size=(160, 160, 160)):
    device = torch.device(device_name)
    transforms = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]), 
        Orientationd(keys=["image"], axcodes="RAS"), 
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear",)), 
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=["image"], source_key="image"), 
        EnsureTyped(keys=["image"])
    ])
    
    data = transforms({"image": image_path})
    image = data["image"]
    input_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = sliding_window_inference(inputs=input_tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
    
    raw_pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    image_np = image[0].cpu().numpy()
    
    return {"image_np": image_np, "raw_pred_np": raw_pred}

def load_label(label_path, image_np_shape):
    # For demo purposes, we simplify label loading to match image_np after transforms.
    # In a real scenario, labels would need the same transforms as images (except intensity scaling).
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata().astype(np.uint8)
    
    # Simple resize/crop for demo if shapes don't match (ideally use MONAI transforms)
    if label_data.shape != image_np_shape:
        # This is a fallback. Ideally, labels should go through the same Orientation/Spacing/Crop transforms.
        # For this demo, we'll try to match it or show a warning.
        st.warning(f"Label shape {label_data.shape} does not match pre-processed image shape {image_np_shape}. Automatic alignment might be needed.")
        # Padding or cropping to match for visualization purposes
        new_label = np.zeros(image_np_shape, dtype=np.uint8)
        min_shape = [min(s1, s2) for s1, s2 in zip(label_data.shape, image_np_shape)]
        new_label[:min_shape[0], :min_shape[1], :min_shape[2]] = label_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        return new_label
    return label_data

# ============================================================
# 2. Sidebar Implementation
# ============================================================

with st.sidebar:
    st.header("📂 Data Upload")
    img_file = st.file_uploader("Upload Image (NIfTI)", type=["nii", "gz"])
    lbl_file = st.file_uploader("Upload Label (NIfTI)", type=["nii", "gz"])
    
    st.header("📐 Visualization Settings")
    axis_choice = st.selectbox("Select Axis", options=["Axial", "Coronal", "Sagittal"])
    axis_map = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
    ax_idx = axis_map[axis_choice]
    
    st.header("🛠️ Post-processing")
    do_lcc = st.checkbox("Largest Connected Component")
    do_sor = st.checkbox("Small Object Removal")
    min_size = st.number_input("Min Size", value=100, min_value=1)
    do_bc = st.checkbox("Binary Closing")
    closing_iters = st.number_input("Closing Iterations", value=1, min_value=1)
    
    apply_post = st.button("Apply Post-processing", type="primary")

# ============================================================
# 3. Model Loading
# =============================-==============================
model_path = "./best_metric_model.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model_cached(model_path, device)

# ============================================================
# 4. Main Logic
# ============================================================

if img_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=img_file.name) as tmp_img:
        tmp_img.write(img_file.getbuffer())
        img_path = tmp_img.name
        
    if "result" not in st.session_state or st.sidebar.button("Run Inference"):
        with st.spinner("Running Inference..."):
            st.session_state.result = run_inference(img_path, model, device)
            
    res = st.session_state.result
    image_np = res["image_np"]
    raw_pred = res["raw_pred_np"]
    
    label_np = None
    if lbl_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=lbl_file.name) as tmp_lbl:
            tmp_lbl.write(lbl_file.getbuffer())
            label_np = load_label(tmp_lbl.name, image_np.shape)
            
    # Post-processing
    if apply_post or "post_pred" not in st.session_state:
        st.session_state.post_pred = post_process(
            raw_pred, 
            lcc=do_lcc, 
            small_obj_removal=do_sor, 
            min_size=min_size, 
            binary_closing=do_bc, 
            closing_iterations=closing_iters
        )
    
    post_pred = st.session_state.post_pred
    
    # Slice Slider
    max_slices = image_np.shape[ax_idx]
    slice_idx = st.sidebar.slider("Select Slice", 0, max_slices - 1, max_slices // 2)
    
    # Helper to get slice
    def get_slice(data, axis, idx):
        if axis == 0: return data[idx, :, :]
        if axis == 1: return data[:, idx, :]
        return data[:, :, idx]

    img_slice = get_slice(image_np, ax_idx, slice_idx)
    raw_pred_slice = get_slice(raw_pred, ax_idx, slice_idx)
    post_pred_slice = get_slice(post_pred, ax_idx, slice_idx)
    lbl_slice = get_slice(label_np, ax_idx, slice_idx) if label_np is not None else np.zeros_like(img_slice)

    # Dice Calculation
    dice_pre = calculate_dice(label_np, raw_pred) if label_np is not None else 0.0
    dice_post = calculate_dice(label_np, post_pred) if label_np is not None else 0.0

    # Display Results
    st.subheader(f"Results Comparison - {axis_choice} Axis (Slice {slice_idx})")
    
    if label_np is not None:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Dice (Before Post-processing)", f"{dice_pre:.4f}")
        col_m2.metric("Dice (After Post-processing)", f"{dice_post:.4f}", delta=f"{dice_post-dice_pre:.4f}")

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        st.write("**Original**")
        fig, ax = plt.subplots()
        ax.imshow(img_slice, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    with row1_col2:
        st.write("**Label**")
        fig, ax = plt.subplots()
        ax.imshow(lbl_slice, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    with row1_col3:
        st.write("**Prediction (Before)**")
        fig, ax = plt.subplots()
        ax.imshow(raw_pred_slice, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.write("**Overlay (Before)**")
        fig, ax = plt.subplots()
        ax.imshow(img_slice, cmap="gray")
        ax.imshow(raw_pred_slice, cmap="Reds", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)

    with row2_col2:
        st.write("**Prediction (After)**")
        fig, ax = plt.subplots()
        ax.imshow(post_pred_slice, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    with row2_col3:
        st.write("**Overlay (After)**")
        fig, ax = plt.subplots()
        ax.imshow(img_slice, cmap="gray")
        ax.imshow(post_pred_slice, cmap="Reds", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)

else:
    st.info("Please upload a medical image (NIfTI) to begin.")
