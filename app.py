import os
import tempfile
import numpy as np
import streamlit as st
import torch
from scipy import ndimage

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
    DivisiblePadd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_metric_model.pth")

def build_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=16),
        EnsureTyped(keys=["image", "label"]),
    ])

def save_uploaded_file(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name

def run_inference(model, image_file, label_file):
    image_path = save_uploaded_file(image_file)
    label_path = save_uploaded_file(label_file)

    data = {"image": image_path, "label": label_path}
    sample = get_transforms()(data)

    image = sample["image"].unsqueeze(0).to(DEVICE)
    label = sample["label"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image,
            roi_size=(160, 160, 160),
            sw_batch_size=1,
            predictor=model,
        )
        pred = torch.argmax(output, dim=1)

    image_np = image[0, 0].cpu().numpy()
    label_np = label[0, 0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()

    return image_np, label_np, pred_np

def binary_dice(pred_np, label_np, target_class=1):
    pred_bin = (pred_np == target_class)
    label_bin = (label_np == target_class)
    intersection = np.sum(pred_bin & label_bin)
    pred_sum = np.sum(pred_bin)
    label_sum = np.sum(label_bin)

    if pred_sum + label_sum == 0:
        return 1.0
    return 2.0 * intersection / (pred_sum + label_sum)

def keep_largest_component(binary_mask):
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(np.uint8)

def remove_small_components(binary_mask, min_size=100):
    labeled, num_features = ndimage.label(binary_mask)
    output = np.zeros_like(binary_mask, dtype=np.uint8)
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if component.sum() >= min_size:
            output[component] = 1
    return output

def apply_postprocessing(pred_np, keep_largest=False, remove_small=False, min_size=100, closing=False):
    binary_mask = (pred_np == 1).astype(np.uint8)

    if keep_largest:
        binary_mask = keep_largest_component(binary_mask)

    if remove_small:
        binary_mask = remove_small_components(binary_mask, min_size=min_size)

    if closing:
        binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

    return binary_mask

def get_slice(volume, axis, idx):
    if axis == "axial":
        return volume[:, :, idx]
    elif axis == "coronal":
        return volume[:, idx, :]
    elif axis == "sagittal":
        return volume[idx, :, :]
    return volume[:, :, idx]

def max_index_by_axis(volume, axis):
    if axis == "axial":
        return volume.shape[2] - 1
    elif axis == "coronal":
        return volume.shape[1] - 1
    elif axis == "sagittal":
        return volume.shape[0] - 1
    return volume.shape[2] - 1

st.set_page_config(layout="wide")
st.title("3D 의료영상 Segmentation 데모 서비스")

if not os.path.exists(MODEL_PATH):
    st.error(f"모델 파일이 없습니다: {MODEL_PATH}")
    st.stop()

model = build_model()

st.sidebar.header("입력 및 후처리 설정")
image_file = st.sidebar.file_uploader("CT 이미지(.nii.gz)", type=["gz"])
label_file = st.sidebar.file_uploader("정답 라벨(.nii.gz)", type=["gz"])

axis = st.sidebar.selectbox("축 선택", ["axial", "coronal", "sagittal"])
keep_largest = st.sidebar.checkbox("Largest Connected Component", value=True)
remove_small = st.sidebar.checkbox("Small Object Removal", value=False)
min_size = st.sidebar.slider("최소 component 크기", 10, 1000, 100)
closing = st.sidebar.checkbox("Binary Closing", value=False)

run_button = st.sidebar.button("예측 실행")

if run_button:
    if image_file is None or label_file is None:
        st.warning("이미지와 라벨 파일을 모두 업로드해주세요.")
        st.stop()

    with st.spinner("예측 중입니다..."):
        image_np, label_np, pred_np = run_inference(model, image_file, label_file)

    pred_post = apply_postprocessing(
        pred_np,
        keep_largest=keep_largest,
        remove_small=remove_small,
        min_size=min_size,
        closing=closing,
    )

    dice_before = binary_dice(pred_np, label_np, target_class=1)
    dice_after = binary_dice(pred_post, label_np, target_class=1)

    max_idx = max_index_by_axis(image_np, axis)
    slice_idx = st.sidebar.slider("Slice 선택", 0, max_idx, max_idx // 2)

    image_slice = get_slice(image_np, axis, slice_idx)
    label_slice = get_slice(label_np, axis, slice_idx)
    pred_slice = get_slice(pred_np, axis, slice_idx)
    pred_post_slice = get_slice(pred_post, axis, slice_idx)

    st.subheader("후처리 전 / 후 비교")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dice (후처리 전)", f"{dice_before:.4f}")
    col2.metric("Dice (후처리 후)", f"{dice_after:.4f}")
    col3.metric("Dice 변화량", f"{dice_after - dice_before:+.4f}")

    st.markdown("### 원본 / 라벨 / 예측 / 오버레이")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.image(image_slice, caption="Original", clamp=True)
    with c2:
        st.image(label_slice.astype(np.float32), caption="Label", clamp=True)
    with c3:
        st.image(pred_slice.astype(np.float32), caption="Prediction (Before)", clamp=True)
    with c4:
        overlay_before = np.stack([image_slice]*3, axis=-1)
        overlay_before[..., 0] = np.maximum(overlay_before[..., 0], pred_slice * overlay_before.max())
        st.image(overlay_before, caption="Overlay (Before)", clamp=True)

    st.markdown("### 후처리 적용 후 결과")
    c5, c6 = st.columns(2)

    with c5:
        st.image(pred_post_slice.astype(np.float32), caption="Prediction (After)", clamp=True)

    with c6:
        overlay_after = np.stack([image_slice]*3, axis=-1)
        overlay_after[..., 0] = np.maximum(overlay_after[..., 0], pred_post_slice * overlay_after.max())
        st.image(overlay_after, caption="Overlay (After)", clamp=True)