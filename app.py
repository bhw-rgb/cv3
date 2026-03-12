import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
import torch
import plotly.graph_objects as go

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
# 0. Streamlit 기본 설정
# ============================================================
st.set_page_config(
    page_title="MONAI 3D Segmentation Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 MONAI + Streamlit 3D 의료영상 데모")
st.markdown("""
이 앱은 NIfTI(.nii / .nii.gz) 파일을 업로드하거나 샘플 데이터를 생성하여 3D Segmentation 추론 결과를 시각적으로 확인합니다.
2D 슬라이스 뷰어와 **인터랙티브 3D 뷰어**를 모두 제공합니다.
""")

# ============================================================
# 1. Session State 초기화
# ============================================================
if "inference_done" not in st.session_state:
    st.session_state.inference_done = False
if "result" not in st.session_state:
    st.session_state.result = None
if "raw_shape" not in st.session_state:
    st.session_state.raw_shape = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "source_token" not in st.session_state:
    st.session_state.source_token = None


# ============================================================
# 2. 유틸리티 함수
# ============================================================
def is_nifti_file_name(file_name: str) -> bool:
    lower = file_name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")

def list_nifti_files(folder_path: str, recursive: bool = False) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir(): return []
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    files = [str(p.resolve()) for p in candidates if p.is_file() and is_nifti_file_name(p.name)]
    files.sort()
    return files

def generate_synthetic_nifti():
    size = (96, 96, 96)
    data = np.zeros(size, dtype=np.float32)
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    center = (48, 48, 48)
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    data[dist_from_center <= 25] = 100.0
    data += np.random.normal(-50, 20, size)
    affine = np.eye(4)
    image = nib.Nifti1Image(data, affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(image, tmp.name)
        return tmp.name

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
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"3D 시각화 에러: {e}")
        return None

def make_overlay_figure(image_slice, pred_slice):
    def norm(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(img)
    img_disp = norm(image_slice)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_disp, cmap="gray"); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(pred_slice, cmap="gray"); axes[1].set_title("AI Prediction"); axes[1].axis("off")
    axes[2].imshow(img_disp, cmap="gray"); axes[2].imshow(pred_slice, cmap="Reds", alpha=0.4)
    axes[2].set_title("Overlay View"); axes[2].axis("off")
    plt.tight_layout()
    return fig

# (이하 모델 로직은 기존과 동일)
def build_model() -> UNet:
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)

@st.cache_resource
def load_model_cached(model_path: str, device_name: str):
    device = torch.device(device_name)
    model = build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def run_inference(image_path, model, device_name, roi_size=(160, 160, 160)):
    device = torch.device(device_name)
    transforms = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Orientationd(keys=["image"], axcodes="RAS"), Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear",)), ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=["image"], source_key="image"), EnsureTyped(keys=["image"])])
    data = transforms({"image": image_path})
    image = data["image"]
    affine = data["image_meta_dict"].get("affine", np.eye(4)) if "image_meta_dict" in data else np.eye(4)
    input_tensor = image.unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        logits = sliding_window_inference(inputs=input_tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
    elapsed = time.time() - start_time
    raw_pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    image_np = image[0].cpu().numpy()
    return {"image_np": image_np, "raw_pred_np": raw_pred, "affine": affine, "elapsed": elapsed}

# ============================================================
# 3. 사이드바 설정
# ============================================================
with st.sidebar:
    st.header("⚙️ 실행 설정")
    model_path = st.text_input("모델 파일 (.pth)", value="./best_metric_model.pth")
    device_name = st.selectbox("디바이스", options=["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
    roi_x = st.number_input("ROI X", 32, 256, 160, 16)
    roi_y = st.number_input("ROI Y", 32, 256, 160, 16)
    roi_z = st.number_input("ROI Z", 32, 256, 160, 16)
    
    st.markdown("---")
    st.info("이 앱은 MONAI 프레임워크를 기반으로 학습된 3D Segmentation 모델을 실시간으로 추론합니다.")

try:
    model = load_model_cached(model_path, device_name)
    st.sidebar.success("✅ 모델 로드 성공")
except Exception as e:
    st.sidebar.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# ============================================================
# 4. 입력 파일 선택
# ============================================================
st.subheader("📁 입력 파일 선택")
input_mode = st.radio("입력 방식 선택", options=["파일 업로드", "가상 샘플 생성 (테스트용)", "서버 폴더 탐색"], horizontal=True)

input_path = None
if input_mode == "파일 업로드":
    up_file = st.file_uploader("NIfTI 파일(.nii / .nii.gz)", type=None)
    if up_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=up_file.name) as tmp:
            tmp.write(up_file.getbuffer())
            input_path = tmp.name
elif input_mode == "가상 샘플 생성 (테스트용)":
    if st.button("샘플 데이터 생성"):
        st.session_state.sample_path = generate_synthetic_nifti()
        st.success("테스트용 샘플이 생성되었습니다!")
    if "sample_path" in st.session_state: input_path = st.session_state.sample_path
else:
    folder = st.text_input("폴더 경로", ".")
    files = list_nifti_files(folder)
    if files: input_path = st.selectbox("파일 선택", options=files, format_func=lambda x: Path(x).name)

# ============================================================
# 5. 추론 실행
# ============================================================
if st.button("🚀 추론 실행", type="primary"):
    if input_path:
        with st.spinner("AI 분석 중... 잠시만 기다려주세요."):
            try:
                result = run_inference(input_path, model, device_name, (roi_x, roi_y, roi_z))
                st.session_state.result = result
                st.session_state.raw_shape = nib.load(input_path).shape
                st.session_state.inference_done = True
            except Exception as e: st.exception(e)
    else: st.warning("분석할 파일을 먼저 선택해주세요.")

# ============================================================
# 6. 결과 표시
# ============================================================
if st.session_state.inference_done and st.session_state.result:
    res = st.session_state.result
    raw_pred = res["raw_pred_np"]
    image_np = res["image_np"]
    
    # 상단 Metric 카드 (중요!)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("원본 Shape", str(st.session_state.raw_shape))
    c2.metric("전처리 Shape", str(image_np.shape))
    c3.metric("양성 Voxel", int((raw_pred > 0).sum()))
    c4.metric("추론 시간", f"{res['elapsed']:.2f}s")
    
    st.markdown("---")
    
    # 2D 뷰어와 3D 뷰어 나란히 배치 (혹은 탭)
    tab1, tab2 = st.tabs(["🖼️ 2D 슬라이스 뷰어", "🧊 3D 입체 시각화"])
    
    with tab1:
        ax = st.radio("축 선택", [0, 1, 2], format_func=lambda x: ["Sagittal", "Coronal", "Axial"][x], horizontal=True, index=2)
        
        # 마스크가 가장 많은 슬라이스 찾기
        slice_sums = [raw_pred[i,:,:].sum() if ax==0 else raw_pred[:,i,:].sum() if ax==1 else raw_pred[:,:,i].sum() for i in range(raw_pred.shape[ax])]
        best_slice = int(np.argmax(slice_sums)) if sum(slice_sums) > 0 else raw_pred.shape[ax]//2
        
        sl_idx = st.slider("슬라이스 인덱스", 0, image_np.shape[ax]-1, best_slice)
        
        def get_sl(vol, axis, idx):
            if axis == 0: return vol[idx, :, :]
            elif axis == 1: return vol[:, idx, :]
            return vol[:, :, idx]
        
        st.pyplot(make_overlay_figure(get_sl(image_np, ax, sl_idx), get_sl(raw_pred, ax, sl_idx)))

    with tab2:
        with st.spinner("3D 모델을 생성 중입니다..."):
            fig_3d = visualize_3d_mask(raw_pred)
            if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)
            else: st.warning("시각화할 양성 영역이 없습니다.")

    # 결과 다운로드 및 세부 정보
    with st.expander("📝 상세 정보 및 다운로드"):
        st.write("Affine Matrix:")
        st.code(str(res["affine"]))
        
        def make_nifti_bytes(np_data, aff):
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                nib.save(nib.Nifti1Image(np_data, aff), tmp.name)
                with open(tmp.name, "rb") as f: return f.read()
        
        st.download_button("💾 결과 마스크 다운로드 (.nii.gz)", make_nifti_bytes(raw_pred, res["affine"]), "prediction.nii.gz")
