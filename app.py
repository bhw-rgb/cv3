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

st.title("MONAI + Streamlit 3D 의료영상 데모")
st.caption("NIfTI(.nii / .nii.gz) 파일을 업로드하거나 폴더에서 선택해 3D segmentation 추론 결과를 확인합니다.")

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
# 2. 공통 유틸 함수
# ============================================================
def is_nifti_file_name(file_name: str) -> bool:
    """파일명이 .nii 또는 .nii.gz 인지 확인"""
    lower = file_name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def list_nifti_files(folder_path: str, recursive: bool = False) -> List[str]:
    """
    폴더 내 .nii / .nii.gz 파일 목록 반환
    recursive=True면 하위 폴더까지 포함
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"폴더 경로가 아닙니다: {folder_path}")

    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    files = []
    for p in candidates:
        if p.is_file() and is_nifti_file_name(p.name):
            files.append(str(p.resolve()))

    files.sort()
    return files


def make_source_token_from_path(file_path: str) -> str:
    """파일 경로 기반 token 생성"""
    p = Path(file_path)
    stat = p.stat()
    return f"path::{str(p.resolve())}::{stat.st_size}::{stat.st_mtime}"


def make_source_token_from_upload(uploaded_file) -> str:
    """업로드 파일 기반 token 생성"""
    return f"upload::{uploaded_file.name}::{uploaded_file.size}"


def get_available_device_options():
    """
    사용 가능한 디바이스 목록 반환
    value: 실제 torch.device에 들어갈 값
    label: 사용자에게 보여줄 이름
    """
    options = []

    # Apple Silicon GPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        options.append(("mps", "GPU (Apple Silicon / MPS)"))

    # NVIDIA GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            options.append((f"cuda:{i}", f"GPU (CUDA:{i} / {gpu_name})"))

    # CPU는 항상 추가
    options.append(("cpu", "CPU"))
    return options


def get_device_debug_text():
    """
    현재 환경에서 GPU가 왜 보이는지 / 왜 안 보이는지 확인용 정보
    """
    lines = [
        f"torch.__version__ = {torch.__version__}",
        f"torch.cuda.is_available() = {torch.cuda.is_available()}",
        f"torch.version.cuda = {torch.version.cuda}",
    ]

    if torch.cuda.is_available():
        lines.append(f"CUDA device count = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"cuda:{i} = {torch.cuda.get_device_name(i)}")

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    lines.append(f"MPS available = {mps_available}")

    return "\n".join(lines)


def build_model() -> UNet:
    """
    학습에 사용한 것과 동일한 3D UNet 생성
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model


@st.cache_resource
def load_model_cached(model_path: str, device_name: str):
    """
    모델을 한 번만 로드해 캐시
    """
    device = torch.device(device_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    model = build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def reset_inference_state():
    """입력 파일이 바뀌면 이전 추론 상태 초기화"""
    st.session_state.inference_done = False
    st.session_state.result = None
    st.session_state.raw_shape = None
    st.session_state.source_name = None

    # 축별 slider 값도 함께 초기화
    for key in list(st.session_state.keys()):
        if key.startswith("slice_idx_axis_"):
            del st.session_state[key]


def save_uploaded_file_to_temp(uploaded_file) -> str:
    """
    업로드 파일을 임시 파일로 저장
    type 필터 대신 앱 내부에서 확장자 검사
    """
    lower_name = uploaded_file.name.lower()
    if lower_name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif lower_name.endswith(".nii"):
        suffix = ".nii"
    else:
        suffix = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def preprocess_case(image_path: str) -> Dict:
    """
    학습 때와 최대한 비슷한 전처리 수행
    결과 마스크는 전처리 후 공간 기준
    """
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear",),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])

    data = {"image": image_path}
    data = transforms(data)
    return data


def extract_affine_from_meta(image_tensor) -> np.ndarray:
    """
    MetaTensor에서 affine을 최대한 안전하게 추출
    """
    if hasattr(image_tensor, "affine") and image_tensor.affine is not None:
        affine = image_tensor.affine
        try:
            affine = affine.cpu().numpy()
        except Exception:
            affine = np.array(affine)

        if affine.ndim == 3:
            affine = affine[0]
        return affine

    if hasattr(image_tensor, "meta"):
        meta = image_tensor.meta
        for key in ["affine", "original_affine"]:
            if key in meta:
                value = meta[key]
                try:
                    value = value.cpu().numpy()
                except Exception:
                    value = np.array(value)
                if value.ndim == 3:
                    value = value[0]
                return value

    return np.eye(4, dtype=float)


def run_inference(
    data: Dict,
    model,
    device_name: str,
    roi_size: Tuple[int, int, int] = (160, 160, 160),
    sw_batch_size: int = 1,
) -> Dict:
    """
    MONAI sliding_window_inference 기반 추론
    raw prediction만 생성하고 후처리는 별도 수행
    """
    device = torch.device(device_name)

    image = data["image"]
    affine = extract_affine_from_meta(image)

    # (C, H, W, D) -> (B, C, H, W, D)
    input_tensor = image.unsqueeze(0).to(device)

    start_time = time.time()

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
        )

    elapsed = time.time() - start_time

    raw_pred = torch.argmax(logits, dim=1)  # (B, H, W, D)
    raw_pred_np = raw_pred[0].cpu().numpy().astype(np.uint8)
    image_np = image[0].cpu().numpy()       # (H, W, D)

    return {
        "image_np": image_np,
        "raw_pred_np": raw_pred_np,
        "affine": affine,
        "elapsed": elapsed,
    }


def keep_largest_connected_component_np(
    pred_np: np.ndarray,
    target_label: int = 1,
) -> Tuple[np.ndarray, str]:
    """
    3D 전체 볼륨 기준으로 가장 큰 connected component 1개만 유지
    """
    mask = (pred_np == target_label).astype(np.uint8)

    if mask.sum() == 0:
        return pred_np.copy(), "양성 영역이 없어 후처리를 적용할 대상이 없습니다."

    try:
        from scipy import ndimage as ndi
    except Exception as e:
        return pred_np.copy(), (
            "후처리를 위해 scipy가 필요합니다. "
            "현재는 원본 예측 결과를 그대로 표시합니다.\n"
            f"원인: {e}"
        )

    # 3D 6-connectivity
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    labeled, num_components = ndi.label(mask, structure=structure)

    if num_components == 0:
        return pred_np.copy(), "connected component를 찾지 못해 원본 예측을 유지합니다."

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # background 제외

    largest_label = int(np.argmax(component_sizes))
    largest_mask = (labeled == largest_label)

    processed = np.zeros_like(pred_np, dtype=np.uint8)
    processed[largest_mask] = target_label

    kept_voxels = int(largest_mask.sum())
    removed_voxels = int(mask.sum() - kept_voxels)

    note = (
        f"3D 기준 connected component {num_components}개 중 "
        f"가장 큰 1개만 유지했습니다. "
        f"(유지 voxel: {kept_voxels}, 제거 voxel: {removed_voxels})"
    )
    return processed, note


def keep_largest_connected_component_2d(
    pred_slice: np.ndarray,
    target_label: int = 1,
) -> np.ndarray:
    """
    현재 화면에 보이는 2D slice에서 가장 큰 connected component 1개만 유지
    """
    mask = (pred_slice == target_label).astype(np.uint8)

    if mask.sum() == 0:
        return pred_slice.copy()

    try:
        from scipy import ndimage as ndi
    except Exception:
        return pred_slice.copy()

    # 2D 4-connectivity
    structure = ndi.generate_binary_structure(rank=2, connectivity=1)
    labeled, num_components = ndi.label(mask, structure=structure)

    if num_components == 0:
        return pred_slice.copy()

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    largest_label = int(np.argmax(component_sizes))
    largest_mask = (labeled == largest_label)

    processed_2d = np.zeros_like(pred_slice, dtype=np.uint8)
    processed_2d[largest_mask] = target_label
    return processed_2d


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    3D 볼륨에서 axis 기준 slice 추출
    axis=0 -> sagittal
    axis=1 -> coronal
    axis=2 -> axial
    """
    if axis == 0:
        sl = volume[index, :, :]
    elif axis == 1:
        sl = volume[:, index, :]
    else:
        sl = volume[:, :, index]

    return np.asarray(sl)


def get_best_slice(mask_3d: np.ndarray, axis: int) -> int:
    """
    예측 마스크가 가장 많이 존재하는 slice를 기본값으로 선택
    마스크가 없으면 가운데 slice 반환
    """
    if mask_3d.sum() == 0:
        return mask_3d.shape[axis] // 2

    slice_sums = []
    for i in range(mask_3d.shape[axis]):
        sl = extract_slice(mask_3d, axis, i)
        slice_sums.append(sl.sum())

    return int(np.argmax(slice_sums))


def normalize_for_display(image_slice: np.ndarray) -> np.ndarray:
    """시각화를 위한 min-max 정규화"""
    image_slice = image_slice.astype(np.float32)
    mn, mx = image_slice.min(), image_slice.max()

    if mx - mn < 1e-8:
        return np.zeros_like(image_slice)

    return (image_slice - mn) / (mx - mn)


def make_overlay_figure(image_slice: np.ndarray, pred_slice: np.ndarray):
    """
    원본 / 예측 / overlay를 한 번에 보여주는 figure 생성
    """
    image_disp = normalize_for_display(image_slice)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_disp, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(pred_slice, cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(image_disp, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Reds", alpha=0.35)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def make_nifti_bytes(pred_np: np.ndarray, affine: np.ndarray) -> bytes:
    """
    예측 마스크를 .nii.gz 바이트로 변환
    """
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        temp_path = tmp.name

    try:
        img = nib.Nifti1Image(pred_np.astype(np.uint8), affine=affine)
        nib.save(img, temp_path)

        with open(temp_path, "rb") as f:
            data = f.read()

        return data
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 3. 사이드바 설정
# ============================================================
with st.sidebar:
    st.header("실행 설정")

    model_path = st.text_input(
        "모델 파일 경로 (.pth)",
        value="./best_metric_model.pth",
        help="학습에 사용한 best_metric_model.pth 경로를 입력하세요."
    )

    device_items = get_available_device_options()
    device_value_to_label = {value: label for value, label in device_items}
    device_values = [value for value, _ in device_items]

    device_name = st.selectbox(
        "디바이스 선택",
        options=device_values,
        index=0,
        format_func=lambda x: device_value_to_label[x],
    )

    with st.expander("디바이스 상태 확인"):
        st.code(get_device_debug_text())

    has_gpu = any(v.startswith("cuda") or v == "mps" for v in device_values)
    if not has_gpu:
        st.warning(
            "현재 실행 중인 파이썬 환경에서 GPU를 감지하지 못했습니다.\n\n"
            "- NVIDIA GPU는 cuda:0 형태로 표시됩니다.\n"
            "- Apple Silicon GPU는 mps 형태로 표시됩니다.\n"
            "- CPU 전용 PyTorch를 설치한 경우 GPU 목록이 나타나지 않습니다."
        )

    st.subheader("Sliding Window 설정")
    roi_x = st.number_input("ROI X", min_value=32, max_value=256, value=160, step=16)
    roi_y = st.number_input("ROI Y", min_value=32, max_value=256, value=160, step=16)
    roi_z = st.number_input("ROI Z", min_value=32, max_value=256, value=160, step=16)

    sw_batch_size = st.number_input(
        "SW Batch Size",
        min_value=1,
        max_value=8,
        value=1,
        step=1
    )

    st.subheader("후처리")
    keep_largest_cc = st.checkbox(
        "후처리 결과 추가 표시 (가장 큰 connected component만 유지)",
        value=False,
        help="기본 예측 결과는 그대로 유지하고, 하단에 후처리 결과를 추가로 보여줍니다."
    )

    st.markdown("---")
    st.info(
        "권장 입력 파일: .nii 또는 .nii.gz\n\n"
        "주의: 다운로드되는 예측 마스크는 현재 앱 기준으로 "
        "전처리/크롭 이후 공간 기준입니다."
    )

# ============================================================
# 4. 모델 로드
# ============================================================
try:
    model = load_model_cached(model_path, device_name)
    st.success(f"모델 로드 완료: {model_path}")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# ============================================================
# 5. 입력 파일 선택
# ============================================================
st.subheader("입력 파일 선택")

input_mode = st.radio(
    "입력 방식",
    options=["내 컴퓨터에서 업로드", "실행 중인 PC/서버 폴더에서 선택"],
    horizontal=True,
)

selected_source_path = None
selected_source_name = None
selected_source_token = None
uploaded_file = None

if input_mode == "내 컴퓨터에서 업로드":
    st.caption(
        "브라우저의 확장자 필터로 인해 .nii.gz 선택이 막히는 문제를 줄이기 위해 "
        "모든 파일 선택을 허용하고, 앱 내부에서 .nii / .nii.gz 여부를 검사합니다."
    )

    uploaded_file = st.file_uploader(
        "NIfTI 파일 업로드 (.nii / .nii.gz)",
        type=None,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.warning("먼저 .nii 또는 .nii.gz 파일을 업로드하세요.")
        st.stop()

    if not is_nifti_file_name(uploaded_file.name):
        st.error("선택한 파일은 .nii 또는 .nii.gz 형식이 아닙니다.")
        st.stop()

    selected_source_name = uploaded_file.name
    selected_source_token = make_source_token_from_upload(uploaded_file)

else:
    st.caption(
        "이 방식은 Streamlit 앱이 실행 중인 컴퓨터/서버의 폴더를 탐색합니다. "
        "로컬 PC에서 앱을 실행 중이면 내 PC 폴더를 선택하는 효과가 있습니다."
    )

    folder_path = st.text_input(
        "폴더 경로",
        value=".",
        help="예: ./data 또는 /Users/you/data 또는 C:\\data"
    )
    recursive_search = st.checkbox("하위 폴더까지 검색", value=False)

    try:
        nifti_files = list_nifti_files(folder_path, recursive=recursive_search)
    except Exception as e:
        st.error(f"폴더 탐색 실패: {e}")
        st.stop()

    if len(nifti_files) == 0:
        st.warning("해당 폴더에서 .nii 또는 .nii.gz 파일을 찾지 못했습니다.")
        st.stop()

    selected_source_path = st.selectbox(
        "폴더 내 NIfTI 파일 선택",
        options=nifti_files,
        format_func=lambda x: str(Path(x).name),
    )

    selected_source_name = Path(selected_source_path).name
    selected_source_token = make_source_token_from_path(selected_source_path)

# 입력 파일이 바뀌면 이전 결과 초기화
if st.session_state.source_token != selected_source_token:
    reset_inference_state()
    st.session_state.source_token = selected_source_token


# ============================================================
# 6. 추론 실행 버튼
# ============================================================
if st.button("추론 실행", type="primary"):
    temp_image_path = None
    try:
        # 입력 방식에 따라 실제 경로 결정
        if input_mode == "내 컴퓨터에서 업로드":
            temp_image_path = save_uploaded_file_to_temp(uploaded_file)
            image_path_for_inference = temp_image_path
        else:
            image_path_for_inference = selected_source_path

        raw_nifti = nib.load(image_path_for_inference)
        raw_shape = raw_nifti.shape

        with st.spinner("영상을 전처리하는 중입니다..."):
            data = preprocess_case(image_path_for_inference)

        with st.spinner("MONAI 추론을 수행하는 중입니다..."):
            result = run_inference(
                data=data,
                model=model,
                device_name=device_name,
                roi_size=(int(roi_x), int(roi_y), int(roi_z)),
                sw_batch_size=int(sw_batch_size),
            )

        st.session_state.result = result
        st.session_state.raw_shape = raw_shape
        st.session_state.source_name = selected_source_name
        st.session_state.inference_done = True

        st.success("추론이 완료되었습니다.")

    except Exception as e:
        st.exception(e)
        st.session_state.inference_done = False
        st.session_state.result = None

    finally:
        if temp_image_path is not None and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ============================================================
# 7. 추론 결과 표시
# ============================================================
if st.session_state.inference_done and st.session_state.result is not None:
    result = st.session_state.result
    image_np = result["image_np"]
    raw_pred_np = result["raw_pred_np"]
    affine = result["affine"]
    elapsed = result["elapsed"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("원본 shape", str(st.session_state.raw_shape))
    c2.metric("전처리 후 shape", str(image_np.shape))
    c3.metric("기본 예측 양성 voxel 수", int((raw_pred_np > 0).sum()))
    c4.metric("추론 시간(초)", f"{elapsed:.2f}")

    axis_name = st.radio(
        "시각화 축 선택",
        options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
        horizontal=True,
        key="axis_name",
    )

    axis_map = {
        "Sagittal (X)": 0,
        "Coronal (Y)": 1,
        "Axial (Z)": 2,
    }
    axis = axis_map[axis_name]

    default_slice = get_best_slice(raw_pred_np, axis)

    slider_key = f"slice_idx_axis_{axis}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_slice

    slice_idx = st.slider(
        "Slice index",
        min_value=0,
        max_value=image_np.shape[axis] - 1,
        value=st.session_state[slider_key],
        key=slider_key,
    )

    image_slice = extract_slice(image_np, axis, slice_idx)
    raw_pred_slice = extract_slice(raw_pred_np, axis, slice_idx)

    st.subheader("기본 예측 결과")
    fig_raw = make_overlay_figure(image_slice, raw_pred_slice)
    st.pyplot(fig_raw, use_container_width=True)
    plt.close(fig_raw)

    with st.expander("기본 예측 추가 정보"):
        st.write("입력 파일명:", st.session_state.source_name)
        st.write("예측 클래스 값:", np.unique(raw_pred_np).tolist())
        st.write("예측 마스크 shape:", raw_pred_np.shape)
        st.write("다운로드되는 마스크는 전처리 후 공간 기준입니다.")
        st.write("Affine:")
        st.code(np.array2string(affine, precision=3, suppress_small=True))

    raw_nifti_bytes = make_nifti_bytes(raw_pred_np, affine)
    st.download_button(
        label="기본 예측 마스크 다운로드 (.nii.gz)",
        data=raw_nifti_bytes,
        file_name="prediction_mask_raw.nii.gz",
        mime="application/gzip",
        on_click="ignore",
    )

    # --------------------------------------------------------
    # 후처리 결과 추가 표시
    # --------------------------------------------------------
    if keep_largest_cc:
        processed_pred_np, postprocess_note = keep_largest_connected_component_np(
            raw_pred_np,
            target_label=1,
        )

        processed_slice_3d = extract_slice(processed_pred_np, axis, slice_idx)

        # 화면에서는 현재 slice 기준 가장 큰 2D component만 표시
        processed_slice_display = keep_largest_connected_component_2d(
            processed_slice_3d,
            target_label=1,
        )

        st.markdown("---")
        st.subheader("후처리 결과 (가장 큰 connected component만 유지)")

        p1, p2, p3 = st.columns(3)
        p1.metric("후처리 후 양성 voxel 수", int((processed_pred_np > 0).sum()))
        p2.metric(
            "제거된 voxel 수",
            int((raw_pred_np > 0).sum() - (processed_pred_np > 0).sum())
        )
        p3.metric("남은 component 수(3D 기준)", "1개")

        st.info(postprocess_note)
        st.caption("화면에는 현재 slice 기준으로 가장 큰 2D 덩어리만 표시합니다.")

        fig_post = make_overlay_figure(image_slice, processed_slice_display)
        st.pyplot(fig_post, use_container_width=True)
        plt.close(fig_post)

        processed_nifti_bytes = make_nifti_bytes(processed_pred_np, affine)
        st.download_button(
            label="후처리 마스크 다운로드 (.nii.gz)",
            data=processed_nifti_bytes,
            file_name="prediction_mask_postprocessed.nii.gz",
            mime="application/gzip",
            on_click="ignore",
        )
else:
    st.info("입력 파일을 선택한 뒤 '추론 실행' 버튼을 눌러주세요.")