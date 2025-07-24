from __future__ import annotations

import base64
import logging
import math
import random
import os
import sys
import time
import warnings
import itertools
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import traceback
import io as py_io
import os.path as osp
import numpy as np
import copy
from einops import rearrange
import cv2

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
# min tokens per image
MIN_TOKENS = 4
# max tokens per image
MAX_TOKENS = 20480
MIN_PIXELS = MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 4 * 28 * 28 = 3,136
MAX_PIXELS = MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 20480 * 28 * 28 = 16,056,320
MAX_RATIO = 200

# min tokens per video frame
VIDEO_MIN_TOKENS = 48
# max tokens per video frame
VIDEO_MAX_TOKENS = 768
# min pixels per video frame
VIDEO_MIN_PIXELS = VIDEO_MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 32 * 28 * 28 = 25,088
# max pixels per video frame
VIDEO_MAX_PIXELS = VIDEO_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 768 * 28 * 28 = 602,112
# max total pixels per video
VIDEO_TOTAL_PIXELS = 65536 * IMAGE_FACTOR * IMAGE_FACTOR # 65,536 * 28 * 28 = 51,380,224
# default fps
FPS = 2.0

FAST_TOKEN_RATIO = 0.3

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    # if int(height < factor//4) + int(width < factor//4):
    #     raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor//4}")

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return max(h_bar, factor), max(w_bar, factor)


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR, is_video = False, **kwargs) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")  ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        # 以image list形式传入的视频
        if is_video:
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            max_pixels = ele.get("max_pixels", VIDEO_MAX_PIXELS)
        else:
            min_pixels = ele.get("min_pixels", MIN_PIXELS)
            max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
        ele: dict,
        total_frames: int,
        video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    # assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    # if "nframes" in ele:
    #     nframes = ele["nframes"]
    # else:
    fps = ele.get("fps", FPS) # 应该是走的默认FPS，按照每秒抽两帧来算
    fps = min(fps, video_fps) # 注意，这里的video_fps是真实的后验FPS
    # 计算每帧使用最少token的情况下，能吃多少帧，这个是用来兜底的
    # 是否允许用户低于这个限制？
    # print("cjx smart nfram debug VIDEO_TOTAL_PIXELS token num in llm side is {}".format(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS)//28//28))
    max_frames = int(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS) / ele.get("min_pixels", VIDEO_MIN_PIXELS))
    fps_nframes = int(total_frames / video_fps * fps) # 换算为秒数，之后计算希望抽多少帧
    nframes = min(fps_nframes, max_frames)
    return nframes


def _read_video_torchvision(
        ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    # process video url
    st = time.time()
    if isinstance(ele["video"], str):
        video_path = ele["video"]
        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            if "http://" in video_path or "https://" in video_path:
                warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
            if "file://" in video_path:
                video_path = video_path[7:]
        video, audio, info = io.read_video(
            video_path,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        total_frames, video_fps = video.size(0), info["video_fps"]
        logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    elif isinstance(ele["video"], bytes):
        video_reader = torchvision.io.VideoReader(ele["video"], "video")
        video_meta = video_reader.get_metadata()["video"]

        start_ptr = ele.get("video_start", 0.0)
        end_pts = ele.get("video_end", video_meta["duration"][-1])
        video = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_pts, video_reader.seek(start_ptr)):
            video.append(frame['data'])
        video = torch.stack(video)
        total_frames, video_fps = video.size(0), video_meta["fps"][-1]
        logger.info(f"torchvision:  {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    return video


def _read_video_torchvision_slowfast(
        ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    print("torchvision")
    # process video url
    st = time.time()
    if isinstance(ele["video"], str):
        video_path = ele["video"]
        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            if "http://" in video_path or "https://" in video_path:
                warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
            if "file://" in video_path:
                video_path = video_path[7:]
        video, audio, info = io.read_video(
            video_path,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        total_frames, video_fps = video.size(0), info["video_fps"]
        logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    elif isinstance(ele["video"], bytes):
        video_reader = torchvision.io.VideoReader(ele["video"], "video")
        video_meta = video_reader.get_metadata()["video"]

        start_ptr = ele.get("video_start", 0.0)
        end_pts = ele.get("video_end", video_meta["duration"][-1])
        video = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_pts, video_reader.seek(start_ptr)):
            video.append(frame['data'])
        video = torch.stack(video)
        total_frames, video_fps = video.size(0), video_meta["fps"][-1]
        logger.info(f"torchvision:  {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    total_frames_time_position = torch.FloatTensor([(1 / video_fps) * i for i in range(total_frames)])
    total_nframes_number = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    selected_indices = torch.linspace(0, total_frames - 1, total_nframes_number).round().long()
    selected_time_position = total_frames_time_position[selected_indices]
    selected_frames = video[selected_indices]

    ##### extract key frames start ######
    # Step#1，对选中的图，假设都为slow，先resize到28*28的倍数，但是会先在256视图下去进行比较
    _, _, height, width = selected_frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=ele.get("min_pixels", VIDEO_MIN_PIXELS),
        max_pixels=256 * IMAGE_FACTOR * IMAGE_FACTOR,
    )
    
    selected_frames_extract = nn.functional.interpolate(
        selected_frames,
        [resized_height, resized_width],
        mode="bicubic",
        antialias=True,
    ).float()
    
    # Step#2 对选中的图，筛选出其中关键帧部分，其余为fast
    slow_frames, fast_frames, slow_fast_order = extract_slow_fast_frames(selected_frames, selected_frames_extract)
    ##### extract key frames start ######

    return slow_frames, fast_frames, selected_time_position.tolist(), slow_fast_order



def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
        ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    st = time.time()
    if isinstance(ele["video"], bytes):
        video_path = ""
        fp = py_io.BytesIO(ele["video"])
        vr = decord.VideoReader(fp)
    else:
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes, fps_ratio = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video, fps_ratio


def cal_sim_pixel(frame1, frame2, patch_size=28, pixel_threshold=5, patch_sim=0.98):
    assert frame1.dim() == 3 and frame2.dim() == 3, "输入必须是3D张量 [C, H, W]"
    
    channel, height, width = frame1.shape
    unchanged_threshold = patch_sim * channel * patch_size * patch_size
    
    diff = (frame1 - frame2).abs()
    unchanged_pixel = rearrange(diff < pixel_threshold, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    unchanged = (unchanged_pixel.sum(-1) < unchanged_threshold)
    
    return unchanged.float().mean().item()


def cal_sim_cosine(frame1, frame2, patch_size=28, cos_threshold = 0.7, epsilon=1e-8):
    assert frame1.dim() == 3 and frame2.dim() == 3, "输入必须是3D张量 [C, H, W]"
    
    patch1 = rearrange(frame1, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(frame2, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    
    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)

    
    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01) # 全黑图
    
    similar = torch.ones_like(cos_sim)  # 默认全部相似
    
    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > cos_threshold).float()
    
    return similar[non_zero_mask].float().mean().item()

def cal_sim_cosine_hsv(frame1, frame2, patch_size=28, cos_threshold=0.7, epsilon=1e-8):
    assert frame1.dim() == 3 and frame2.dim() == 3, "输入必须是3D张量 [C, H, W]"
    
    # 将PyTorch张量转换为OpenCV格式的numpy数组
    def to_numpy_cvt(tensor):
        # 确保张量在CPU上并转换为HWC格式
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            tensor = (tensor).astype(np.uint8)
        # 转换为HSV颜色空间
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2HSV)
    
    # 转换颜色空间
    frame1_hsv = to_numpy_cvt(frame1)
    frame2_hsv = to_numpy_cvt(frame2)
    
    # 将HSV图像转回PyTorch张量
    frame1_tensor = torch.from_numpy(frame1_hsv).permute(2, 0, 1).to(frame1.device).float()
    frame2_tensor = torch.from_numpy(frame2_hsv).permute(2, 0, 1).to(frame2.device).float()
    
    # 分块处理
    patch1 = rearrange(frame1_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(frame2_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    
    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)
    
    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)  # 全黑图
    
    similar = torch.ones_like(cos_sim)  # 默认全部相似
    
    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > cos_threshold).float()
    
    return similar[non_zero_mask].float().mean().item()


def extract_key_frame(frames, patch_size=28, threshold=0.9):
    assert frames.dim() == 4, "输入必须是4D张量 [N, C, H, W]"
    
    key_frame_indices = [0]
    last_key_frame = frames[0]
    similarity_list = []
    for i in range(1, frames.size(0)):
        current_frame = frames[i]
        
        global_sim = cal_sim_cosine_hsv(last_key_frame, current_frame)
        similarity_list.append(global_sim)
        if global_sim < threshold:
            key_frame_indices.append(i)
            last_key_frame = current_frame  # 更新关键帧

    # print("cjx similarity debug {}".format(similarity_list))

    return key_frame_indices


def extract_slow_fast_frames(selected_frames, selected_frames_extract):
    # print("selected_frames size {}, selected_frames_extract size {}".format(selected_frames.size(), selected_frames_extract.size()))
    slow_indices = extract_key_frame(selected_frames_extract)

    slow_mask = torch.zeros(size=(selected_frames.size(0), ), dtype=torch.bool)
    slow_mask[slow_indices] = True

    slow_frames = selected_frames[slow_mask]
    fast_frames = selected_frames[~slow_mask]

    slow_fast_order = torch.ones(size=(selected_frames.size(0), ), dtype=torch.long)
    slow_fast_order[slow_indices] = 0

    return slow_frames, fast_frames, slow_fast_order.tolist()


def _read_video_decord_slowfast(
        ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    st = time.time()
    if isinstance(ele["video"], bytes):
        video_path = ""
        fp = py_io.BytesIO(ele["video"])
        vr = decord.VideoReader(fp)
    else:
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    # timestamp start from 0.0
    total_frames_time_position = torch.FloatTensor([(1 / video_fps) * i for i in range(total_frames)])
    # print(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    
    total_nframes_number = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    selected_indices = torch.linspace(0, total_frames - 1, total_nframes_number).round().long()
    selected_frames = vr.get_batch(selected_indices.tolist()).asnumpy()
    selected_frames = torch.tensor(selected_frames).permute(0, 3, 1, 2)
    selected_time_position = total_frames_time_position[selected_indices]

    ##### extract key frames start ######
    # Step#1，对选中的图，假设都为slow，先resize到28*28的倍数，但是会先在256视图下去进行比较
    _, _, height, width = selected_frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=ele.get("min_pixels", VIDEO_MIN_PIXELS),
        max_pixels=256 * IMAGE_FACTOR * IMAGE_FACTOR,
    )
    
    selected_frames_extract = nn.functional.interpolate(
        selected_frames,
        [resized_height, resized_width],
        mode="bicubic",
        antialias=True,
    ).float()
    
    # Step#2 对选中的图，筛选出其中关键帧部分，其余为fast
    slow_frames, fast_frames, slow_fast_order = extract_slow_fast_frames(selected_frames, selected_frames_extract)
    ##### extract key frames start ######

    return slow_frames, fast_frames, selected_time_position.tolist(), slow_fast_order


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "slowfast_torchvision": _read_video_torchvision_slowfast,
    "slowfast_decord": _read_video_decord_slowfast,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    # return video_reader_backend
    # Hack
    return f"slowfast_{video_reader_backend}"



def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, slowfast: bool = True, **kwargs) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str) or isinstance(ele["video"], bytes):
        video_reader_backend = get_video_reader_backend()
        slow_frames, fast_frames, time_position, slow_fast_order = VIDEO_READER_BACKENDS[video_reader_backend](ele)

    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = []
        for video_element in ele["video"]:
            # preprocess images
            if isinstance(video_element, dict):
                images.append(fetch_image(video_element, size_factor=image_factor, is_video = True))
            else:
                images.append(
                    fetch_image({"image": video_element, **process_info}, size_factor=image_factor, is_video = True)
                )
        total_frames = len(images)
        
        tensor_images = [torch.from_numpy(np.array(pil_image)).permute(2, 0, 1) for pil_image in images]
        tensor_images = torch.stack(tensor_images, dim=0)

        slow_frames, fast_frames, slow_fast_order = extract_slow_fast_frames(tensor_images, tensor_images.clone())
        time_position = None
    
    ### 计算slow fast的token量 begin ###
    slow_number = slow_frames.size(0)
    if fast_frames.size(0) == 0:
        fast_frames = None
    fast_number = fast_frames.size(0) if fast_frames is not None else 0

    ####### 二分，精准，但暂时弃用 #####
    min_pixels = max(int(ele.get("min_pixels", VIDEO_MIN_PIXELS)), VIDEO_MIN_PIXELS)
    min_tokens = int(min_pixels / IMAGE_FACTOR / IMAGE_FACTOR)
    left = min_pixels / IMAGE_FACTOR / IMAGE_FACTOR
    right = ele.get("max_pixels", VIDEO_MAX_PIXELS) / IMAGE_FACTOR / IMAGE_FACTOR
    def _estimate_total_pixels(tokens_per_frame):
        return slow_number * tokens_per_frame * IMAGE_FACTOR * IMAGE_FACTOR + \
            fast_number * max(int(FAST_TOKEN_RATIO * tokens_per_frame), min_tokens) * IMAGE_FACTOR * IMAGE_FACTOR

    while left < right:
        mid = int(left+right) // 2
        if _estimate_total_pixels(mid) > ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS):
            right = mid
        else:
            left = mid + 1
    slow_max_pixels = left * IMAGE_FACTOR * IMAGE_FACTOR
    ######

    # accum_slow_number = slow_number + FAST_TOKEN_RATIO * fast_number
    # rough_slow_token = int(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS) / accum_slow_number / IMAGE_FACTOR / IMAGE_FACTOR)
    # slow_max_pixels = max(rough_slow_token * IMAGE_FACTOR * IMAGE_FACTOR, VIDEO_MIN_PIXELS)

    # fast tokens下限为min_tokens，极端情况下slow和fast数量一样
    # fast_max_pixels = max(int(FAST_TOKEN_RATIO * left), min_tokens) * IMAGE_FACTOR * IMAGE_FACTOR
    ### 计算slow fast的token量 end ###

    nframes, _, height, width = slow_frames.shape

    #### slow part ######
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=slow_max_pixels,
    )
    real_slow_token = resized_height * resized_width / IMAGE_FACTOR / IMAGE_FACTOR
    fast_max_pixels = max(int(real_slow_token * FAST_TOKEN_RATIO) * IMAGE_FACTOR * IMAGE_FACTOR, VIDEO_MIN_PIXELS)
    fast_resized_height, fast_resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=fast_max_pixels,
    )


    if time_position is None: # image list
        slow_frames = []
        fast_frames = []
        for idx, value in enumerate(slow_fast_order):
            if value == 0:
                slow_frames.append(images[idx].resize((resized_width, resized_height)))
            else:
                fast_frames.append(images[idx].resize((fast_resized_width, fast_resized_height)))
        
        if len(fast_frames) == 0:
            fast_frames = None
        
        if len(slow_frames) > 1:
            pass
            # 避免太多的 pad log
            # print("cjx vl debug for image list, slow frames {}, fast frames {}, slow token is {}, fast token is {}".format(len(slow_frames), len(fast_frames) if fast_frames is not None else 0, resized_height*resized_width//28//28, fast_resized_height*fast_resized_width//28//28))
        assert (len(slow_frames) if slow_frames is not None else 0) + (len(fast_frames) if fast_frames is not None else 0) == len(slow_fast_order)
        return slow_frames, fast_frames, slow_fast_order

    else: # mp4
        slow_frames = nn.functional.interpolate(
            slow_frames,
            [resized_height, resized_width],
            mode="bicubic",
            antialias=True,
        ).float()
        slow_frames = list(slow_frames.split(1, dim=0))
        #### fast part ######
        if fast_frames is not None:
            fast_frames = nn.functional.interpolate(
                fast_frames,
                [fast_resized_height, fast_resized_width],
                mode="bicubic",
                antialias=True,
            ).float()
            fast_frames = list(fast_frames.split(1, dim=0))
        if random.randint(0, 10000) < 5:
            print("cjx vl debug for mp4, slow frames {}, fast frames {}, slow token is {}, fast token is {}, video dir".format(len(slow_frames), len(fast_frames) if fast_frames is not None else 0, resized_height*resized_width//28//28, fast_resized_height*fast_resized_width//28//28), ele["video"])
        
        assert (len(slow_frames) if slow_frames is not None else 0) + (len(fast_frames) if fast_frames is not None else 0) == len(slow_fast_order)
        return slow_frames, fast_frames, time_position, slow_fast_order
    

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
        conversations: list[dict] | list[list[dict]] = None, vision_infos: list[dict] = None,
        image_factor: int = IMAGE_FACTOR, **kwargs
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    assert conversations is not None or vision_infos is not None
    torch.set_num_threads(1)
    image = kwargs.get("image", True)
    video = kwargs.get("video", True)

    if vision_infos is None:
        vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if image or video:
            if image and ("image" in vision_info or "image_url" in vision_info):
                image_inputs.append(fetch_image(vision_info, image_factor, **kwargs))
            elif video and "video" in vision_info:
                if isinstance(vision_info["video"], str) and "480p_60s_4fps_v2" in vision_info["video"]:
                    path = vision_info["video"]
                    pid_str = osp.basename(osp.splitext(path)[0])
                    if not osp.exists(path):
                        post = str(int(pid_str[-4:]))
                        path = path.replace("480p_60s_4fps_v2", "480p_60s_4fps_0215_0316/{}".format(post))
                    vision_info["video"] = path
                video_inputs.append(fetch_video(vision_info, image_factor, **kwargs))
            else:
                raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs

def get_rope_index_slowfast(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    fast_video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: Optional[int] = None,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    vision_start_token_id: Optional[int] = None,
    fast_video_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index, fast_video_index = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]

            if image_grid_thw is not None:
                image_nums = image_grid_thw.size(0) # 这里实际上是图片的数量
            else:
                image_nums = 0

            if video_grid_thw is not None:
                video_nums = video_grid_thw.size(0) # 这里实际上是slow_frame的数量
            else:
                video_nums = 0

            if fast_video_grid_thw is not None:
                fast_video_nums = fast_video_grid_thw.size(0) # 这里实际上是fast_frame的数量
            else:
                fast_video_nums = 0

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos_frames, remain_fast_videos_frames = image_nums, video_nums, fast_video_nums
            # remain_images, remain_videos = image_nums, video_grid_thw.size(0)//2
            for _ in range(image_nums + video_nums + fast_video_nums):

                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                if video_token_id in input_tokens and remain_videos_frames > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                
                if fast_video_token_id in input_tokens and remain_fast_videos_frames > 0:
                    ed_fast_video = input_tokens.index(fast_video_token_id, st)
                else:
                    ed_fast_video = len(input_tokens) + 1
                
                if ed_image < min(ed_video, ed_fast_video):
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                elif ed_video < min(ed_image, ed_fast_video):
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos_frames -= 1
                    ed = ed_video
                
                elif ed_fast_video < min(ed_image, ed_video):
                    t, h, w = (
                        fast_video_grid_thw[fast_video_index][0],
                        fast_video_grid_thw[fast_video_index][1],
                        fast_video_grid_thw[fast_video_index][2],
                    )
                    fast_video_index += 1
                    remain_fast_videos_frames -= 1
                    ed = ed_fast_video


                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                t_index = expanded_range.flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )

        return position_ids