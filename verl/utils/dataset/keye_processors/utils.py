from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
import itertools
from functools import lru_cache
from io import BytesIO

import requests
import torch
#from examples.keye.processors.base import SampleType
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Any, Optional, List, Tuple

import io as py_io
import os.path as osp


IMAGE_FACTOR = int(os.environ.get("KEYE_IMAGE_FACTOR", 28))
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", 4 * IMAGE_FACTOR * IMAGE_FACTOR))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 16384 * IMAGE_FACTOR * IMAGE_FACTOR))
print(f"recovlm IMAGE_FACTOR: {IMAGE_FACTOR} {MIN_PIXELS=} {MAX_PIXELS=}")

MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = int(os.environ.get("VIDEO_TOTAL_PIXELS", 24576 * 28 * 28))
print(f"recovlm VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")
# VIDEO_TOTAL_PIXELS = 24576 * 28 * 28

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = int(os.environ.get("FPS_MAX_FRAMES", 768))
print(f"recovlm FPS_MAX_FRAMES: {FPS_MAX_FRAMES}")

FAST_TOKEN_RATIO = 0.5
VIDEO_MAX_TOKENS = 768
SLOW_FAST_FRAMES_RATIO = 0.8


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
    print(f'{height=} {width=} {factor=} {min_pixels=} {max_pixels=} {h_bar=} {w_bar=}')
    return h_bar, w_bar


def fetch_image(
    ele: dict[str, str | Image.Image],
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
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
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")  # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
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
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> torch.Tensor:
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
                warnings.warn(
                    "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
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
        logger.info(
            f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

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


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
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
    print(
        f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
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
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str) or isinstance(ele["video"], bytes):
        video_reader_backend = get_video_reader_backend()
        video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        if image_factor is None:
            return None

        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels /
                         nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = []
        for video_element in ele["video"]:
            # preprocess images
            if isinstance(video_element, dict):
                images.append(fetch_image(video_element, size_factor=image_factor))
            else:
                images.append(
                    fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
                )
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def calculate_image_seq_len(
    height: int,
    width: int,
    patch_size: int,
    merge_size: int,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    height, width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return (height // patch_size) * (width // patch_size) // (merge_size ** 2)


def calculate_video_seq_len(
    num_frames: int,
    height: int,
    width: int,
    temporal_patch_size: int,
    patch_size: int,
    merge_size: int,
    fps: float = FPS,
    video_total_pixels: int = VIDEO_TOTAL_PIXELS,
    video_min_pixels: int = VIDEO_MIN_PIXELS,
    video_max_tokens: int = VIDEO_MAX_TOKENS,
):
    max_frames = int(video_total_pixels / video_min_pixels)
    fps = min(fps, FPS)
    fps_nframes = int(num_frames / fps * FPS)
    num_frames = min(fps_nframes, max_frames)
    height, width = smart_resize(height, width, min_pixels=video_min_pixels, max_pixels=VIDEO_MAX_PIXELS)
    token_per_frame = (height // patch_size) * (width // patch_size) // (merge_size ** 2)
    token_per_frame = min(video_max_tokens, token_per_frame)
    t_grid = (num_frames + temporal_patch_size - 1) // temporal_patch_size
    return t_grid * token_per_frame


def calculate_video_seq_len_slow_fast(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    merge_size: int,
    fps: float = FPS,
    video_total_pixels: int = VIDEO_TOTAL_PIXELS,
    video_min_pixels: int = VIDEO_MIN_PIXELS,
    video_max_tokens: int = VIDEO_MAX_TOKENS,
    fast_token_ratio: float = FAST_TOKEN_RATIO,
    slow_fast_frames_ratio: float = SLOW_FAST_FRAMES_RATIO,
):
    max_frames = int(video_total_pixels / video_min_pixels)
    fps = min(fps, FPS)
    fps_nframes = int(num_frames / fps * fps)
    num_frames = min(fps_nframes, max_frames)
    # slow token nums
    slow_token_nums = (height // patch_size) * (width // patch_size) // (merge_size ** 2)
    slow_token_nums = min(video_max_tokens, slow_token_nums)

    # fast token nums
    fast_token_nums = int(slow_token_nums * fast_token_ratio)

    num_fast_frames = slow_fast_frames_ratio * num_frames
    num_slow_frames = num_frames - num_fast_frames
    return int(num_slow_frames * slow_token_nums + num_fast_frames * fast_token_nums)

def extract_vision_meta(
    vision_infos: list[dict],
    metadata: dict[str, Any],
    path_to_name: Optional[dict[str, str]] = None
) -> tuple[list[dict], list[dict]]:
    image_metas, video_metas = [], []
    for vision_info in vision_infos:
        if "image" in vision_info:
            images_info = metadata["images_info"]
            image_meta = None
            if images_info is not None:
                image_meta = images_info.get(vision_info["image"])
                if image_meta is None and path_to_name is not None:
                    name = path_to_name.get(vision_info["image"])
                    if name is not None:
                        image_meta = images_info.get(name)
            if image_meta is None:
                logger.warning("cannot find metadata of image %s, skipping", vision_info["image"])
                continue
            image_meta["type"] = "image"
            image_metas.append(image_meta)
        elif "video" in vision_info:
            video_info = metadata["video_info"]
            video_meta = None
            if video_info is not None:
                video_meta = video_info.get(vision_info["video"])
                if video_meta is None and path_to_name is not None:
                    name = path_to_name.get(vision_info["video"])
                    if name is not None:
                        video_meta = video_info.get(name)
            if video_meta is None:
                logger.warning("cannot find metadata of video %s, skipping", vision_info["video"])
                continue
            video_meta["type"] = "video"
            video_metas.append(video_meta)
    return image_metas, video_metas


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
    conversations: list[dict] | list[list[dict]] = None,
    vision_infos: list[dict] = None,
    image_factor: int = IMAGE_FACTOR,
    **kwargs,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    assert conversations is not None or vision_infos is not None

    image = kwargs.get("image", True)
    video = kwargs.get("video", True)

    if vision_infos is None:
        vision_infos = extract_vision_info(conversations)
    # Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if image or video:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info, image_factor))
            elif "video" in vision_info:
                if isinstance(vision_info["video"], str) and "480p_60s_4fps_v2" in vision_info["video"]:
                    path = vision_info["video"]
                    pid_str = osp.basename(osp.splitext(path)[0])
                    if not osp.exists(path):
                        post = str(int(pid_str[-4:]))
                        path = path.replace("480p_60s_4fps_v2",
                                            "480p_60s_4fps_0215_0316/{}".format(post))
                    vision_info["video"] = path
                video_inputs.append(fetch_video(vision_info, image_factor))
            else:
                raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs


def cut_sample(inputs: SampleType, packable_length: int, vision_start_token_id: int, vision_end_token_id: int) -> SampleType:
    inputs["input_ids"] = inputs["input_ids"][:, :packable_length]
    inputs["loss_mask"] = inputs["loss_mask"][:, :packable_length]

    inputs["position_ids"] = inputs["position_ids"][..., :packable_length]

    vision_starts = torch.nonzero(inputs["input_ids"][0] == vision_start_token_id)
    vision_ends = torch.nonzero(inputs["input_ids"][0] == vision_end_token_id)

    if len(vision_starts) and len(vision_starts) > len(vision_ends):  # 说明图片不完整
        # 继续截断,截断到vision_starts token,因为vision_start之后的内容都不会有loss
        inputs["input_ids"] = inputs["input_ids"][:, :vision_starts[-1]]
        inputs["loss_mask"] = inputs["loss_mask"][:, :vision_starts[-1]]
        inputs["position_ids"] = inputs["position_ids"][..., :vision_starts[-1]]

    if 'image_grid_thw' in inputs and len(inputs["pixel_values"]) and 'video_grid_thw' in inputs and len(inputs["pixel_values_videos"]):
        raise Exception("Unexpected inputs: there are both pixel_values and pixel_values_videos: {}/{}".format(
            inputs["pixel_values"].shape, inputs["pixel_values_videos"].shape))

    if 'image_grid_thw' in inputs:  # 如果有图片
        n_tokens = 0
        for i in range(len(vision_ends), len(inputs["image_grid_thw"])):
            n_tokens_hw = inputs["image_grid_thw"][i]
            n_tokens += n_tokens_hw[1] * n_tokens_hw[2]

        if n_tokens:
            inputs["pixel_values"] = inputs["pixel_values"][:-n_tokens]
        inputs["image_grid_thw"] = inputs["image_grid_thw"][:len(vision_ends)]

    elif 'video_grid_thw' in inputs:  # 如果有视频
        n_tokens = 0
        for i in range(len(vision_ends), len(inputs["video_grid_thw"])):
            n_tokens_hw = inputs["video_grid_thw"][i]
            n_tokens += n_tokens_hw[0] * n_tokens_hw[1] * n_tokens_hw[2]

        if n_tokens:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"][:-n_tokens]
        inputs["video_grid_thw"] = inputs["video_grid_thw"][:len(vision_ends)]
        inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"][:len(vision_ends)]

        if len(inputs["pixel_values_videos"]) == 0:
            del inputs["pixel_values_videos"]
            del inputs["video_grid_thw"]
            del inputs["second_per_grid_ts"]
    return inputs


def get_assistant_mask(batch_input_ids: torch.Tensor,
                       start_pattern: Optional[List[int]],
                       end_pattern: Optional[List[int]]):
    if not start_pattern:
        start_pattern = [151644, 77091, 198]
    if not end_pattern:
        end_pattern = [151645, 198]

    masks = []
    for input_ids in batch_input_ids:
        mask = []
        assistant_start = []
        assistant_end = []
        to_mask = False
        for _id in input_ids:
            mask.append(int(to_mask))
            if not to_mask:
                if _id in start_pattern:
                    assistant_start.append(_id.item())
                else:
                    assistant_start = []
                if assistant_start[-3:] == start_pattern:
                    to_mask = True
                    assistant_start = []
            else:
                if _id in end_pattern:
                    assistant_end.append(_id.item())
                else:
                    assistant_end = []
                if assistant_end[-2:] == end_pattern:
                    to_mask = False
                    assistant_end = []
        masks.append(mask)
    return torch.tensor(masks)


def get_rope_index(
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        spatial_merge_size: Optional[int] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        vision_start_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    # spatial_merge_size = self.config.vision_config.spatial_merge_size
    # image_token_id = self.config.image_token_id
    # video_token_id = self.config.video_token_id
    # vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device)
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + \
                    1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(
                    text_len).view(1, -1).expand(3, -1) + st_idx)


                # hjm dbg
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                _second_per_grid_t = 1
                _tokens_per_second = 2
                time_tensor = expanded_range * _second_per_grid_t * _tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()
                # dbg end

                #t_index = torch.arange(llm_grid_t).view(-1,
                #                                        1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(
                    1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(
                    1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack(
                    [t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + \
                    1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(
                    text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] ==
                         1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas,
            device=input_ids.device).unsqueeze(1)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(
                0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[
                0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids
