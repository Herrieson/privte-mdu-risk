"""Shared helpers for PriVTE algorithm modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CONTAINER_BOX_TYPES = {
    "moov",
    "trak",
    "mdia",
    "minf",
    "stbl",
    "edts",
    "dinf",
    "udta",
}


def collect_clips(person_record: dict[str, Any]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for session in person_record.get("sessions", []):
        clips.extend(session.get("clips", []))
    return clips


def count_available(clips: list[dict[str, Any]], modality: str) -> int:
    return sum(1 for clip in clips if clip.get(modality, {}).get("available"))


def count_nonempty_json(clips: list[dict[str, Any]], modality: str) -> int:
    return sum(
        1
        for clip in clips
        if clip.get(modality, {}).get("available")
        and not clip.get(modality, {}).get("is_empty", True)
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 3)


def round_mb(size_bytes: int | float | None) -> float | None:
    if size_bytes is None:
        return None
    return round(float(size_bytes) / (1024 * 1024), 2)


def video_file_entries(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for clip in clips:
        video = clip.get("video", {})
        if not video.get("available"):
            continue
        for file_info in video.get("files", []):
            if file_info.get("path"):
                entries.append(file_info)
    return entries


def evenly_sample(items: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    if max_count == 1:
        return items[:1]
    step = (len(items) - 1) / (max_count - 1)
    return [items[round(index * step)] for index in range(max_count)]


def size_bucket(size_mb: float | None) -> str:
    if size_mb is None:
        return "unknown"
    if size_mb < 5:
        return "<5MB"
    if size_mb < 20:
        return "5-20MB"
    if size_mb < 50:
        return "20-50MB"
    return ">=50MB"


def duration_bucket(duration_sec: float | None) -> str:
    if duration_sec is None:
        return "unknown"
    if duration_sec < 10:
        return "<10s"
    if duration_sec < 30:
        return "10-30s"
    if duration_sec < 90:
        return "30-90s"
    return ">=90s"


def resolution_bucket(width: float | int | None, height: float | int | None) -> str:
    if not width or not height:
        return "unknown"
    pixels = int(width) * int(height)
    if pixels < 640 * 480:
        return "below_vga"
    if pixels < 1280 * 720:
        return "vga_to_720p"
    if pixels < 1920 * 1080:
        return "720p_to_1080p"
    return "1080p_or_above"


def fps_bucket(fps: float | None) -> str:
    if not fps or fps <= 0:
        return "unknown"
    if fps < 15:
        return "<15fps"
    if fps < 24:
        return "15-24fps"
    if fps <= 30:
        return "24-30fps"
    return ">30fps"


def parse_mvhd_duration(payload: bytes) -> float | None:
    if len(payload) < 20:
        return None
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return None
        timescale = int.from_bytes(payload[20:24], "big")
        duration = int.from_bytes(payload[24:32], "big")
    else:
        timescale = int.from_bytes(payload[12:16], "big")
        duration = int.from_bytes(payload[16:20], "big")
    if timescale <= 0 or duration <= 0:
        return None
    return duration / timescale


def parse_tkhd_dimensions(payload: bytes) -> tuple[float | None, float | None]:
    if len(payload) < 8:
        return None, None
    width_fixed = int.from_bytes(payload[-8:-4], "big")
    height_fixed = int.from_bytes(payload[-4:], "big")
    width = width_fixed / 65536
    height = height_fixed / 65536
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def probe_mp4_box_metadata(path: Path) -> dict[str, Any]:
    """Read coarse MP4 container metadata without decoding frames."""

    result: dict[str, Any] = {
        "readable": False,
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
        "backend": "mp4_box_parser",
    }

    try:
        file_size = path.stat().st_size
        with path.open("rb") as file:

            def read_box_header(end: int) -> tuple[int, str, int, int] | None:
                start = file.tell()
                if start + 8 > end:
                    return None
                header = file.read(8)
                if len(header) < 8:
                    return None
                size = int.from_bytes(header[:4], "big")
                box_type = header[4:8].decode("latin-1", errors="replace")
                header_size = 8
                if size == 1:
                    large_size_raw = file.read(8)
                    if len(large_size_raw) < 8:
                        return None
                    size = int.from_bytes(large_size_raw, "big")
                    header_size = 16
                elif size == 0:
                    size = end - start
                if size < header_size:
                    return None
                box_end = min(start + size, end)
                return start, box_type, header_size, box_end

            def parse_range(end: int, depth: int = 0) -> None:
                while file.tell() + 8 <= end:
                    header = read_box_header(end)
                    if header is None:
                        return
                    start, box_type, header_size, box_end = header
                    payload_size = max(0, box_end - start - header_size)

                    if box_type == "mvhd":
                        duration = parse_mvhd_duration(file.read(payload_size))
                        if duration is not None:
                            result["duration_sec"] = duration
                            result["readable"] = True
                    elif box_type == "tkhd":
                        width, height = parse_tkhd_dimensions(file.read(payload_size))
                        current_pixels = (result.get("width") or 0) * (
                            result.get("height") or 0
                        )
                        new_pixels = (width or 0) * (height or 0)
                        if (
                            width is not None
                            and height is not None
                            and new_pixels > current_pixels
                        ):
                            result["width"] = width
                            result["height"] = height
                            result["readable"] = True
                    elif box_type in CONTAINER_BOX_TYPES and depth < 8:
                        parse_range(box_end, depth + 1)

                    file.seek(box_end)

            parse_range(file_size)
    except OSError:
        return result

    return result
