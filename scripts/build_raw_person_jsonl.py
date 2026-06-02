#!/usr/bin/env python3
"""Build an internal raw person-level JSONL file from local PriVTE data.

The output is intentionally raw/internal: it keeps source paths, questionnaire
answers, app names, exact heart-rate records, and raw label-sheet fields.
Do not publish this JSONL without a later privacy filtering step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

PATH_LIKE_LABEL_COLUMNS = (
    "视频路径",
    "数据所在文件夹",
    "心率文件路径",
    "心率图像路径",
    "问卷内容路径",
    "问卷内容json路径",
)

SUMMARY_LABEL_COLUMNS = (
    "情绪状态",
    "是否沉迷",
    "沉迷等级",
    "应用类型",
    "心率置信度",
    "视频置信度",
    "问卷置信度",
    "应用置信度",
    "标注人数",
    "标注完整性",
)


def ns(tag: str, namespace: str = SPREADSHEET_NS) -> str:
    return f"{{{namespace}}}{tag}"


def column_letters(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def column_index(letters: str) -> int:
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - 64)
    return index - 1


def resolve_workbook_target(target: str) -> str:
    target = target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return posixpath.normpath(posixpath.join("xl", target))


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet with the standard library.

    This handles the simple workbook shape used by data/label.xlsx and avoids
    adding openpyxl/pandas as project dependencies.
    """
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(ns("si")):
                shared_strings.append("".join(text.text or "" for text in item.iter(ns("t"))))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }

        first_sheet = workbook.find(f".//{ns('sheet')}")
        if first_sheet is None:
            return []
        relationship_id = first_sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
        if relationship_id is None or relationship_id not in rel_targets:
            raise ValueError(f"Cannot resolve first worksheet target in {path}")
        worksheet_path = resolve_workbook_target(rel_targets[relationship_id])

        worksheet = ET.fromstring(archive.read(worksheet_path))
        raw_rows: list[list[str]] = []
        for row in worksheet.findall(f".//{ns('row')}"):
            values_by_index: dict[int, str] = {}
            for cell in row.findall(ns("c")):
                ref = cell.attrib.get("r", "")
                letters = column_letters(ref)
                if not letters:
                    continue
                index = column_index(letters)
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join(text.text or "" for text in cell.iter(ns("t")))
                else:
                    raw_value = cell.find(ns("v"))
                    if raw_value is not None and raw_value.text is not None:
                        if cell_type == "s":
                            value = shared_strings[int(raw_value.text)]
                        else:
                            value = raw_value.text
                values_by_index[index] = value
            if values_by_index:
                raw_rows.append(
                    [values_by_index.get(i, "") for i in range(max(values_by_index) + 1)]
                )

        if not raw_rows:
            return []
        column_count = max(len(row) for row in raw_rows)
        for row in raw_rows:
            row.extend([""] * (column_count - len(row)))

        headers = raw_rows[0]
        return [dict(zip(headers, row)) for row in raw_rows[1:]]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rel_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def parse_segment_index(segment_dir: Path) -> int | None:
    match = re.search(r"_(\d{3})$", segment_dir.name)
    if not match:
        return None
    return int(match.group(1))


def get_video_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
    }
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        metadata["metadata_status"] = "cv2_unavailable"
        metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"
        return metadata

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        metadata["metadata_status"] = "unreadable"
        return metadata

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    metadata.update(
        {
            "metadata_status": "ok",
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": fps,
            "frame_count": int(frame_count),
            "duration_sec": frame_count / fps if fps else None,
        }
    )
    capture.release()
    return metadata


def build_label_index(label_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    indexed: list[tuple[str, dict[str, str]]] = []
    for row in label_rows:
        searchable = " ".join(str(row.get(column, "")) for column in PATH_LIKE_LABEL_COLUMNS)
        indexed.append((searchable, row))
    return indexed


def labels_for_segment(
    segment_name: str,
    indexed_labels: list[tuple[str, dict[str, str]]],
) -> list[dict[str, str]]:
    return [row for searchable, row in indexed_labels if segment_name in searchable]


def risk_label_from_raw(row: dict[str, str]) -> str:
    is_addiction = (row.get("是否沉迷") or "").strip()
    level = (row.get("沉迷等级") or "").strip()
    if is_addiction == "否":
        return "no_observed_risk"
    if is_addiction == "是" and level == "轻度":
        return "mild_risk"
    if is_addiction == "是" and level == "中度":
        return "moderate_risk"
    if is_addiction == "是" and level == "重度":
        return "high_risk"
    return "insufficient_evidence"


def summarize_labels(label_rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": len(label_rows),
        "risk_label_counts": dict(Counter(risk_label_from_raw(row) for row in label_rows)),
        "raw_column_counts": {},
    }
    for column in SUMMARY_LABEL_COLUMNS:
        summary["raw_column_counts"][column] = dict(
            Counter((row.get(column) or "<blank>") for row in label_rows)
        )
    participant_ids = sorted({row.get("参与者编号", "") for row in label_rows if row.get("参与者编号")})
    summary["participant_ids_from_labels"] = participant_ids
    return summary


def build_person_record(
    person_dir: Path,
    repo_root: Path,
    indexed_labels: list[tuple[str, dict[str, str]]],
    include_video_metadata: bool,
) -> dict[str, Any]:
    segment_dirs = sorted(
        [path for path in person_dir.iterdir() if path.is_dir()],
        key=lambda path: (parse_segment_index(path) is None, parse_segment_index(path) or 0, path.name),
    )

    questionnaire_payloads: dict[str, dict[str, Any]] = {}
    all_person_labels: list[dict[str, str]] = []
    matched_segment_indices: list[int | str] = []
    unmatched_segment_indices: list[int | str] = []
    segments: list[dict[str, Any]] = []

    for segment_dir in segment_dirs:
        segment_index = parse_segment_index(segment_dir)
        display_index: int | str = segment_index if segment_index is not None else segment_dir.name

        video_files = sorted(segment_dir.glob("*.mp4"))
        questionnaire_json = segment_dir / "questionnaire.json"
        questionnaire_png = segment_dir / "questionnaire_list.png"
        usage_json = segment_dir / "usage.json"
        heart_rate_json = segment_dir / "heartRate.json"
        heart_rate_png = segment_dir / "heartRate.png"

        questionnaire_raw = read_json(questionnaire_json) if questionnaire_json.exists() else None
        if questionnaire_raw is not None:
            payload_hash = stable_payload_hash(questionnaire_raw)
            payload_group = questionnaire_payloads.setdefault(
                payload_hash,
                {
                    "representative_path": rel_path(questionnaire_json, repo_root),
                    "segment_indices": [],
                    "paths": [],
                    "raw": questionnaire_raw,
                },
            )
            payload_group["segment_indices"].append(display_index)
            payload_group["paths"].append(rel_path(questionnaire_json, repo_root))

        usage_raw = read_json(usage_json) if usage_json.exists() else None
        heart_rate_raw = read_json(heart_rate_json) if heart_rate_json.exists() else None
        segment_labels = labels_for_segment(segment_dir.name, indexed_labels)
        all_person_labels.extend(segment_labels)
        if segment_labels:
            matched_segment_indices.append(display_index)
        else:
            unmatched_segment_indices.append(display_index)

        files = {
            "video_paths": [rel_path(path, repo_root) for path in video_files],
            "questionnaire_json_path": rel_path(questionnaire_json, repo_root)
            if questionnaire_json.exists()
            else None,
            "questionnaire_image_path": rel_path(questionnaire_png, repo_root)
            if questionnaire_png.exists()
            else None,
            "usage_json_path": rel_path(usage_json, repo_root) if usage_json.exists() else None,
            "heart_rate_json_path": rel_path(heart_rate_json, repo_root)
            if heart_rate_json.exists()
            else None,
            "heart_rate_image_path": rel_path(heart_rate_png, repo_root)
            if heart_rate_png.exists()
            else None,
        }
        file_sizes = {
            path.name: path.stat().st_size for path in sorted(segment_dir.iterdir()) if path.is_file()
        }
        video_metadata = [
            get_video_metadata(path) if include_video_metadata else {
                "path": rel_path(path, repo_root),
                "size_bytes": path.stat().st_size,
                "metadata_status": "skipped",
            }
            for path in video_files
        ]
        for item in video_metadata:
            if "path" in item:
                item["path"] = rel_path(Path(item["path"]), repo_root)

        segments.append(
            {
                "segment_dir_name": segment_dir.name,
                "segment_dir_path": rel_path(segment_dir, repo_root),
                "segment_index": segment_index,
                "files": files,
                "file_sizes_bytes": file_sizes,
                "video_metadata": video_metadata,
                "usage": {
                    "path": files["usage_json_path"],
                    "record_count": len(usage_raw) if isinstance(usage_raw, list) else None,
                    "raw": usage_raw,
                },
                "heart_rate": {
                    "path": files["heart_rate_json_path"],
                    "point_count": len(heart_rate_raw) if isinstance(heart_rate_raw, list) else None,
                    "raw": heart_rate_raw,
                },
                "label_match_status": "matched" if segment_labels else "unmatched",
                "label_row_count": len(segment_labels),
                "label_rows": segment_labels,
            }
        )

    questionnaire_values = list(questionnaire_payloads.values())
    representative_questionnaire = questionnaire_values[0] if questionnaire_values else None
    questionnaire_scope = (
        "person_level_repeated_in_segment_dirs"
        if len(questionnaire_payloads) == 1 and len(segments) > 1
        else "person_level_with_payload_mismatch"
        if len(questionnaire_payloads) > 1
        else "missing"
    )

    return {
        "schema_version": "raw_person_level_v0",
        "person_dir_name": person_dir.name,
        "person_dir_path": rel_path(person_dir, repo_root),
        "source_type": "raw_internal",
        "segment_count": len(segments),
        "matched_label_segment_count": len(matched_segment_indices),
        "unmatched_label_segment_count": len(unmatched_segment_indices),
        "matched_segment_indices": matched_segment_indices,
        "unmatched_segment_indices": unmatched_segment_indices,
        "questionnaire": {
            "scope": questionnaire_scope,
            "unique_payload_count": len(questionnaire_payloads),
            "is_identical_across_segments": len(questionnaire_payloads) <= 1,
            "representative": representative_questionnaire,
            "payload_groups": questionnaire_values,
        },
        "label_summary": summarize_labels(all_person_labels),
        "segments": segments,
    }


def iter_person_dirs(data_root: Path) -> list[Path]:
    return sorted(
        [path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("person_")],
        key=lambda path: path.name,
    )


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate raw PriVTE person directories into one-person-per-line JSONL."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--label-xlsx", type=Path, default=Path("data/label.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("data/raw_person_level.jsonl"))
    parser.add_argument(
        "--skip-video-metadata",
        action="store_true",
        help="Do not open videos to collect width/height/fps/duration metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    data_root = args.data_root
    if not data_root.exists():
        print(f"Data root not found: {data_root}", file=sys.stderr)
        return 1
    if not args.label_xlsx.exists():
        print(f"Label file not found: {args.label_xlsx}", file=sys.stderr)
        return 1

    label_rows = read_xlsx_first_sheet(args.label_xlsx)
    indexed_labels = build_label_index(label_rows)
    person_dirs = iter_person_dirs(data_root)
    records = [
        build_person_record(
            person_dir=person_dir,
            repo_root=repo_root,
            indexed_labels=indexed_labels,
            include_video_metadata=not args.skip_video_metadata,
        )
        for person_dir in person_dirs
    ]
    write_jsonl(records, args.output)

    total_segments = sum(record["segment_count"] for record in records)
    matched_segments = sum(record["matched_label_segment_count"] for record in records)
    print(f"wrote_records={len(records)}")
    print(f"output={args.output}")
    print(f"label_rows={len(label_rows)}")
    print(f"segments={total_segments}")
    print(f"segments_with_labels={matched_segments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
