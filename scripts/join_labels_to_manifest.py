#!/usr/bin/env python3
"""Join label.xlsx rows into PriVTE internal manifests.

This script reads:

    internal_person_manifest.<subset>.<version>.jsonl
    internal_clip_manifest.<subset>.<version>.jsonl
    data/label.xlsx

and writes labeled person/clip manifests plus a compact join report.

The join is internal-only. It records label summaries and row references, but it
does not inline raw path columns from the label sheet.
"""

from __future__ import annotations

import argparse
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

SEGMENT_NAME_RE = re.compile(
    r"(?P<segment>[^/\\\s]+_VID_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d+)"
)

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

DIMENSION_CONFIDENCE_COLUMNS = {
    "video": "视频置信度",
    "heart_rate": "心率置信度",
    "questionnaire": "问卷置信度",
    "app_usage": "应用置信度",
}


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
    """Read the first worksheet using only the standard library."""
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
        rows = [dict(zip(headers, row)) for row in raw_rows[1:]]
        for index, row in enumerate(rows, start=2):
            row["_label_row_index"] = str(index)
            row["_risk_level"] = risk_label_from_row(row)
        return rows


def risk_label_from_row(row: dict[str, str]) -> str:
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def extract_segment_names(row: dict[str, str]) -> set[str]:
    names: set[str] = set()
    searchable = " ".join(row.get(column, "") or "" for column in PATH_LIKE_LABEL_COLUMNS)
    for match in SEGMENT_NAME_RE.finditer(searchable):
        names.add(match.group("segment"))
    return names


def build_label_indexes(
    label_rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    by_segment: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_person_raw_id: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in label_rows:
        for segment_name in extract_segment_names(row):
            by_segment[segment_name].append(row)
        person_raw_id = (row.get("参与者编号") or "").strip()
        if person_raw_id:
            by_person_raw_id[person_raw_id].append(row)

    return by_segment, by_person_raw_id


def majority_label(risk_counts: Counter[str]) -> tuple[str | None, bool]:
    if not risk_counts:
        return None, False
    ordered = risk_counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "insufficient_evidence", True
    return ordered[0][0], False


def value_counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(Counter((row.get(column) or "<blank>") for row in rows))


def summarize_label_rows(
    rows: list[dict[str, str]],
    *,
    label_source: str,
    match_method: str,
    aggregation_method: str,
) -> dict[str, Any]:
    risk_counts = Counter(row["_risk_level"] for row in rows)
    risk_level, has_tie = majority_label(risk_counts)
    raw_column_counts = {
        column: value_counts(rows, column)
        for column in SUMMARY_LABEL_COLUMNS
        if any(row.get(column) for row in rows)
    }
    return {
        "available": bool(rows),
        "risk_level": risk_level,
        "label_source": label_source,
        "match_method": match_method if rows else None,
        "aggregation_method": aggregation_method if rows else None,
        "has_tie": has_tie,
        "matched_label_row_count": len(rows),
        "matched_label_row_indices": [int(row["_label_row_index"]) for row in rows],
        "risk_label_counts": dict(risk_counts),
        "raw_column_counts": raw_column_counts,
        "dimension_confidence_counts": {
            dimension: value_counts(rows, column)
            for dimension, column in DIMENSION_CONFIDENCE_COLUMNS.items()
        },
        "dimension_labels": {
            "video": None,
            "heart_rate": None,
            "questionnaire": None,
            "app_usage": None,
        },
    }


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        row_index = row["_label_row_index"]
        if row_index in seen:
            continue
        seen.add(row_index)
        unique.append(row)
    return unique


def join_clip_labels(
    clip_records: list[dict[str, Any]],
    by_segment: dict[str, list[dict[str, str]]],
    label_source: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    rows_by_clip_uid: dict[str, list[dict[str, str]]] = {}
    labeled_records: list[dict[str, Any]] = []
    for record in clip_records:
        rows = unique_rows(by_segment.get(record["segment_dir_name"], []))
        rows_by_clip_uid[record["clip_uid"]] = rows
        labeled = dict(record)
        labeled["label"] = summarize_label_rows(
            rows,
            label_source=label_source,
            match_method="segment_dir_name_in_label_paths",
            aggregation_method="majority_vote_over_matched_label_rows",
        )
        labeled_records.append(labeled)
    return labeled_records, rows_by_clip_uid


def clip_rows_for_person(
    person_record: dict[str, Any],
    rows_by_clip_uid: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for session in person_record.get("sessions", []):
        for clip in session.get("clips", []):
            rows.extend(rows_by_clip_uid.get(clip["clip_uid"], []))
    return unique_rows(rows)


def add_nested_clip_labels(
    person_record: dict[str, Any],
    labeled_clip_by_uid: dict[str, dict[str, Any]],
) -> None:
    for session in person_record.get("sessions", []):
        for clip in session.get("clips", []):
            labeled_clip = labeled_clip_by_uid.get(clip["clip_uid"])
            if labeled_clip is not None:
                clip["label"] = labeled_clip["label"]


def join_person_labels(
    person_records: list[dict[str, Any]],
    labeled_clip_records: list[dict[str, Any]],
    rows_by_clip_uid: dict[str, list[dict[str, str]]],
    by_person_raw_id: dict[str, list[dict[str, str]]],
    label_source: str,
) -> list[dict[str, Any]]:
    labeled_clip_by_uid = {record["clip_uid"]: record for record in labeled_clip_records}
    labeled_person_records: list[dict[str, Any]] = []
    for record in person_records:
        labeled = dict(record)
        rows = clip_rows_for_person(labeled, rows_by_clip_uid)
        match_method = "aggregate_from_exact_clip_matches"
        if not rows:
            rows = unique_rows(by_person_raw_id.get(labeled.get("person_raw_id", ""), []))
            match_method = "person_raw_id_fallback"
        labeled["label"] = summarize_label_rows(
            rows,
            label_source=label_source,
            match_method=match_method,
            aggregation_method="majority_vote_over_matched_label_rows",
        )
        labeled["label"]["clip_label_coverage"] = {
            "num_clips": labeled.get("modality_completeness", {}).get("num_clips"),
            "num_labeled_clips": sum(
                1
                for session in labeled.get("sessions", [])
                for clip in session.get("clips", [])
                if rows_by_clip_uid.get(clip["clip_uid"])
            ),
        }
        add_nested_clip_labels(labeled, labeled_clip_by_uid)
        labeled_person_records.append(labeled)
    return labeled_person_records


def derive_output_path(input_path: Path, suffix: str) -> Path:
    if input_path.suffix == ".jsonl":
        return input_path.with_name(f"{input_path.stem}.{suffix}.jsonl")
    return input_path.with_name(f"{input_path.name}.{suffix}")


def subset_from_manifest_name(path: Path) -> str:
    name = path.name
    name = re.sub(r"^internal_(person|clip)_manifest\.", "", name)
    name = re.sub(r"\.jsonl$", "", name)
    return name


def build_report(
    *,
    label_rows: list[dict[str, str]],
    person_records: list[dict[str, Any]],
    clip_records: list[dict[str, Any]],
    labeled_person_records: list[dict[str, Any]],
    labeled_clip_records: list[dict[str, Any]],
    output_person_path: Path,
    output_clip_path: Path,
) -> dict[str, Any]:
    clips_with_labels = [record for record in labeled_clip_records if record["label"]["available"]]
    clips_without_labels = [
        record for record in labeled_clip_records if not record["label"]["available"]
    ]
    persons_with_labels = [
        record for record in labeled_person_records if record["label"]["available"]
    ]
    return {
        "schema_version": "label_join_report.v0",
        "label_rows": len(label_rows),
        "input_person_records": len(person_records),
        "input_clip_records": len(clip_records),
        "output_person_manifest": output_person_path.as_posix(),
        "output_clip_manifest": output_clip_path.as_posix(),
        "matched_persons": len(persons_with_labels),
        "unmatched_persons": len(labeled_person_records) - len(persons_with_labels),
        "matched_clips": len(clips_with_labels),
        "unmatched_clips": len(clips_without_labels),
        "matched_clip_counts_by_person_raw_id": dict(
            Counter(record["person_raw_id"] for record in clips_with_labels)
        ),
        "unmatched_clip_counts_by_person_raw_id": dict(
            Counter(record["person_raw_id"] for record in clips_without_labels)
        ),
        "unmatched_clip_segment_dir_names": [
            record["segment_dir_name"] for record in clips_without_labels
        ],
        "person_risk_label_counts": dict(
            Counter(record["label"]["risk_level"] for record in persons_with_labels)
        ),
        "clip_risk_label_counts": dict(
            Counter(record["label"]["risk_level"] for record in clips_with_labels)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join label.xlsx into PriVTE person and clip manifests."
    )
    parser.add_argument("--person-manifest", type=Path, required=True)
    parser.add_argument("--clip-manifest", type=Path, required=True)
    parser.add_argument("--label-xlsx", type=Path, default=Path("data/label.xlsx"))
    parser.add_argument("--output-person", type=Path, default=None)
    parser.add_argument("--output-clip", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--suffix", default="labeled")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.person_manifest.exists():
        print(f"Person manifest not found: {args.person_manifest}", file=sys.stderr)
        return 1
    if not args.clip_manifest.exists():
        print(f"Clip manifest not found: {args.clip_manifest}", file=sys.stderr)
        return 1
    if not args.label_xlsx.exists():
        print(f"Label file not found: {args.label_xlsx}", file=sys.stderr)
        return 1

    person_records = read_jsonl(args.person_manifest)
    clip_records = read_jsonl(args.clip_manifest)
    label_rows = read_xlsx_first_sheet(args.label_xlsx)
    by_segment, by_person_raw_id = build_label_indexes(label_rows)

    label_source = args.label_xlsx.as_posix()
    labeled_clip_records, rows_by_clip_uid = join_clip_labels(
        clip_records,
        by_segment,
        label_source,
    )
    labeled_person_records = join_person_labels(
        person_records,
        labeled_clip_records,
        rows_by_clip_uid,
        by_person_raw_id,
        label_source,
    )

    output_person = args.output_person or derive_output_path(args.person_manifest, args.suffix)
    output_clip = args.output_clip or derive_output_path(args.clip_manifest, args.suffix)
    subset = subset_from_manifest_name(args.person_manifest)
    output_report = args.output_report or args.person_manifest.with_name(
        f"label_join_report.{subset}.{args.suffix}.json"
    )

    write_jsonl(output_person, labeled_person_records)
    write_jsonl(output_clip, labeled_clip_records)
    report = build_report(
        label_rows=label_rows,
        person_records=person_records,
        clip_records=clip_records,
        labeled_person_records=labeled_person_records,
        labeled_clip_records=labeled_clip_records,
        output_person_path=output_person,
        output_clip_path=output_clip,
    )
    write_json(output_report, report)

    print(f"output_person={output_person}")
    print(f"output_clip={output_clip}")
    print(f"output_report={output_report}")
    print(f"label_rows={len(label_rows)}")
    print(f"matched_persons={report['matched_persons']}")
    print(f"matched_clips={report['matched_clips']}")
    print(f"unmatched_clips={report['unmatched_clips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
