#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SESSION_MARKER = "_VID_"
KEEP_TOP_LEVEL_DIRS = {"manifests"}
KNOWN_GROUP_NAMES = ["6.1data", "6.5数据测试", "数据测试2"]


@dataclass
class Stats:
    group_dirs: int = 0
    samples_seen: int = 0
    samples_moved: int = 0
    samples_merged: int = 0
    files_moved_into_existing: int = 0
    duplicate_files_left_in_backup: int = 0
    conflict_files_left_in_backup: int = 0
    manifest_files_updated: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten grouped PriVTE data directories so sample/session folders "
            "live directly under the data root."
        )
    )
    parser.add_argument(
        "--data-root",
        default="/mnt/e/new_data",
        help="Data root to flatten.",
    )
    parser.add_argument(
        "--backup-root",
        default=None,
        help=(
            "Backup directory. Defaults to a timestamped sibling of data-root, "
            "for example /mnt/e/new_data_flatten_backup_YYYYmmdd_HHMMSS."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files and update manifests. Without this flag, only prints a plan.",
    )
    return parser.parse_args()


def is_session_dir(path: Path) -> bool:
    return path.is_dir() and SESSION_MARKER in path.name


def find_group_dirs(data_root: Path) -> list[Path]:
    groups: list[Path] = []
    for child in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name in KEEP_TOP_LEVEL_DIRS:
            continue
        if is_session_dir(child):
            continue
        has_session_children = any(is_session_dir(grandchild) for grandchild in child.iterdir())
        if has_session_children:
            groups.append(child)
    return groups


def same_file_size(left: Path, right: Path) -> bool:
    try:
        return left.stat().st_size == right.stat().st_size
    except OSError:
        return False


def prune_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def merge_into_existing(src: Path, dst: Path, stats: Stats, apply: bool) -> None:
    for path in sorted(src.rglob("*"), key=lambda p: (len(p.parts), str(p))):
        rel = path.relative_to(src)
        target = dst / rel

        if path.is_dir():
            if apply:
                target.mkdir(parents=True, exist_ok=True)
            continue

        if not path.is_file():
            continue

        if not target.exists():
            stats.files_moved_into_existing += 1
            print(f"  add missing file: {path} -> {target}")
            if apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
            continue

        if target.is_file() and same_file_size(path, target):
            stats.duplicate_files_left_in_backup += 1
            continue

        stats.conflict_files_left_in_backup += 1
        print(f"  conflict left in backup: {path} ; existing target: {target}")

    if apply:
        prune_empty_dirs(src)


def backup_group_dirs(data_root: Path, backup_root: Path, groups: list[Path], apply: bool) -> list[Path]:
    backed_up_groups: list[Path] = []
    for group in groups:
        backup_group = backup_root / group.name
        backed_up_groups.append(backup_group)
        print(f"backup group: {group} -> {backup_group}")
        if apply:
            if backup_group.exists():
                raise FileExistsError(f"Backup group already exists: {backup_group}")
            backup_group.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(group), str(backup_group))
    if not apply:
        return groups
    return backed_up_groups


def flatten_groups(data_root: Path, backup_groups: list[Path], stats: Stats, apply: bool) -> None:
    for group in backup_groups:
        for sample in sorted((p for p in group.iterdir() if is_session_dir(p)), key=lambda p: p.name):
            stats.samples_seen += 1
            target = data_root / sample.name
            if not target.exists():
                stats.samples_moved += 1
                print(f"move sample: {sample} -> {target}")
                if apply:
                    shutil.move(str(sample), str(target))
                continue

            stats.samples_merged += 1
            print(f"merge duplicate sample directory: {sample} -> {target}")
            merge_into_existing(sample, target, stats, apply)


def update_manifest_paths(data_root: Path, backup_root: Path, group_names: list[str], stats: Stats, apply: bool) -> None:
    manifests_dir = data_root / "manifests"
    if not manifests_dir.exists():
        print("manifest update: skipped; no manifests directory")
        return

    manifest_files = [
        p
        for p in sorted(manifests_dir.rglob("*"), key=lambda p: str(p))
        if p.is_file() and p.suffix in {".json", ".jsonl"}
    ]
    if not manifest_files:
        print("manifest update: skipped; no JSON/JSONL manifest files")
        return

    backup_manifest_dir = backup_root / "manifests_before_flatten"
    for manifest in manifest_files:
        text = manifest.read_text(encoding="utf-8")
        updated = text
        for group_name in group_names:
            updated = updated.replace(f"data/{group_name}/", "data/")
            updated = updated.replace(f"/mnt/e/data/{group_name}/", "/mnt/e/data/")
            updated = updated.replace(
                f"/mnt/e/new_data/{group_name}/",
                "/mnt/e/new_data/",
            )
            updated = updated.replace(f'"data/{group_name}"', '"data"')
            updated = updated.replace(f'"/mnt/e/data/{group_name}"', '"/mnt/e/data"')
            updated = updated.replace(
                f'"/mnt/e/new_data/{group_name}"',
                '"/mnt/e/new_data"',
            )

        if updated == text:
            continue

        stats.manifest_files_updated += 1
        print(f"update manifest paths: {manifest}")
        if apply:
            backup_manifest_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_manifest_dir / manifest.relative_to(manifests_dir)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest, backup_path)
            manifest.write_text(updated, encoding="utf-8")


def infer_stale_group_names_from_manifests(data_root: Path) -> list[str]:
    manifests_dir = data_root / "manifests"
    if not manifests_dir.exists():
        return []

    found: set[str] = set()
    for manifest in manifests_dir.rglob("*"):
        if not manifest.is_file() or manifest.suffix not in {".json", ".jsonl"}:
            continue
        text = manifest.read_text(encoding="utf-8")
        for group_name in KNOWN_GROUP_NAMES:
            if (
                f"data/{group_name}/" in text
                or f"/mnt/e/data/{group_name}/" in text
                or f"/mnt/e/new_data/{group_name}/" in text
                or f'"data/{group_name}"' in text
                or f'"/mnt/e/data/{group_name}"' in text
                or f'"/mnt/e/new_data/{group_name}"' in text
            ):
                found.add(group_name)
    return [name for name in KNOWN_GROUP_NAMES if name in found]


def verify_no_group_dirs(data_root: Path) -> None:
    remaining = find_group_dirs(data_root)
    if remaining:
        names = ", ".join(str(p) for p in remaining)
        raise RuntimeError(f"Grouped directories still remain under data root: {names}")


def verify_manifest_paths(data_root: Path) -> None:
    manifest = data_root / "manifests" / "internal_clip_manifest.all_current.v0.jsonl"
    if not manifest.exists():
        print("manifest path verification: skipped; all_current clip manifest not found")
        return

    with manifest.open("r", encoding="utf-8") as f:
        rec = json.loads(next(f))

    paths: list[str] = []
    for item in rec.get("video", {}).get("files", []):
        if isinstance(item, dict) and item.get("path"):
            paths.append(item["path"])
    for key in ("usage", "heart_rate", "questionnaire_ref"):
        value = rec.get(key, {})
        if isinstance(value, dict) and value.get("path"):
            paths.append(value["path"])

    missing: list[str] = []
    for raw in paths:
        path = Path(raw)
        ok = path.exists()
        print(f"verify manifest path: {path} {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(raw)

    if missing:
        raise RuntimeError(f"Manifest paths missing after flatten: {missing}")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {data_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = Path(args.backup_root).resolve() if args.backup_root else data_root.parent / f"data_flatten_backup_{timestamp}"

    groups = find_group_dirs(data_root)
    stats = Stats(group_dirs=len(groups))

    print(f"data_root={data_root}")
    print(f"backup_root={backup_root}")
    print(f"mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print("group_dirs=" + ", ".join(group.name for group in groups) if groups else "group_dirs=<none>")

    if not groups:
        stale_group_names = infer_stale_group_names_from_manifests(data_root)
        if stale_group_names:
            print("stale_manifest_group_prefixes=" + ", ".join(stale_group_names))
            update_manifest_paths(data_root, backup_root, stale_group_names, stats, args.apply)
            if args.apply:
                verify_manifest_paths(data_root)
            print("summary:")
            for key, value in stats.__dict__.items():
                print(f"  {key}: {value}")
            if not args.apply:
                print("Dry-run only. Re-run with --apply to execute.")
            else:
                print(f"Done. Manifest backups are preserved in: {backup_root}")
            return 0
        verify_manifest_paths(data_root)
        print("Nothing to flatten.")
        return 0

    if args.apply and backup_root.exists():
        raise FileExistsError(f"Backup root already exists: {backup_root}")

    backup_groups = backup_group_dirs(data_root, backup_root, groups, args.apply)
    flatten_groups(data_root, backup_groups, stats, args.apply)
    update_manifest_paths(data_root, backup_root, [group.name for group in groups], stats, args.apply)

    if args.apply:
        verify_no_group_dirs(data_root)
        verify_manifest_paths(data_root)

    print("summary:")
    for key, value in stats.__dict__.items():
        print(f"  {key}: {value}")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to execute.")
    else:
        print(f"Done. Original grouped data and unchanged duplicates are preserved in: {backup_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
