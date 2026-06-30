#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

target="${PRIVTE_DATA_ROOT:-/mnt/e/new_data}"
src="data"
backup_base="${PRIVTE_DATA_BACKUP:-data.local_backup_before_mnt_link}"

log() {
  printf '[migrate-data] %s\n' "$*"
}

fail() {
  printf '[migrate-data][ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

resolve_path() {
  readlink -f "$1" 2>/dev/null || true
}

run_rsync() {
  local mode="$1"
  shift

  log "$mode: syncing $src/ -> $target/"
  rsync -rlD \
    --no-owner \
    --no-group \
    --no-perms \
    --no-times \
    --omit-dir-times \
    --inplace \
    --size-only \
    --info=progress2 \
    "$@" \
    "$src/" "$target/"
}

verify_rsync_clean() {
  local tmp
  tmp="$(mktemp)"

  rsync -rlD \
    --no-owner \
    --no-group \
    --no-perms \
    --no-times \
    --omit-dir-times \
    --inplace \
    --size-only \
    --dry-run \
    --itemize-changes \
    --out-format='%i %n' \
    "$src/" "$target/" >"$tmp"

  if [[ -s "$tmp" ]]; then
    log "Dry-run verification still sees pending changes. First lines:"
    sed -n '1,80p' "$tmp"
    rm -f "$tmp"
    fail "Sync is not clean yet. Rerun this script; it will resume safely."
  fi

  rm -f "$tmp"
  log "Dry-run verification is clean."
}

verify_manifest_paths() {
  local manifest="data/manifests/internal_clip_manifest.all_current.v0.jsonl"

  if [[ ! -f "$manifest" ]]; then
    log "Skip manifest path check: $manifest not found."
    return 0
  fi

  log "Checking sample paths from $manifest"
  python3 - <<'PY'
import json
from pathlib import Path
import sys

manifest = Path("data/manifests/internal_clip_manifest.all_current.v0.jsonl")
with manifest.open("r", encoding="utf-8") as f:
    rec = json.loads(next(f))

paths = []
for item in rec.get("video", {}).get("files", []):
    if isinstance(item, dict):
        paths.append(item.get("path"))
for key in ("usage", "heart_rate", "questionnaire_ref"):
    value = rec.get(key, {})
    if isinstance(value, dict):
        paths.append(value.get("path"))

missing = []
for raw in paths:
    if not raw:
        continue
    path = Path(raw)
    ok = path.exists()
    print(f"{path} {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(str(path))

if missing:
    print("\nMissing manifest paths:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    sys.exit(1)
PY
}

require_cmd rsync
require_cmd readlink
require_cmd python3

log "Repository: $repo_root"
log "Target data root: $target"

mkdir -p "$target"

target_real="$(resolve_path "$target")"
[[ -n "$target_real" ]] || fail "Cannot resolve target path: $target"

if [[ -L "$src" ]]; then
  src_real="$(resolve_path "$src")"
  log "$src is already a symlink -> $src_real"
  if [[ "$src_real" != "$target_real" ]]; then
    fail "$src points to $src_real, not $target_real. Please inspect it manually."
  fi
  verify_manifest_paths
  log "Done. data already points to $target_real"
  exit 0
fi

[[ -d "$src" ]] || fail "Local source directory not found: $src"

log "Local source size:"
du -sh "$src" || true
log "Target current size:"
du -sh "$target" || true

run_rsync "copy"

log "Dry-run verification before replacing local data with a symlink."
verify_rsync_clean

backup="$backup_base"
if [[ -e "$backup" || -L "$backup" ]]; then
  backup="${backup_base}.$(date +%Y%m%d_%H%M%S)"
fi

log "Renaming local $src -> $backup"
mv "$src" "$backup"

log "Creating symlink: $src -> $target"
ln -s "$target" "$src"

log "Symlink result:"
ls -ld "$src"
log "Resolved data path: $(resolve_path "$src")"

verify_manifest_paths

log "Migration complete."
log "Local original data is preserved at: $backup"
log "After later pipeline checks pass, you may delete that backup manually:"
log "  rm -rf $backup"
