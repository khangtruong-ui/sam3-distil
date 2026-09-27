#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/mose"

# Accept repo ids or full URLs via args or env
MOSE1_REPO_RAW="${1:-${MOSE1_REPO:-FudanCVL/MOSE}}"
MOSE2_REPO_RAW="${2:-${MOSE2_REPO:-FudanCVL/MOSEv2}}"

normalize_repo_id() {
  local in="$1"
  local out="$in"
  if [[ "$out" == https://huggingface.co/datasets/* ]]; then
    out="${out#https://huggingface.co/datasets/}"
    out="${out%%/tree/*}"
  fi
  echo "$out"
}

MOSE1_REPO="$(normalize_repo_id "$MOSE1_REPO_RAW")"
MOSE2_REPO="$(normalize_repo_id "$MOSE2_REPO_RAW")"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Please install it: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/mose1" "$OUTPUT_ROOT/mose2"

download_hf_repo() {
  local repo_id="$1"
  local dest_dir="$2"
  echo "Downloading Hugging Face dataset repo: $repo_id -> $dest_dir"
  mkdir -p "$dest_dir"
  huggingface-cli download "$repo_id" \
    --repo-type dataset \
    --include "*" \
    --local-dir "$dest_dir" \
    --local-dir-use-symlinks False
}

# Join split archives like train.tar.gz.aa, train.tar.gz.ab -> train.tar.gz
join_multipart_archives() {
  local root="$1"
  echo "Joining multipart archives under $root if present"

  declare -A seen=()
  # Find parts ending with two-letter suffixes (e.g., .aa, .ab)
  while IFS= read -r -d '' f; do
    if [[ "$f" =~ \.(tar\.gz|zip)\.[a-z][a-z]$ ]]; then
      local base
      base="${f%.[a-z][a-z]}"
      seen["$base"]=1
    fi
  done < <(find "$root" -type f \( -name "*.tar.gz.*" -o -name "*.zip.*" \) -print0 2>/dev/null || true)

  for base in "${!seen[@]}"; do
    echo "Reassembling multipart: $base"
    # Collect parts sorted lexicographically (.aa, .ab, ...)
    mapfile -t parts < <(ls -1 "${base}".?? 2>/dev/null | sort)
    if (( ${#parts[@]} > 0 )); then
      cat "${parts[@]}" > "$base"
      rm -f "${parts[@]}"
    fi
  done
}

unzip_and_cleanup() {
  local dest_dir="$1"
  echo "Recursively extracting archives under $dest_dir and removing them afterward"

  extract_once() {
    local root="$1"
    local changed=0

    # .zip
    while IFS= read -r -d '' f; do
      echo "Unzipping: $f"
      unzip -o "$f" -d "$(dirname "$f")" >/dev/null
      rm -f "$f"
      changed=1
    done < <(find "$root" -type f -name "*.zip" -print0)

    # .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz
    while IFS= read -r -d '' f; do
      echo "Untarring: $f"
      local dir
      dir="$(dirname "$f")"
      case "$f" in
        *.tar) tar -xf "$f" -C "$dir" ;;
        *.tar.gz|*.tgz) tar -xzf "$f" -C "$dir" ;;
        *.tar.bz2) tar -xjf "$f" -C "$dir" ;;
        *.tar.xz) tar -xJf "$f" -C "$dir" ;;
        *) ;;
      esac
      rm -f "$f"
      changed=1
    done < <(find "$root" -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.bz2" -o -name "*.tar.xz" \) -print0)

    return $changed
  }

  # Keep extracting until no archives remain
  while true; do
    if extract_once "$dest_dir"; then
      echo "Another extraction pass needed..."
    else
      echo "No more nested archives in $dest_dir"
      break
    fi
  done
}

# MOSE 1 (train/val)
download_hf_repo "$MOSE1_REPO" "$OUTPUT_ROOT/mose1"
unzip_and_cleanup "$OUTPUT_ROOT/mose1"

# MOSE 2 (train/val)
download_hf_repo "$MOSE2_REPO" "$OUTPUT_ROOT/mose2"
# Some MOSEv2 releases use multipart archives (e.g., train.tar.gz.aa, .ab, .ac)
join_multipart_archives "$OUTPUT_ROOT/mose2"
unzip_and_cleanup "$OUTPUT_ROOT/mose2"
