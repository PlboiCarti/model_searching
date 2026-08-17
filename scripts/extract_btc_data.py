"""Giải nén dữ liệu BTC từ thư mục ZIP/ vào data/.

Tự động xử lý mapping tên thư mục gốc trong zip sang thư mục đích mà code đọc:
  - clip-features-32/ → data/clip-features/
  - keyframes/        → data/keyframes/
  - map-keyframes/    → data/map-keyframes/
  - media-info/       → data/media-info/
"""
import sys
import shutil
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Mapping: prefix bên trong zip → thư mục đích trong data/
PREFIX_MAP = {
    "keyframes":        "keyframes",
    "map-keyframes":    "map-keyframes",
    "mapkeyframes":     "map-keyframes",
    "media-info":       "media-info",
    "clip-features":    "clip-features",
    "clip-features-32": "clip-features",
    "clipfeatures":     "clip-features",
    "videos":           "videos",
    "video":            "videos",
}

sys.path.insert(0, str(ROOT_DIR))

from aic_model_searching.config import DATA_ROOT, ZIP_DIR  # noqa: E402


def _detect_prefix(zip_path: Path) -> tuple[str, str, bool]:
    """Detect root prefix inside zip and map to target directory.
    Returns: (root_prefix, target_dir_name, should_strip_prefix)
    """
    with zipfile.ZipFile(zip_path) as zf:
        # Kiểm tra xem có cấu trúc chuẩn kiểu `keyframes/L21_V001/...` không
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) >= 2 and parts[0]:
                root_prefix = parts[0]
                normalized_root = root_prefix.lower().replace("_", "-")
                for known_prefix, target in PREFIX_MAP.items():
                    # Nếu thư mục gốc thực sự tên là `keyframes` v.v. -> Strip nó đi
                    normalized_known = known_prefix.lower().replace("_", "-")
                    if normalized_root == normalized_known or normalized_root.startswith(normalized_known + "-"):
                        return root_prefix, target, True
        
        # Nếu không có thư mục gốc chuẩn, đoán qua tên file zip
        stem = zip_path.stem.lower()
        for known_prefix, target in PREFIX_MAP.items():
            if known_prefix in stem:
                return "", target, False  # Không strip gì cả, giữ nguyên cấu trúc bên trong

    return "", "", False

def extract_zip(zip_path: Path, dry_run: bool = False) -> int:
    """Extract one zip file to the correct data/ subdirectory."""
    root_prefix, target_dir_name, should_strip = _detect_prefix(zip_path)
    if not target_dir_name:
        print(f"  SKIP: Could not detect target dir for {zip_path.name}")
        return 0

    target_dir = DATA_ROOT / target_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {zip_path.name}")
    print(f"    Target: data/{target_dir_name}/" + (f" (Stripping root '{root_prefix}/')" if should_strip else " (Keeping internal structure)"))

    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Skip directory entries
            if member.endswith("/"):
                continue

            # Strip root prefix nếu cần (ví dụ: "keyframes/L21_V001/001.jpg" → "L21_V001/001.jpg")
            if should_strip and root_prefix and member.startswith(root_prefix + "/"):
                relative = member[len(root_prefix) + 1:]
            else:
                relative = member

            if not relative:
                continue

            dest = target_dir / relative
            if dest.exists():
                continue  # Skip already extracted

            if dry_run:
                print(f"    [dry-run] {relative}")
                extracted += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted += 1

    print(f"    Extracted {extracted} files" + (" (dry-run)" if dry_run else ""))
    return extracted


def main():
    dry_run = "--dry-run" in sys.argv

    if not ZIP_DIR.exists():
        print(f"ERROR: ZIP directory not found: {ZIP_DIR}")
        return

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in {ZIP_DIR}")
        return

    print(f"Found {len(zip_files)} zip files in {ZIP_DIR}")
    if dry_run:
        print("(DRY RUN - no files will be extracted)\n")
    print()

    total = 0
    for zp in zip_files:
        total += extract_zip(zp, dry_run=dry_run)

    print(f"\nTotal extracted: {total} files")
    if not dry_run:
        print("\nDone! Now run:")
        print("  python scripts/import_btc_data.py")


if __name__ == "__main__":
    main()
