"""Скачать FineWeb-Edu через HF Hub — качественный web-text dataset."""
import sys
from pathlib import Path

DATA_DIR = Path("data/sources/fineweb_edu")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Target: {DATA_DIR.absolute()}")

try:
    from huggingface_hub import snapshot_download
    print("Downloading FineWeb-Edu sample-10BT (10B tokens)...")
    print("This may take a while - up to 50GB depending on sample")
    snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        cache_dir=str(DATA_DIR),
        allow_patterns=["sample/10BT/*"],
    )
    print("Download complete")
except Exception as e:
    print(f"HF download failed: {e}")
    print("Falling back to curl...")

    # Fallback: скачиваем sample jsonl
    import urllib.request
    files = [
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet",
    ]
    for url in files:
        target = DATA_DIR / Path(url).name
        print(f"Downloading {url} -> {target}")
        try:
            urllib.request.urlretrieve(url, target)
            print(f"  OK: {target.stat().st_size/1e6:.1f} MB")
        except Exception as e2:
            print(f"  Failed: {e2}")

print("Done")
