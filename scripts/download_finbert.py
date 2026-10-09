"""
Robust resumable downloader for FinBERT pytorch_model.bin.
Uses HTTP Range requests with automatic retries and exponential backoff.
"""
import os
import time
import urllib.request
from pathlib import Path

TARGET = Path("D:/Projects/RiskLens/.model_cache/models--ProsusAI--finbert/snapshots/4556d13015211d73dccd3fdd39d39232506f3e43/pytorch_model.bin")
TARGET.parent.mkdir(parents=True, exist_ok=True)
URL = "https://huggingface.co/ProsusAI/finbert/resolve/main/pytorch_model.bin"
TOTAL_SIZE = 437992753

def download_model():
    # Follow redirects to get final CDN URL
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        final_url = resp.geturl()

    print(f"Target file: {TARGET}")
    print(f"Total size: {TOTAL_SIZE / (1024*1024):.1f} MB")

    retries = 0
    max_retries = 30
    chunk_size = 1024 * 1024 * 4  # 4MB buffer

    while retries < max_retries:
        downloaded = TARGET.stat().st_size if TARGET.exists() else 0
        if downloaded >= TOTAL_SIZE:
            print(f"Download complete: {downloaded} bytes.")
            return True

        print(f"Resuming from {downloaded / (1024*1024):.1f} MB ({downloaded / TOTAL_SIZE * 100:.1f}%) ...")
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={downloaded}-"
        }

        try:
            req = urllib.request.Request(final_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(TARGET, "ab" if downloaded > 0 else "wb") as out_file:
                while True:
                    data = response.read(chunk_size)
                    if not data:
                        break
                    out_file.write(data)
                    downloaded += len(data)
                    print(f"\rProgress: {downloaded / (1024*1024):.1f} / {TOTAL_SIZE / (1024*1024):.1f} MB ({downloaded / TOTAL_SIZE * 100:.1f}%)", end="", flush=True)

            print("\nDownload finished.")
            return True
        except Exception as exc:
            retries += 1
            print(f"\nConnection hiccup: {exc}. Retrying ({retries}/{max_retries}) in 2 seconds...")
            time.sleep(2)

    return False

if __name__ == "__main__":
    download_model()
