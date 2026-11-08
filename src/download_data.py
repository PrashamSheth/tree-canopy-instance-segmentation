"""Automated Dataset Downloader for Solafune Tree Canopy Segmentation Challenge.
Downloads competition imagery and annotations with streaming chunks and progress display.
"""

import os
import zipfile
import requests
from typing import Dict

URLS: Dict[str, str] = {
    "train_annotations": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/train_annotations.json",
    "train_images": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/train_images.zip",
    "evaluation_images": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/evaluation_images.zip",
    "sample_answer": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/sample_answer.json",
}


def download_file(url: str, save_path: str, chunk_size: int = 65536):
    """Downloads a file in streaming chunks if not already present."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"[Info] File already exists: {save_path}")
        return

    print(f"[Download] Fetching {url} -> {save_path} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print(f"[Success] Downloaded {save_path}")


def extract_zip(zip_path: str, extract_to: str):
    """Extracts a zip file if destination does not exist."""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"[Info] Archive already extracted: {extract_to}")
        return

    print(f"[Extract] Unzipping {zip_path} -> {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"[Success] Extracted to {extract_to}")


def main(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    for name, url in URLS.items():
        ext = ".zip" if "images" in name else ".json"
        dest_path = os.path.join(data_dir, f"{name}{ext}")
        download_file(url, dest_path)
        if ext == ".zip":
            extract_zip(dest_path, os.path.join(data_dir, name))


if __name__ == "__main__":
    main()
