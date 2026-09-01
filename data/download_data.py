"""
Download the MovieLens Latest Small dataset.
Source: https://grouplens.org/datasets/movielens/
License: Free for research and educational use.

This script downloads and extracts the dataset automatically.
It is idempotent — skips download if data already exists.
"""

import os
import ssl
import urllib.request
import zipfile
import shutil

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-latest-small")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")

EXPECTED_FILES = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]


def is_dataset_present():
    """Check if all expected dataset files exist."""
    return all(
        os.path.isfile(os.path.join(EXTRACT_DIR, f))
        for f in EXPECTED_FILES
    )


def _download_file(url, dest):
    """Download a file, handling SSL certificate issues gracefully."""
    try:
        # Try standard download first
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("  [!] SSL certificate issue - using unverified context")
            print("      (Safe for this well-known public dataset)")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            with opener.open(url) as response, open(dest, "wb") as out:
                out.write(response.read())
        else:
            raise


def download_dataset():
    """Download and extract the MovieLens Latest Small dataset."""
    if is_dataset_present():
        print(f"[OK] Dataset already exists at: {EXTRACT_DIR}")
        return EXTRACT_DIR

    print(f"Downloading MovieLens dataset from:\n  {DATASET_URL}")
    try:
        _download_file(DATASET_URL, ZIP_PATH)
        print(f"[OK] Download complete ({os.path.getsize(ZIP_PATH) / 1e6:.1f} MB)")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

    print("Extracting...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"[OK] Extracted to: {EXTRACT_DIR}")
    except zipfile.BadZipFile:
        raise RuntimeError("Downloaded file is corrupted. Delete and retry.")
    finally:
        # Clean up zip file
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

    # Verify extraction
    if not is_dataset_present():
        raise RuntimeError(
            f"Extraction succeeded but expected files not found in {EXTRACT_DIR}. "
            f"Expected: {EXPECTED_FILES}"
        )

    print(f"[OK] Dataset ready! Files: {EXPECTED_FILES}")
    return EXTRACT_DIR


if __name__ == "__main__":
    download_dataset()
