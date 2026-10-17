from pathlib import Path
import shutil
import zipfile
from huggingface_hub import snapshot_download

REPO_ID = "LibreYOLO/animals-ij5d2"
TARGET = Path("dataset/animals-ij5d2")
ZIP_PATH = Path("dataset/animals-ij5d2.zip")

def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {REPO_ID} ...")
    downloaded = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(TARGET),
        local_dir_use_symlinks=False,
    )
    print("Dataset folder:", downloaded)
    print("Creating ZIP archive ...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in TARGET.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(TARGET.parent))
    print("Dataset ZIP created:", ZIP_PATH.resolve())
    print("Check dataset/animals-ij5d2/data.yaml before training.")

if __name__ == "__main__":
    main()
