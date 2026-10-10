"""Package repository into clean lightweight zip excluding models, cache, git, venv."""
import os
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", ".model_cache", "__pycache__", "venv", ".venv", "env", ".pytest_cache",
    "Financial Headlines", "Sentimental Dataset", "Social Source", ".idea", ".vscode"
}
EXCLUDE_EXTS = {".pyc", ".pyo"}

def package():
    zip_path = Path("RiskLens-submission.zip")
    root_dir = Path(".")
    
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for f in files:
                fpath = Path(root) / f
                if fpath == zip_path or fpath.suffix in EXCLUDE_EXTS:
                    continue
                arcname = fpath.relative_to(root_dir)
                zf.write(fpath, arcname)
                file_count += 1
                
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Packaged {file_count} files into {zip_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    package()
