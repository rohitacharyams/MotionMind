"""
Setup script — installs all dependencies and MMPose model weights.

Usage:
    python scripts/setup_env.py
    python scripts/setup_env.py --cpu-only
"""

import argparse
import subprocess
import sys
import os


def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  WARNING: Command exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Setup environment")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU/CUDA packages")
    args = parser.parse_args()

    python = sys.executable

    # 1. Core dependencies
    run(f"{python} -m pip install --upgrade pip", "Upgrading pip")
    run(f"{python} -m pip install numpy scipy opencv-python Pillow pyyaml tqdm h5py imageio imageio-ffmpeg matplotlib scikit-learn",
        "Installing core dependencies")

    # 2. PyTorch (GPU or CPU)
    if args.cpu_only:
        run(f"{python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "Installing PyTorch (CPU)")
    else:
        run(f"{python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
            "Installing PyTorch (CUDA 12.1)")

    # 3. FAISS
    run(f"{python} -m pip install faiss-cpu", "Installing FAISS")

    # 4. MMPose stack via OpenMIM
    run(f"{python} -m pip install openmim", "Installing OpenMIM")
    run(f"{python} -m mim install mmengine", "Installing mmengine")
    run(f"{python} -m mim install 'mmcv>=2.0.0'", "Installing mmcv")
    run(f"{python} -m mim install 'mmdet>=3.0.0'", "Installing mmdet")
    run(f"{python} -m mim install 'mmpose>=1.0.0'", "Installing mmpose")

    # 5. Download model weights
    print("\n" + "="*60)
    print("  Downloading RTMPose whole-body model weights...")
    print("  (This may take a few minutes)")
    print("="*60)

    run(f"{python} -m mim download mmdet --config rtmdet_m_640-8xb32_coco-person --dest checkpoints",
        "Downloading RTMDet person detector")
    run(f"{python} -m mim download mmpose --config rtmpose-l_8xb32-270e_coco-wholebody-384x288 --dest checkpoints",
        "Downloading RTMPose whole-body model")

    # 6. Create data directories
    for d in ["data/input_videos", "data/motion_db", "data/output_videos", "checkpoints"]:
        os.makedirs(d, exist_ok=True)

    # 7. Verify GPU
    print("\n" + "="*60)
    print("  Verifying installation...")
    print("="*60)

    verify_code = """
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
try:
    import mmpose
    print(f"  MMPose: {mmpose.__version__}")
except: print("  MMPose: not installed")
try:
    import mmdet
    print(f"  MMDet: {mmdet.__version__}")
except: print("  MMDet: not installed")
try:
    import faiss
    print(f"  FAISS: OK")
except: print("  FAISS: not installed")
print("\\n  Setup complete!")
"""

    subprocess.run([python, "-c", verify_code])


if __name__ == "__main__":
    main()
