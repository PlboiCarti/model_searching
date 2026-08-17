"""Trích xuất CLIP features cho các custom keyframes.

Đọc tất cả các file .jpg trong data/keyframes/, chạy qua mô hình CLIP,
và lưu kết quả thành các file .npy trong data/clip-features/.
Định dạng: data/clip-features/<video_id>/<frame_stem>.npy
"""
import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import BTC_CLIP_FEATURES_DIR, DEVICE, KEYFRAMES_DIR
from backend.embedding.clip_encoder import _load_clip, _normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KeyframeDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                frame_rgb = img.convert("RGB")
                img = self.preprocess(frame_rgb)
        except Exception:
            return torch.zeros((3, 224, 224)), str(path), False
        return img, str(path), True


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for local keyframes")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per inference batch")
    parser.add_argument("--limit", type=int, default=0, help="Limit pending keyframes for a smoke test (0=all)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 in Windows notebooks; increase only after a stable GPU run.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    if not KEYFRAMES_DIR.exists():
        logger.error("Keyframes directory not found: %s", KEYFRAMES_DIR)
        return
        
    image_paths = sorted(KEYFRAMES_DIR.rglob("*.jpg"))
    if not image_paths:
        logger.warning("No .jpg files found in %s", KEYFRAMES_DIR)
        return
        
    # Lọc những file chưa extract
    paths_to_process = []
    for img_path in image_paths:
        video_id = img_path.parent.name
        frame_stem = img_path.stem
        save_path = BTC_CLIP_FEATURES_DIR / video_id / f"{frame_stem}.npy"
        if not save_path.exists():
            paths_to_process.append(img_path)
            if args.limit > 0 and len(paths_to_process) >= args.limit:
                break
            
    if not paths_to_process:
        logger.info("Tất cả %d keyframes đã được extract (resume success).", len(image_paths))
        return

    logger.info("Found %d pending keyframes. Loading CLIP model on %s...", len(paths_to_process), DEVICE)
    model, preprocess = _load_clip()
    
    dataset = KeyframeDataset(paths_to_process, preprocess)
    use_cuda = str(DEVICE).startswith("cuda")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    
    success_count = 0
    with torch.no_grad():
        with tqdm(total=len(paths_to_process), desc="Extracting Features (Batched)") as pbar:
            for imgs, paths, valids in dataloader:
                # Xử lý những ảnh hợp lệ trong batch
                valid_mask = valids.numpy()
                if not valid_mask.any():
                    pbar.update(len(paths))
                    continue
                
                valid_imgs = imgs[valids].to(DEVICE)
                embs = model.encode_image(valid_imgs).float()
                embs = _normalize(embs.cpu().numpy())
                
                # Lưu file
                emb_idx = 0
                for i, path_str in enumerate(paths):
                    if valids[i]:
                        img_path = Path(path_str)
                        video_id = img_path.parent.name
                        frame_stem = img_path.stem
                        
                        output_dir = BTC_CLIP_FEATURES_DIR / video_id
                        output_dir.mkdir(parents=True, exist_ok=True)
                        save_path = output_dir / f"{frame_stem}.npy"
                        
                        np.save(save_path, embs[emb_idx])
                        emb_idx += 1
                        success_count += 1
                
                pbar.update(len(paths))
            
    logger.info("Successfully extracted features for %d/%d keyframes.", success_count, len(paths_to_process))
    logger.info("Features saved to: %s", BTC_CLIP_FEATURES_DIR)

if __name__ == "__main__":
    main()
