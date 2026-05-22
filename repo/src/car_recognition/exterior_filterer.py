import argparse
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ExteriorFilterer:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading CLIP model {model_id} on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.labels = [
            "a photo of the outside of a car showing the full body, paint, wheels, or bumper",
            "a photo of the inside of a car showing the dashboard, steering wheel, seats, or center console",
            "a photo of a car engine or motor under the hood",
            "a photo of the trunk or boot of a car, open or closed, showing cargo area",
        ]
        self.extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def filter_images(self, input_folder: str, dry_run: bool = False):
        input_path = Path(input_folder)
        if not input_path.is_dir():
            print(f"Error: {input_folder} is not a valid directory.")
            return

        all_images = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in self.extensions]
        total_images = len(all_images)

        if total_images == 0:
            print("No valid images found in the specified directory.")
            return

        kept_count = 0
        deleted_count = 0

        print(f"Processing {total_images} images...")

        with torch.no_grad():
            for i, img_path in enumerate(all_images, 1):
                try:
                    image = Image.open(img_path).convert("RGB")

                    inputs = self.processor(
                        text=self.labels,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)

                    # Keep only if exterior (index 0) has the highest probability
                    best_label = probs[0].argmax().item()
                    if best_label != 0:
                        if dry_run:
                            print(f"[{i}/{total_images}] would delete: {img_path.name}")
                        else:
                            img_path.unlink()
                            print(f"[{i}/{total_images}] deleted: {img_path.name}")
                        deleted_count += 1
                    else:
                        kept_count += 1

                except Exception as e:
                    print(f"[{i}/{total_images}] Warning: Could not process {img_path.name} due to error: {e}")
                    continue

        print("\n--- Summary ---")
        print(f"Total processed: {total_images}")
        print(f"Kept: {kept_count}")
        print(f"Deleted: {deleted_count}")
        if dry_run:
            print("Note: This was a dry run, no files were actually deleted.")

def main():
    parser = argparse.ArgumentParser(description="Filter car images by removing interiors using CLIP zero-shot classification.")
    parser.add_argument("--input", type=str, default="datasets/car_images_v3", help="Path to the folder containing car images")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without actually deleting")
    args = parser.parse_args()

    filterer = ExteriorFilterer()
    filterer.filter_images(args.input, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
