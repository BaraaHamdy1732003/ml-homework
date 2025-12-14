"""
extractor.py
Detect tops + pants in a full-body photo, crop them, describe with LLAVA (Ollama),
and find best matching images from local dataset using glm-4.6:cloud similarity.

Before running:
 - pip install ultralytics pillow numpy
 - Ensure Ollama models are installed: `ollama pull llava` and `ollama pull glm-4.6:cloud`
 - Place input image at INPUT_IMAGE or change path below
 - Set DATASET_FOLDER to your dataset: E:\barbar\ml\hw05\dataset\image
"""

import os
import subprocess
import json
import base64
import tempfile
import time
from PIL import Image
import numpy as np

# ===== CONFIG =====
# Path to ollama executable (Windows). Change if needed.
OLLAMA = r"C:\Users\ali\AppData\Local\Programs\Ollama\ollama.exe"

# Models
LLAVA_MODEL = "llava"               # multimodal model name installed in Ollama
SIM_MODEL = "glm-4.6:cloud"         # text model used for similarity scoring

# Files / folders (change to your exact paths if needed)
INPUT_IMAGE = r"E:\barbar\ml\hw05\image.png"
DATASET_FOLDER = r"E:\barbar\ml\hw05\dataset\image"

# YOLO model name (ultralytics will download if not present)
YOLO_MODEL = "yolov8n.pt"

# similarity threshold for printing matches
MIN_SIMILARITY = 1.0   # keep >= 0 to always get a best

# Timeout for ollama run (seconds)
OLLAMA_TIMEOUT = 60


# ===== Helpers for running Ollama (bytes safe) =====
def run_ollama_cmd(args_list, input_bytes=None, timeout=OLLAMA_TIMEOUT):
    """Run ollama subprocess and return stdout decoded as utf-8 (ignore decode errors)."""
    try:
        proc = subprocess.Popen(
            args_list,
            stdin=subprocess.PIPE if input_bytes is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate(input=input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return ""
    if stdout is None:
        return ""
    return stdout.decode("utf-8", errors="ignore")


def ollama_vision_prompt(image_path: str, prompt: str, model: str = LLAVA_MODEL):
    """Send an image (base64) + text prompt to a multimodal Ollama model via `ollama run`."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return ""
    with open(image_path, "rb") as f:
        img_b = f.read()
    img_b64 = base64.b64encode(img_b).decode("utf-8")
    payload = json.dumps({"model": model, "prompt": prompt, "images": [img_b64]})
    out = run_ollama_cmd([OLLAMA, "generate", model], input_bytes=payload.encode("utf-8"))

    return out.strip()


def ollama_text_score(user_text: str, item_text: str, model: str = SIM_MODEL):
    """
    Ask the text model to rate similarity (0-100) between user_text and item_text.
    Returns float score or 0.0 on parse fail.
    """
    prompt = (
        "Rate similarity between these two clothing descriptions from 0 to 100.\n\n"
        "User clothing:\n" + user_text + "\n\n"
        "Dataset item:\n" + item_text + "\n\n"
        "Return ONLY a number between 0 and 100.\n"
    )
    out = run_ollama_cmd([OLLAMA, "run", model], input_bytes=prompt.encode("utf-8"))
    if not out:
        return 0.0
    # try extract first number
    for token in out.split():
        token_clean = token.strip().strip(".,;:")
        try:
            val = float(token_clean.replace(",", "."))
            # clamp
            if val < 0: val = 0.0
            if val > 100: val = min(val, 100.0)
            return val
        except:
            continue
    return 0.0


# ===== YOLO detection and cropping =====
def detect_person_bboxes(image_path: str):
    """Run ultralytics YOLO detect to get person bounding boxes (xyxy). Returns list of (x1,y1,x2,y2)."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ERROR: ultralytics is required. Install: pip install ultralytics")
        raise

    model = YOLO(YOLO_MODEL)  # will download the weights if not present
    results = model.predict(source=image_path, save=False, conf=0.25, verbose=False)
    boxes = []
    for res in results:
        # res.boxes: for each detection
        if hasattr(res, "boxes"):
            for box in res.boxes:
                cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else None
                # COCO person class is 0
                if cls == 0:
                    xyxy = box.xyxy.cpu().numpy()[0]  # array [x1,y1,x2,y2]
                    boxes.append(tuple(int(v) for v in xyxy))
    return boxes


def crop_top_and_pants(image: Image.Image, bbox):
    """Given a person bbox, split it horizontally into top and pants crops.
       bbox = (x1,y1,x2,y2)"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    # split at approx 55% height (upper part = top)
    split = y1 + int(h * 0.55)
    # ensure within image
    split = max(y1 + 1, min(split, y2 - 1))
    top = image.crop((x1, y1, x2, split))
    pants = image.crop((x1, split, x2, y2))
    return top, pants


# ===== dataset description caching =====
def cache_dataset_descriptions(dataset_folder: str, llava_model=LLAVA_MODEL):
    """
    Describe each dataset image once with LLAVA and cache descriptions in dict:
    {filename: description}
    """
    cache = {}
    files = sorted(os.listdir(dataset_folder))
    for i, fn in enumerate(files):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(dataset_folder, fn)
        print(f"Describing dataset image {i+1}/{len(files)}: {fn}")
        desc = ollama_vision_prompt(path, "Describe the clothing item in one short sentence.", model=llava_model)
        if not desc:
            desc = "no description"
        cache[fn] = desc
        # small sleep to avoid spamming CLI
        time.sleep(0.3)
    return cache


# ===== main pipeline =====
def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Input image not found: {INPUT_IMAGE}")
        return

    print("\n======== STEP 1: DETECT PERSONS ========")
    person_boxes = detect_person_bboxes(INPUT_IMAGE)
    if not person_boxes:
        print("No person detected. Trying to describe whole image instead.")
        whole_desc = ollama_vision_prompt(INPUT_IMAGE, "Describe clothing items visible in this image.", model=LLAVA_MODEL)
        print("Description:\n", whole_desc)
        return

    print(f"Detected {len(person_boxes)} person(s). Cropping top/pants...")

    img = Image.open(INPUT_IMAGE).convert("RGB")

    # Cache dataset descriptions
    print("\n======== PREPARE DATASET DESCRIPTIONS ========")
    dataset_cache = cache_dataset_descriptions(DATASET_FOLDER)

    # For each person, crop top + pants, describe, then find best matches
    for idx, bbox in enumerate(person_boxes, start=1):
        print(f"\n--- Person #{idx} --- bbox={bbox}")
        top_crop, pants_crop = crop_top_and_pants(img, bbox)

        # Save temp crops to files so Ollama can read them
        tmp_dir = tempfile.gettempdir()
        top_path = os.path.join(tmp_dir, f"hw05_top_{idx}.jpg")
        pants_path = os.path.join(tmp_dir, f"hw05_pants_{idx}.jpg")
        top_crop.save(top_path)
        pants_crop.save(pants_path)

        # Describe with LLAVA
        print("Describing TOP crop with LLAVA...")
        top_desc = ollama_vision_prompt(top_path, "Describe this clothing item (type, color, pattern) in one sentence.", model=LLAVA_MODEL)
        print("Top description:", top_desc)

        print("Describing PANTS crop with LLAVA...")
        pants_desc = ollama_vision_prompt(pants_path, "Describe this clothing item (type, color, pattern) in one sentence.", model=LLAVA_MODEL)
        print("Pants description:", pants_desc)

        # Find best dataset matches (text-similarity via GLM)
        print("\nSearching best TOP match in dataset (text-similarity)...")
        best_top = ("", None, -1.0)   # (filename, path, score)
        for fn, item_desc in dataset_cache.items():
            # only compare to dataset items that likely are tops or pants; naive filter by word
            # We'll compare both and keep irrespective, user dataset is mixed
            score = ollama_text_score(top_desc, item_desc)
            if score > best_top[2]:
                best_top = (fn, os.path.join(DATASET_FOLDER, fn), score)

        print(f"Best TOP match: {best_top[0]} (score {best_top[2]:.1f})")
        if best_top[1] and best_top[2] >= MIN_SIMILARITY:
            # show crop and matched item
            print("Showing cropped TOP and matched dataset image...")
            try:
                top_crop.show(title="Detected TOP")
            except:
                pass
            try:
                Image.open(best_top[1]).show(title="Matched TOP")
            except:
                pass

        print("\nSearching best PANTS match in dataset (text-similarity)...")
        best_pants = ("", None, -1.0)
        for fn, item_desc in dataset_cache.items():
            score = ollama_text_score(pants_desc, item_desc)
            if score > best_pants[2]:
                best_pants = (fn, os.path.join(DATASET_FOLDER, fn), score)

        print(f"Best PANTS match: {best_pants[0]} (score {best_pants[2]:.1f})")
        if best_pants[1] and best_pants[2] >= MIN_SIMILARITY:
            print("Showing cropped PANTS and matched dataset image...")
            try:
                pants_crop.show(title="Detected PANTS")
            except:
                pass
            try:
                Image.open(best_pants[1]).show(title="Matched PANTS")
            except:
                pass

    print("\nDone.")

if __name__ == "__main__":
    main()
