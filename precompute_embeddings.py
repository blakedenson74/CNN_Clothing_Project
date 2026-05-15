import os
import pickle
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor
from transformers import CLIPVisionModelWithProjection

IMAGE_DIR = "data/fashion-dataset/images"
METADATA_FILE = "data/fashion-dataset/styles.csv"

OUTPUT_FILE = "embeddings.pkl"

MAX_IMAGES = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")

model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.to(device)

model.eval()

print("Loading metadata...")

metadata = pd.read_csv(
    METADATA_FILE,
    on_bad_lines="skip"
)

image_paths = []
embeddings = []
item_metadata = []

count = 0

for _, row in metadata.iterrows():

    if count >= MAX_IMAGES:
        break

    image_id = str(row["id"])

    image_path = os.path.join(
        IMAGE_DIR,
        image_id + ".jpg"
    )

    if not os.path.exists(image_path):
        continue

    try:

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            embedding = outputs.image_embeds

        embedding = embedding / embedding.norm(
            p=2,
            dim=-1,
            keepdim=True
        )

        image_paths.append(image_path)

        embeddings.append(
            embedding.cpu().squeeze().numpy()
        )

        item_metadata.append({
            "productDisplayName": row.get(
                "productDisplayName",
                ""
            ),
            "articleType": row.get(
                "articleType",
                ""
            ),
            "baseColour": row.get(
                "baseColour",
                ""
            ),
            "usage": row.get(
                "usage",
                ""
            )
        })

        count += 1

        if count % 100 == 0:

            print(f"Processed {count} images")

    except Exception as e:

        print(f"Skipping {image_path}: {e}")

print("Saving embeddings...")

with open(OUTPUT_FILE, "wb") as f:

    pickle.dump({
        "image_paths": image_paths,
        "embeddings": embeddings,
        "metadata": item_metadata
    }, f)

print(f"Saved {len(image_paths)} embeddings.")