import os
import pickle
import re
import torch
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = "embeddings.pkl"

st.set_page_config(page_title="Clothing Item Retrieval", layout="wide")

st.title("Vision-Language Clothing Item Retrieval")
st.write("Upload a clothing image. BLIP describes it, and CLIP retrieves similar clothing items.")

@st.cache_resource
def load_clip_model():
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return model, processor

@st.cache_data
def load_saved_embeddings():
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)

    return data["image_paths"], data["embeddings"], data["metadata"]

clip_model, clip_processor = load_clip_model()
blip_model, blip_processor = load_blip_model()

if not os.path.exists(EMBEDDINGS_FILE):
    st.error("embeddings.pkl not found. Run: python precompute_embeddings.py")
    st.stop()

image_paths, dataset_embeddings, metadata = load_saved_embeddings()

def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = clip_model(**inputs)
        embedding = outputs.image_embeds

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding

def clean_caption(caption):
    caption = caption.lower().strip()
    caption = re.sub(r"\s+", " ", caption)

    remove_phrases = [
        "a close up of",
        "a picture of",
        "an image of",
        "a photo of",
        "there is",
        "there are"
    ]

    for phrase in remove_phrases:
        caption = caption.replace(phrase, "")

    caption = caption.strip(" .")

    if caption == "":
        caption = "clothing item"

    return caption

def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5
        )

    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    return clean_caption(caption)

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(query_image, width=300)

    caption = generate_caption(query_image)

    st.subheader("BLIP Generated Description")
    st.write(caption)

    query_embedding = get_image_embedding(query_image)
    query_embedding = query_embedding.squeeze().numpy().reshape(1, -1)

    similarities = cosine_similarity(query_embedding, dataset_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:5]

    st.subheader("Top Similar Clothing Items")

    cols = st.columns(5)

    for col, idx in zip(cols, top_indices):
        item = metadata[idx]

        with col:
            st.image(image_paths[idx], use_container_width=True)

            st.write("Similarity:", round(float(similarities[idx]), 3))

            if item.get("productDisplayName"):
                st.write(item["productDisplayName"])

            st.caption(
                f"{item.get('baseColour', '')} "
                f"{item.get('articleType', '')} | "
                f"{item.get('usage', '')}"
            )