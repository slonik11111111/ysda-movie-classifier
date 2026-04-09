import json
import torch
import torch.nn as nn
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel


MODEL_REPO = "slonik11111111/deberta-movie-genres"


@st.cache_resource
def load_model():
    genres_path = hf_hub_download(repo_id=MODEL_REPO, filename="genres.json")
    genres = json.load(open(genres_path, encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)

    backbone = AutoModel.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    clf_path = hf_hub_download(repo_id=MODEL_REPO, filename="classifier.pt")
    classifier = nn.Linear(backbone.config.hidden_size, len(genres))
    classifier.load_state_dict(
        torch.load(clf_path, map_location="cpu", weights_only=True)
    )

    backbone.eval()
    classifier.eval()
    return genres, tokenizer, backbone, classifier


def predict(title, description, genres, tokenizer, backbone, classifier, threshold=0.3):
    text = f"{title}: {description}" if title else description
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = backbone(**inputs)
        cls = out.last_hidden_state[:, 0]
        logits = classifier(cls)
        probs = torch.sigmoid(logits).squeeze().tolist()
    results = sorted(zip(genres, probs), key=lambda x: -x[1])
    return [(g, p) for g, p in results if p >= threshold]


st.title("Классификатор жанров фильма")
st.write("Введите название и описание фильма (на английском) - модель предскажет жанры.")

genres, tokenizer, backbone, classifier = load_model()

title = st.text_input("Название фильма", placeholder="Например: Inception")
description = st.text_area("Описание", placeholder="Например: Dom Cobb, a professional thief specializing in extracting secrets from within the subconscious during dream-sharing...")

if st.button("Определить жанры"):
    if not title and not description:
        st.warning("моя не умей предсказывать по пустой поля")
    else:
        results = predict(title, description, genres, tokenizer, backbone, classifier)
        if results:
            st.subheader("Жанры:")
            for genre, prob in results:
                st.progress(prob, text=f"{genre} — {prob:.0%}")
        else:
            st.info("Не удалось определить жанры с уверенностью выше 30%.(Если ты с девушкой стоит пикнуть другое кино)")