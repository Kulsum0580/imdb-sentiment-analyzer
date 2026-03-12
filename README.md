---
title: CineScope · Sentiment AI
emoji: 🎬
colorFrom: yellow
colorTo: orange
sdk: docker
app_file: app.py
pinned: true
---

# 🎬 CineScope — Movie Review Sentiment Analyzer

A deep learning NLP app that classifies movie reviews as **Positive**, **Negative**, or **Uncertain** using a **Bidirectional LSTM with Attention mechanism**, trained on the IMDB 50K dataset.

## 🧠 Model Architecture
- **Embedding Layer** — 10,000 vocab → 128-dim vectors
- **Bidirectional LSTM** — 128 units, reads sequence both directions
- **Custom Attention Layer** — focuses on sentiment-relevant words
- **Dense layers** — 64 → 1 (sigmoid output)

## 📊 Performance
- Test Accuracy: ~87%
- Dataset: IMDB 50K Movie Reviews
- Train/Test Split: 80/20

## ✨ Features
- Live sentiment prediction with confidence score
- Word-level importance visualizer
- Uncertainty detection (flags weak predictions)
- Review history tracking
- Beautiful dark cinematic UI
