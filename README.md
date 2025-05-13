
# SnapMeal: AI-Powered Recipe Recommender Using Vision and Language

> From raw ingredients to a tailored meal plan 

SnapMeal combines computer vision and NLP to bridge the gap between what's in your kitchen and what you can cook. The system is built using a fine-tuned YOLOv8m object detector for ingredient recognition, Sentence-BERT for semantic recipe retrieval, and a modular interface developed in Gradio.

---

## Overview

SnapMeal is a lightweight yet intelligent system designed to:

* Take a user-uploaded image of raw ingredients
* Detect ingredients using a fine-tuned YOLOv8m model
* Embed recipes using Sentence-BERT and retrieve relevant matches via cosine similarity
* Refine suggestions using a fuzzy logic layer based on dietary and cuisine preferences
* Deliver results through a simple Gradio front-end

It’s structured to be fast, interpretable, and extensible

---

## Core Components

### 1. Ingredient Detection (YOLOv8m)

I opted for YOLOv8m after testing multiple variants from the YOLO family and others like SSD and EfficientDet. YOLOv8m struck the right balance i.e., good performance on food detection without requiring heavy GPU resources. Given my hardware limitations, this model was more realistic to fine-tune. It also offered a simpler training and deployment cycle compared to other object detectors.

The model was trained using transfer learning. Initial layers were frozen to retain general visual features, while deeper layers were fine-tuned on a custom dataset of labeled ingredient images. The dataset had to be curated to address class imbalance (e.g., some vegetables were underrepresented in public datasets like COCO).

Training Specs:

* Image Size: 640x640
* Epochs: 30
* Batch Size: 8
* Trained on local machine with RTX GPU, optimized for lower memory usage

The output is a ranked list of predicted ingredients with bounding boxes and confidence scores.

---

### 2. Recipe Recommendation Engine (BERT + Fuzzy Logic)

This module links vision outputs to natural language essentially turning detected objects into a meaningful meal.

#### a. Semantic Search with BERT

I used Sentence-BERT (all-MiniLM-L6-v2) for encoding both recipe descriptions and ingredient queries. The smaller BERT variant gave decent semantic representations without excessive latency or memory usage.

Cosine similarity is used to identify the top-k semantically similar recipes from a pre-embedded dataset. Simpler keyword methods or TF-IDF weren’t sufficient bec they missed context, especially when ingredients had alternative names or forms (e.g., "chickpeas" vs "garbanzo").

#### b. Fuzzy Rule-Based Filtering

The top-N results are filtered through fuzzy logic rules, allowing personalization without retraining. This layer checks:

* Dietary preference (vegan, keto, etc.)
* Cuisine type (e.g., Indian, Asian-fusion)
* Time/difficulty constraints

The scoring formula:
final\_score = 0.7 × semantic\_similarity + 0.3 × fuzzy\_logic\_score

This hybrid approach was chosen because it offers semantic accuracy with user-specific filtering without needing a complex recommendation model.

---

### 3. Gradio Interface

The user interface, built using Gradio, serves as the front-facing application. Users can:

* Upload a photo
* Enter optional filters like diet, cuisine, or cooking time
* View detected ingredients and top 3 recipe matches (with ingredients and instructions)

The Gradio UI made rapid prototyping easier, and it integrates seamlessly with Python backend logic. The goal was usability, not just model performance.

---

## Project Structure

<pre><code>SnapMeal/ 
├── backend/ 
│ ├── RAW_recipes.csv              # Main recipe dataset  
│ ├── recipe_embeddings.json       # Precomputed BERT vectors 
│ ├── ingredient_identification.py # Vision-based detection and parsing  
│ └── recipe_generator.py          # Semantic matching and filtering logic
├── gradio_app.py                  # Frontend launcher
├── models/ 
│ ├── yolov8m.pt                   # Fine-tuned model weights
├── notebooks/ 
│ └── model_training.ipynb         # Training and evaluation notebooks
├── assets/ 
│ └── demo_images/                 # Sample inputs
├── requirements.txt               # Dependencies
└── System_Architecture.png        # System diagram</code></pre>

---

## Setup & Installation

```bash
git clone https://github.com/ruru-lyy/SnapMeal.git
cd SnapMeal
pip install -r requirements.txt
python gradio_app.py
```

---

## Future Work

* Add explainability layer for both vision and language outputs
* Integrate text-to-video cooking guides (for recipe steps)
* Host using Hugging Face Spaces or Streamlit Cloud
* Explore mobile deployment with Flutter and TensorFlow Lite

---

## Author

Nirupama Laishram
Data Analyst & Aspiring Data Engineer — Bangalore
LinkedIn: [Nirupama Laishram](https://www.linkedin.com/in/nirupama-l-a14179221/)
