# 🥗 SnapMeal: AI-Powered Recipe Recommender with Computer Vision & NLP

> *From Fridge to Feast — An intelligent system that "sees" your ingredients and crafts the perfect dish.*

SnapMeal is a smart, end-to-end meal recommendation system that uses **deep learning**, **computer vision**, and **natural language processing** to identify ingredients from an image and recommend recipes tailored to the user's dietary preferences. It integrates a fine-tuned **YOLOv8 model**, a **BERT-based semantic search engine**, and a sleek **Gradio-powered user interface**.

---

## 🧭 Table of Contents
- [🚀 Overview](#-overview)
- [🧠 Core Components](#-core-components)
  - [1. Ingredient Detection (YOLOv8)](#1-ingredient-detection-yolov8)
  - [2. Recipe Recommendation Engine (BERT + Fuzzy Logic)](#2-recipe-recommendation-engine-bert--fuzzy-logic)
  - [3. Gradio Interface](#3-gradio-interface)
- [📦 Project Structure](#-project-structure)
- [🛠️ Setup & Installation](#️-setup--installation)
- [📚 Future Work](#-future-work)

---

## 🚀 Overview

SnapMeal is not just a recipe app — it’s a system that:
- 📸 Accepts a **photo of your raw ingredients**
- 🧠 Identifies the ingredients via a **fine-tuned YOLOv8m object detector**
- 🧾 Uses **BERT embeddings** to semantically search a dataset of real-world recipes
- 🍱 Filters results using **fuzzy logic** based on your preferences (diet, cuisine, difficulty)
- 💬 Interacts with users via an intuitive **Gradio interface**

It's built for **speed**, **modularity**, and **extensibility** — perfect for both personal use and as a resume-worthy portfolio project.

---

## 🧠 Core Components

### 1. Ingredient Detection (YOLOv8)

> *Trained to see what's in your fridge — literally.*

SnapMeal uses **YOLOv8m** from Ultralytics, fine-tuned on a custom dataset of food ingredients.

- ✅ **Transfer Learning**: Frozen initial layers to preserve general features from COCO, fine-tuned later layers on food images.
- 📊 **Balanced Dataset**: Combined datasets to address class imbalance (e.g., more samples for "spring onion", underrepresented in COCO).
- 🏋️ **Training Setup**:
  - Image Size: 640x640  
  - Epochs: 30  
  - Batch Size: 8  
  - Optimized for CPU (low-resource friendly)  
- 🔖 Output: Returns a list of detected ingredients with confidence scores.

> YOLOv8m was chosen for its trade-off between speed and accuracy, especially on constrained hardware.

---

### 2. Recipe Recommendation Engine (BERT + Fuzzy Logic)

> *Where NLP meets your taste buds.*

The recommendation engine works in two layers:

#### 🔍 **a. Semantic Search with BERT**
- Recipe descriptions are embedded using `Sentence-BERT`.
- Detected ingredients are converted into text queries.
- Similarity is computed using **cosine similarity** in the embedding space.
- Top-N recipes are shortlisted.

#### 🔁 **b. Fuzzy Filtering**
- Uses **fuzzy logic rules** to filter based on:
  - Diet type (e.g., vegan, keto)
  - Cuisine (e.g., Indian, Mediterranean)
  - Time/Complexity (e.g., under 30 minutes)
- Recipes are ranked on a combined score:  
  `final_score = 0.7 * cosine_similarity + 0.3 * fuzzy_score`

This hybrid approach allows flexibility and interpretability — even if BERT returns perfect cosine matches, the fuzzy layer ensures personalization.

---

### 3. Gradio Interface

> *Talk to your meal assistant — no terminal needed.*

SnapMeal offers a clean **Gradio UI** that:
- Accepts user-uploaded food images 📷
- Asks optional inputs like:
  - Preferred cuisine
  - Dietary restrictions
  - Cooking time
- Displays:
  - Recognized ingredients
  - Top 3 recommended recipes with name, ingredients, and instructions

The UI binds everything together into a single experience — user-friendly, visually elegant, and ready for demo.

---

## 📦 Project Structure

<pre><code>SnapMeal/ 
├── backend/ 
│ ├── RAW_recipes.csv # Recipe dataset 
│ ├── recipe_embeddings.json # BERT-encoded vectors 
│ └── ingredient_identification.py # Scoring logic 
│ └── recipe_generator.py 
├── gradio_app.py # Launches the interface
├── models/ 
| ├── model1.pt
| ├── model2.pt
├── notebooks/ 
│ └── model_training.ipynb # EDA and model experiments 
├── assets/ 
│ └── demo_images/ # Sample input images 
├── requirements.txt # All dependencies 
└── System_Architecture.png # Visual system design </code></pre>


---

## 🛠️ Setup & Installation

```bash
# Clone repo
git clone https://github.com/ruru-lyy/SnapMeal.git
cd SnapMeal

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python gradio_app.py
```

## 📚 Future Work

🍳 Add step-by-step cooking video generation (text-to-video)

🔬 Model explainability for detected ingredients & recipe reasoning

🌐 Host on Hugging Face Spaces or Streamlit Cloud

📱 Mobile app version using Flutter + TensorFlow Lite

This is more than a project — it's a machine that eats vision and serves intelligence.
Designed with ambition, coded with precision. 🍽️

Made with love (and Tensor cores) by ruru-lyy 💖
