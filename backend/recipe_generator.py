import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import os
import ast
import torch
import gradio as gr
import random

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and prepare dataset
df = pd.read_csv('backend\\RAW_recipes.csv')
df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
df = df[df['ingredients'].apply(lambda x: isinstance(x, list))]
df = df.dropna(subset=['nutrition', 'tags'])

# Load precomputed recipe embeddings
EMBEDDINGS_PATH = 'backend\\recipe_embeddings.json'
recipe_embeddings = {}
try:
    with open(EMBEDDINGS_PATH, 'r') as f:
        recipe_embeddings = json.load(f)
    print(f"Loaded {len(recipe_embeddings)} pre-computed embeddings.")
except FileNotFoundError:
    print(f"Error: {EMBEDDINGS_PATH} not found. Run generate_embeddings.py first!")
    recipe_embeddings = {}

# Utility: Veg check
def is_veg(ingredients):
    non_veg_keywords = ['chicken', 'beef', 'pork', 'lamb', 'shrimp', 'fish', 'mutton', 'meat', 'bacon', 'egg']
    return not any(word.lower() in ing.lower() for ing in ingredients for word in non_veg_keywords)

# Optimized Fuzzy Matching
def get_fuzzy_matches(recipe_ingredients, detected_ingredients, threshold=0.7):
    if not recipe_ingredients or not detected_ingredients:
        return 0.0, [], recipe_ingredients

    recipe_embs = bert_model.encode(recipe_ingredients, convert_to_tensor=True)
    detected_embs = bert_model.encode(detected_ingredients, convert_to_tensor=True)
    sims = util.cos_sim(recipe_embs, detected_embs)

    match_mask = torch.any(sims > threshold, dim=1)
    common = [ing for ing, match in zip(recipe_ingredients, match_mask) if match]
    missing = [ing for ing, match in zip(recipe_ingredients, match_mask) if not match]

    match_perc = round((len(common) / len(recipe_ingredients)) * 100, 2)
    return match_perc, common, missing

# Semantic similarity (cached)
embedding_cache = {}

def encode_text(text):
    return bert_model.encode(text, convert_to_tensor=True)

def semantic_similarity(detected_ingredients, recipe_ingredients):
    try:
        query = ', '.join(detected_ingredients)
        ref_str = ', '.join(recipe_ingredients)

        if query not in embedding_cache:
            embedding_cache[query] = encode_text(query)
        emb1 = embedding_cache[query]

        if ref_str in recipe_embeddings:
            emb2 = torch.tensor(recipe_embeddings[ref_str])
        else:
            if ref_str not in embedding_cache:
                embedding_cache[ref_str] = encode_text(ref_str)
            emb2 = embedding_cache[ref_str]

        return float(util.pytorch_cos_sim(emb1, emb2)[0][0])
    except Exception as e:
        print(f"Error in semantic_similarity: {e}")
        return 0.0

# Filter data BEFORE searching

def filter_dataframe(detected_ingredients, preference, allergies, max_calories):
    filtered = df.copy()

    if preference == 'veg':
        filtered = filtered[filtered['ingredients'].apply(is_veg)]

    if allergies:
        allergy_check = lambda ings: not any(allergy in ' '.join(ings).lower() for allergy in allergies)
        filtered = filtered[filtered['ingredients'].apply(allergy_check)]

    if max_calories:
        def calorie_check(row):
            try:
                return ast.literal_eval(row['nutrition'])[0] <= max_calories
            except:
                return False
        filtered = filtered[filtered.apply(calorie_check, axis=1)]

    return filtered

# Master matcher
def match_recipes(detected_ingredients, preference='veg', allergies=[], max_calories=None, top_n=5, progress: gr.Progress = gr.Progress()):
    matches = []
    filtered_df = filter_dataframe(detected_ingredients, preference, allergies, max_calories)
    random_df = filtered_df.sample(n=min(500, len(filtered_df)), random_state=random.randint(0, 1000))
    total_recipes = len(random_df)

    for i, (_, row) in enumerate(random_df.iterrows()):
        progress(i / total_recipes, desc="Searching Recipes")
        try:
            recipe_ingredients = row['ingredients']

            match_perc, common, missing = get_fuzzy_matches(recipe_ingredients, detected_ingredients)
            semantic_score = semantic_similarity(detected_ingredients, recipe_ingredients)
            total_score = 0.7 * (match_perc / 100) + 0.3 * semantic_score

            matches.append({
                'Recipe': row['name'],
                'Match %': match_perc,
                'Semantic Score': round(semantic_score, 2),
                'Final Score': round(total_score, 4),
                'Common Ingredients': common,
                'Missing Ingredients': missing,
                'Num Missing': len(missing),
                'Steps': row['steps'],
                'Calories': ast.literal_eval(row['nutrition'])[0],
                'Tags': row['tags'],
                'Time (mins)': row['minutes'],
            })

        except Exception as e:
            print(f" Skipping row due to error: {e}")
            continue

    matches.sort(key=lambda x: (-x['Final Score'], x['Num Missing']))
    return matches[:top_n]

# Test
if __name__ == '__main__':
    detected_ingredients = ['onion', 'garlic', 'potato', 'olive oil']
    allergy_list = ['milk']
    results = match_recipes(detected_ingredients, preference='veg', allergies=allergy_list, max_calories=500, top_n=3)

    for i, res in enumerate(results, 1):
        print(f"\n\U0001F538 Top {i} Recipe: {res['Recipe']} ({res['Match %']:.2f}%)")
        print(f"Calories: {res['Calories']}")
        print(f"Time Required: {res['Time (mins)']} mins")
        print(f"Common Ingredients: {res['Common Ingredients']}")
        print(f"Missing Ingredients: {res['Missing Ingredients']}")
        print(f"Tags: {res['Tags']}")
        print(f"Steps: {res['Steps']}...")
