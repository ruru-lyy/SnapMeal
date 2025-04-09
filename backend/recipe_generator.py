import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import os
import ast
import torch
import gradio as gr 
import random

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('backend\\RAW_recipes.csv')
df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
df = df[df['ingredients'].apply(lambda x: isinstance(x, list))]
df = df.dropna(subset=['nutrition', 'tags'])

EMBEDDINGS_PATH = 'recipe_embeddings.json'
recipe_embeddings = {}
try:
    with open(EMBEDDINGS_PATH, 'r') as f:
        recipe_embeddings = json.load(f)
    print(f"Loaded {len(recipe_embeddings)} pre-computed embeddings.")
except FileNotFoundError:
    print(f"Error: {EMBEDDINGS_PATH} not found. Run generate_embeddings.py first!")
    recipe_embeddings = {}

def is_veg(ingredients):
    non_veg_keywords = ['chicken', 'beef', 'pork', 'lamb', 'shrimp', 'fish', 'mutton', 'meat', 'bacon', 'egg']
    return not any(word.lower() in ing.lower() for ing in ingredients for word in non_veg_keywords)

# 🧠 Fuzzy ingredient matcher
def get_fuzzy_matches(recipe_ingredients, detected_ingredients, threshold=0.7):
    common = []
    missing = []

    for r_ing in recipe_ingredients:
        matched = False
        for d_ing in detected_ingredients:
            sim = util.cos_sim(
                bert_model.encode(r_ing, convert_to_tensor=True),
                bert_model.encode(d_ing, convert_to_tensor=True)
            )[0][0].item()
            if sim > threshold:
                common.append(r_ing)
                matched = True
                break
        if not matched:
            missing.append(r_ing)
    
    match_perc = round((len(common) / len(recipe_ingredients)) * 100, 2) if recipe_ingredients else 0.0
    return match_perc, common, missing

def semantic_similarity(detected_ingredients, recipe_ingredients):
    try:
        query = ', '.join(detected_ingredients)
        ref_str = ', '.join(recipe_ingredients)
        emb1 = bert_model.encode(query, convert_to_tensor=True)
        if ref_str in recipe_embeddings:
            emb2 = torch.tensor(recipe_embeddings[ref_str])
            return float(util.pytorch_cos_sim(emb1, emb2)[0][0])
        else:
            return 0.0
    except Exception as e:
        print(f"Error in semantic_similarity: {e}")
        return 0.0

def match_recipes(detected_ingredients, preference='veg', allergies=[], max_calories=None, top_n=5, progress: gr.Progress = gr.Progress()):
    matches = []
    random_df = df.sample(n=min(500, len(df)), random_state=random.randint(0, 1000))
    total_recipes = len(random_df)

    for i, (_, row) in enumerate(random_df.iterrows()):
        progress(i / total_recipes, desc="Searching Recipes")
        try:
            recipe_ingredients = row['ingredients']

            if preference == 'veg' and not is_veg(recipe_ingredients):
                continue

            if any(allergy in ' '.join(recipe_ingredients).lower() for allergy in allergies):
                continue

            nutrition = ast.literal_eval(row['nutrition'])
            if max_calories and nutrition[0] > max_calories:
                continue

            # 🧠 Use fuzzy matching instead of exact match
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
                'Calories': nutrition[0],
                'Tags': row['tags'],
                'Time (mins)': row['minutes'],
            })

        except Exception as e:
            print(f" Skipping row due to error: {e}")
            continue

    matches.sort(key=lambda x: (-x['Final Score'], x['Num Missing']))
    return matches[:top_n]

if __name__ == '__main__':
    # Example Test
    detected_ingredients = ['onion', 'garlic', 'potato', 'olive oil']
    allergy_list = ['milk']
    results = match_recipes(detected_ingredients, preference='veg', allergies=allergy_list, max_calories=500, top_n=3)

    for i, res in enumerate(results, 1):
        print(f"\n🔸 Top {i} Recipe: {res['Recipe']} ({res['Match %']:.2f}%)")
        print(f"Calories: {res['Calories']}")
        print(f"Time Required: {res['Time (mins)']} mins")
        print(f"Common Ingredients: {res['Common Ingredients']}")
        print(f"Missing Ingredients: {res['Missing Ingredients']}")
        print(f"Tags: {res['Tags']}")
        print(f"Steps: {res['Steps']}...")
