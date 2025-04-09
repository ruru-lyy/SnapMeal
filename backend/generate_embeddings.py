# generate_embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import os
import ast


bert_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('backend\\RAW_recipes.csv')

df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
df = df[df['ingredients'].apply(lambda x: isinstance(x, list))]
df = df.dropna(subset=['nutrition', 'tags'])

# Function to generate embeddings
def generate_recipe_embeddings(dataframe):
    embeddings = {}
    for index, row in dataframe.iterrows():
        ingredients_str = ', '.join(row['ingredients'])
        try:
            embedding = bert_model.encode(ingredients_str).tolist() # Convert to list for JSON serialization
            embeddings[ingredients_str] = embedding
            if index % 1000 == 0:
                print(f"Processed {index} recipes...")
        except Exception as e:
            print(f"Error encoding ingredients for recipe ID {row['id']}: {e}")
    return embeddings

if __name__ == "__main__":
    print("Generating recipe embeddings...")
    recipe_embeddings = generate_recipe_embeddings(df)

    # Save embeddings to a JSON file
    output_path = 'recipe_embeddings.json'
    with open(output_path, 'w') as f:
        json.dump(recipe_embeddings, f)

    print(f"Recipe embeddings saved to {output_path}")