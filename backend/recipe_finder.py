# -*- coding: utf-8 -*-
import pandas as pd
import ast
import os
import numpy as np

# fn to calculate match percentage
def get_match_percentage(recipe_ingredients, detected_list):
    if not recipe_ingredients:  
        return 0, [], recipe_ingredients  

    common_ingredients = list(set(recipe_ingredients) & set(detected_list))
    match_percentage = (len(common_ingredients) / len(recipe_ingredients)) * 100
    missing_ingredients = list(set(recipe_ingredients) - set(detected_list))

    return round(match_percentage, 2), common_ingredients, missing_ingredients

# fn to find top 3 matching recipes
def find_top_matching_recipes(detected_ingredients, recipes_df):
    recipe_results = []
    
    for _, row in recipes_df.iterrows():
        match_percentage, common_ing, missing_ing = get_match_percentage(row.get("ingredients", []), detected_ingredients)

        recipe_results.append({
            "Recipe Name": row["name"],
            "Instructions": row["steps"],
            "Required Ingredients": row["ingredients"],
            "Common Ingredients": common_ing,
            "Grocery List (Missing Ingredients)": missing_ing,
            "Match %": match_percentage
        })

    recipe_results = sorted(recipe_results, key=lambda x: x["Match %"], reverse=True)

    return recipe_results[:3]  



if __name__ == '__main__':
    detected_ingredients_example = ["tomato", "onion", "garlic", "olive oil"]

    recipe_file_path = os.path.join("backend", "recipes.csv")
    if not os.path.exists(recipe_file_path):
        raise FileNotFoundError(f"Error: 'recipes.csv' not found. Place it in the '{recipe_file_path}' directory.")

    # Load recipe data
    recipes_df = pd.read_csv(recipe_file_path)

    # Convert 'ingredients' column from string to list safely
    recipes_df["ingredients"] = recipes_df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
    )

    # Find top 3 matching recipes (no filtering by match percentage)
    top_recipes = find_top_matching_recipes(detected_ingredients_example, recipes_df)
    
    # Print results
    if not top_recipes:
        print("No matching recipes found.")
    else:
        for index, recipe in enumerate(top_recipes, start=1):
            print(f"Top {index} Recipe: {recipe['Recipe Name']}, Match: {recipe['Match %']}%")
            print(f"Instructions: {recipe['Instructions']}")
            print(f"Required Ingredients: {recipe['Required Ingredients']}")
            print(f"Common Ingredients Detected: {recipe['Common Ingredients']}")
            print(f"Grocery List (Missing Ingredients): {recipe['Grocery List (Missing Ingredients)']}")
            print("-" * 80)
