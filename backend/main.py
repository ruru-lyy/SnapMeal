#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import gradio as gr
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
import ast


from ingredient_identification import identify_ingredients
from recipe_finder import find_top_matching_recipes



def load_recipe_data():
    recipe_file_path = os.path.join("backend", "recipes.csv")
    if not os.path.exists(recipe_file_path):
        raise FileNotFoundError(f"Error: 'recipes.csv' not found. Place it in the '{recipe_file_path}' directory.")
    
    recipes_df = pd.read_csv(recipe_file_path)
    
    # Convert 'ingredients' column from string to list safely
    recipes_df["ingredients"] = recipes_df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
    )
    
    return recipes_df

def process_image(image_path: str) -> Tuple[str, List[Dict[str, Any]], np.ndarray]:
    detected_ingredients, annotated_image = identify_ingredients(image_path)
    
    try:
        recipes_df = load_recipe_data()
    except FileNotFoundError as e:
        return str(e), [], None
    
    top_recipes = find_top_matching_recipes(detected_ingredients, recipes_df)
    
    if not detected_ingredients:
        result_message = "No ingredients detected in the image. Please try another image."
    else:
        result_message = f"Detected ingredients: {', '.join(detected_ingredients)}"
    
    return result_message, top_recipes, annotated_image

def format_recipe(recipe: Dict[str, Any]) -> str:
    formatted = f"## {recipe['Recipe Name']} (Match: {recipe['Match %']}%)\n\n"
    
    formatted += "### Ingredients You Have:\n"
    if recipe['Common Ingredients']:
        formatted += ", ".join(recipe['Common Ingredients'])
    else:
        formatted += "None of the required ingredients detected"
    
    formatted += "\n\n### Shopping List (Missing Ingredients):\n"
    if recipe['Grocery List (Missing Ingredients)']:
        formatted += ", ".join(recipe['Grocery List (Missing Ingredients)'])
    else:
        formatted += "You have all required ingredients!"
    
    formatted += "\n\n### Instructions:\n"
    formatted += recipe['Instructions']
    
    return formatted

# convertin opencv image (BGR) to RGB format for Gradio
def bgr_to_rgb(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




# Main Gradio interface function
def recipe_finder_app(image):
    if image is None:
        return "Please upload an image to detect ingredients.", "", "", "", None
    
    result_message, top_recipes, annotated_image = process_image(image)
    
    recipe1 = format_recipe(top_recipes[0]) if len(top_recipes) > 0 else "No matching recipe found."
    recipe2 = format_recipe(top_recipes[1]) if len(top_recipes) > 1 else "No additional recipe found."
    recipe3 = format_recipe(top_recipes[2]) if len(top_recipes) > 2 else "No additional recipe found."
    
    rgb_annotated_image = bgr_to_rgb(annotated_image)
    
    return result_message, recipe1, recipe2, recipe3, rgb_annotated_image

def create_gradio_interface():
    with gr.Blocks(title="SnapMeal") as app:
        gr.Markdown("# 🍳 SnapMeal")
        gr.Markdown("Don't know what to cook with the ingredients left in your fridge?? Upload an image of your ingredients to get recipe recommendations now!")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image of Ingredients")
                submit_btn = gr.Button("Find Recipes", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Row():
                    ingredients_output = gr.Textbox(label="Detected Ingredients")
                
                with gr.Row():
                    annotated_image_output = gr.Image(label="Detected Ingredients", show_label=True)
        
        with gr.Tabs():
            with gr.TabItem("Recipe 1"):
                recipe1_output = gr.Markdown()
            with gr.TabItem("Recipe 2"):
                recipe2_output = gr.Markdown()
            with gr.TabItem("Recipe 3"):
                recipe3_output = gr.Markdown()
        
        submit_btn.click(
            fn=recipe_finder_app,
            inputs=[image_input],
            outputs=[ingredients_output, recipe1_output, recipe2_output, recipe3_output, annotated_image_output]
        )
        
        example_dir = "assets"
        examples = []
        if os.path.exists(example_dir):
            for file in os.listdir(example_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    examples.append(os.path.join(example_dir, file))
        
        if examples:
            gr.Examples(
                examples=examples,
                inputs=image_input,
            )
    
    return app




if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)