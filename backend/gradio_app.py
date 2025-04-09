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
from recipe_generator import match_recipes

# Default user settings
default_preference = 'veg'
default_allergies = []
default_max_calories = 800


def process_image(image_path: str, preference: str, allergies: List[str], max_calories: int, progress: gr.Progress = gr.Progress()) -> Tuple[str, List[Dict[str, Any]], np.ndarray]:
    detected_ingredients, annotated_image = identify_ingredients(image_path)

    if not detected_ingredients:
        return "No ingredients detected in the image. Please try another image.", [], None

    top_recipes = match_recipes(
        detected_ingredients=detected_ingredients,
        preference=preference,
        allergies=allergies,
        max_calories=max_calories,
        top_n=3,
        progress=progress
    )

    result_message = f"Detected ingredients: {', '.join(detected_ingredients)}"
    return result_message, top_recipes, annotated_image

def format_recipe(recipe: Dict[str, Any]) -> str:
    formatted = f"## {recipe['Recipe']} (Match: {recipe['Match %']}%)\n\n"
    formatted += "### Ingredients You Have:\n"
    formatted += ", ".join(recipe['Common Ingredients']) if recipe['Common Ingredients'] else "None of the required ingredients detected"
    formatted += "\n\n### Shopping List (Missing Ingredients):\n"
    formatted += ", ".join(recipe['Missing Ingredients']) if recipe['Missing Ingredients'] else "You have all required ingredients!"
    formatted += "\n\n### Instructions:\n" + '\n'.join(ast.literal_eval(recipe['Steps']))
    return formatted

def bgr_to_rgb(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, max_width: int = 400) -> np.ndarray:
    if image is None:
        return None
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def recipe_finder_app(image, preference, allergies, max_calories, progress=gr.Progress()):
    if image is None:
        return "Please upload an image to detect ingredients.", "", "", "", None

    result_message, top_recipes, annotated_image = process_image(image, preference, allergies, max_calories, progress)
    recipe1 = format_recipe(top_recipes[0]) if len(top_recipes) > 0 else "No matching recipe found."
    recipe2 = format_recipe(top_recipes[1]) if len(top_recipes) > 1 else "No additional recipe found."
    recipe3 = format_recipe(top_recipes[2]) if len(top_recipes) > 2 else "No additional recipe found."

    rgb_annotated_image = bgr_to_rgb(annotated_image)
    resized_annotated_image = resize_image(rgb_annotated_image)
    return result_message, recipe1, recipe2, recipe3, resized_annotated_image

def create_gradio_interface():
    with gr.Blocks(title="SnapMeal") as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# 🍳 SnapMeal")
                gr.Markdown("Don't know what to cook with the ingredients left in your fridge?? Let us help you!")

        # States
        user_pref_state = gr.State(default_preference)
        user_allergy_state = gr.State(default_allergies)
        user_calorie_state = gr.State(default_max_calories)

        with gr.Group():
            gr.Markdown("## Your Preferences")
            dietary_preference = gr.Radio(choices=['veg', 'non-veg', 'any'], label="Dietary Preference", value=default_preference)
            allergy_input = gr.Textbox(label="Allergies (comma-separated)", placeholder="e.g., peanuts, dairy, gluten", value=", ".join(default_allergies))
            max_calorie_input = gr.Number(label="Max Calories per Recipe", value=default_max_calories, step=100)
            preference_button = gr.Button("Set Preferences")
            preference_output = gr.Textbox(label="Status")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image of Ingredients")
                submit_btn = gr.Button("Find Recipes", variant="primary")
            with gr.Column(scale=2):
                ingredients_output = gr.Textbox(label="Detected Ingredients")
                annotated_image_output = gr.Image(label="Detected Ingredients", show_label=True)

        with gr.Tabs():
            with gr.TabItem("Recipe 1"):
                recipe1_output = gr.Markdown()
            with gr.TabItem("Recipe 2"):
                recipe2_output = gr.Markdown()
            with gr.TabItem("Recipe 3"):
                recipe3_output = gr.Markdown()

        def update_preferences(pref, allergies_str, max_cal):
            allergies_list = [a.strip() for a in allergies_str.split(',') if a.strip()]
            return pref, allergies_list, int(max_cal), "Got it! Your preferences are saved."

        preference_button.click(
            fn=update_preferences,
            inputs=[dietary_preference, allergy_input, max_calorie_input],
            outputs=[user_pref_state, user_allergy_state, user_calorie_state, preference_output]
        )

        submit_btn.click(
            fn=recipe_finder_app,
            inputs=[image_input, user_pref_state, user_allergy_state, user_calorie_state],
            outputs=[ingredients_output, recipe1_output, recipe2_output, recipe3_output, annotated_image_output]
        )

        # Optional examples
        example_dir = "assets"
        examples = [os.path.join(example_dir, file) for file in os.listdir(example_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(example_dir) else []
        if examples:
            gr.Examples(examples=examples, inputs=image_input)

    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)
