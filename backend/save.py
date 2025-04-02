import sys
import os
import ast
import io
import pandas as pd
from fastapi import FastAPI
from PIL import Image
import gradio as gr
import uvicorn
from ingredient_identification import identify_ingredients
from recipe_finder import find_top_matching_recipes

# --- Configuration ---
backend_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(backend_dir)
sys.path.extend([backend_dir, root_dir])  # Ensure paths are included

RECIPES_CSV_PATH = os.path.join(root_dir, "backend/recipes.csv")

# --- Load Recipe Dataset ---
if not os.path.exists(RECIPES_CSV_PATH):
    raise FileNotFoundError(f"Error: Recipe dataset not found at {RECIPES_CSV_PATH}")

try:
    recipes_df = pd.read_csv(RECIPES_CSV_PATH)
    recipes_df["ingredients"] = recipes_df["ingredients"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    print(f"✅ Recipe dataset loaded from: {RECIPES_CSV_PATH}")
except Exception as e:
    raise RuntimeError(f"Error loading recipe dataset: {e}")

# --- Image Processing & Recipe Finder ---
def process_image_and_find_recipes(image):
    if image is None:
        return "Error: No image uploaded.", []

    try:
        image = image.convert("RGB")  # Ensure compatibility
        image_path = "temp_image.jpg"
        image.save(image_path)

        detected_ingredients = identify_ingredients(image_path)
        os.remove(image_path)  # Clean up

        if not detected_ingredients:
            return "No ingredients detected.", []

        top_recipes = find_top_matching_recipes(detected_ingredients, recipes_df)
        recipe_output = "<div style='color: #f06; font-weight: bold;'><h3>Top Matching Recipes:</h3></div>" if top_recipes else "<div style='color: #f06; font-weight: bold;'>No matching recipes found.</div>"

        for i, recipe in enumerate(top_recipes):
            recipe_output += f"""
                <div style='margin-bottom: 15px; padding: 15px; border: 1px solid #333; background-color: #222; color: #eee;'>
                    <h4 style='color: #f06;'>{i+1}. {recipe['Recipe Name']} (Match: {recipe['Match %']}%)</h4>
                    <p style='color: #ccc;'><strong>Instructions:</strong> {recipe['Instructions']}</p>
                    <p style='color: #ccc;'><strong>Required Ingredients:</strong> {', '.join(recipe['Required Ingredients'])}</p>
                    <p style='color: #ccc;'><strong>Common Ingredients:</strong> {', '.join(recipe['Common Ingredients'])}</p>
            """
            if recipe['Grocery List (Missing Ingredients)']:
                recipe_output += f"<p style='color: #f99;'><strong>Missing Ingredients:</strong> {', '.join(recipe['Grocery List (Missing Ingredients)'])}</p>"
            recipe_output += "</div>"

        return f"<div style='color: #f06; font-weight: bold;'>Detected Ingredients:</div> <div style='color: #eee;'>{', '.join(detected_ingredients)}</div>", recipe_output

    except Exception as e:
        return f"<div style='color: #f06; font-weight: bold;'>Error processing image:</div> <div style='color: #eee;'>{e}</div>", []

# --- Gradio Interface with Black and Pink Theme ---
with gr.Blocks(theme=gr.themes.Base(primary_hue="pink", secondary_hue="gray")) as iface:
    gr.Markdown("# <center><span style='color: #f06;'>Snap</span><span style='color: #333;'>Meal</span>: Ingredient Detection & Recipe Finder</center>")
    gr.Markdown("Upload an image of food to detect ingredients and find matching recipes.")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Food Image")
        with gr.Column():
            detected_ingredients_output = gr.Textbox(label="Detected Ingredients")
            matching_recipes_output = gr.HTML(label="Matching Recipes")
    image_input.upload(
        process_image_and_find_recipes,
        inputs=image_input,
        outputs=[detected_ingredients_output, matching_recipes_output],
    )

# --- FastAPI App ---
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "SnapMeal API is running. Access the Gradio interface at /"}

app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)