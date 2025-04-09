

from transformers import pipeline

explanation_generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_explanations(ingredients):
    """
    Generate a simple explanation describing each ingredient using BERT-style model.
    """
    explanations = []
    for ingredient in ingredients:
        prompt = f"What is {ingredient} used for in cooking?"
        response = explanation_generator(prompt, max_length=40, do_sample=False)
        explanation = response[0]['generated_text']
        explanations.append(f"{ingredient.capitalize()}: {explanation}")