"""
Generate recipe images using DALL-E 3 based on LLM recipe recommendations.

Usage:
    python generate_recipe_image.py --food "눈 건강 계란 노른자 볼" --ingredients "계란 노른자, 호박, 브로콜리, 당근, 블루베리"
    python generate_recipe_image.py --food "소고기 병아리콩 스튜" --ingredients "소고기, 병아리콩"
    python generate_recipe_image.py --food "사골 육수 시금치 현미 볼" --ingredients "사골 육수, 시금치, 현미" --breed "골든 리트리버"

Requires:
    pip install openai python-dotenv Pillow requests
    .env file with OPENAI_API_KEY=sk-...
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Paths
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent.parent / ".env"
OUTPUT_DIR = BASE_DIR / "output"


def load_env():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"[ERROR] OPENAI_API_KEY not found. Please set it in {ENV_PATH}")
        sys.exit(1)
    return api_key


def build_image_prompt(food_name, ingredients, breed=None, disease=None):
    """Build a DALL-E prompt for the recipe image."""
    prompt = (
        f"A real photograph of a homemade dog food dish. "
        f"Ingredients: {ingredients}. "
        f"Served in a simple white ceramic bowl on a pure white background. "
        f"Only the bowl and food, nothing else in the frame. No animals, no people, no props. "
        f"The ingredients look fresh, moist, and cooked. Realistic food textures. "
        f"Bright, even studio lighting. Slight overhead angle. "
        f"Clean, minimal commercial food photography. "
        f"No text, no labels, no illustrations, no cartoon style."
    )

    return prompt


def generate_image(client, prompt, size="1024x1024", quality="standard"):
    """Call DALL-E 3 to generate an image."""
    print(f"\n[INFO] Generating image with DALL-E 3...")
    print(f"[INFO] Prompt: {prompt[:100]}...")

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )

    image_url = response.data[0].url
    revised_prompt = response.data[0].revised_prompt
    return image_url, revised_prompt


def save_image(image_url, food_name):
    """Download and save the generated image."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename from food name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = food_name.replace(" ", "_")[:30]
    filename = f"{safe_name}_{timestamp}.png"
    filepath = OUTPUT_DIR / filename

    print(f"[INFO] Downloading image...")
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"[INFO] Saved: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate recipe images using DALL-E 3")
    parser.add_argument("--food", type=str, required=True, help="Recipe name in Korean")
    parser.add_argument("--ingredients", type=str, required=True, help="Comma-separated ingredients")
    parser.add_argument("--breed", type=str, default=None, help="Dog breed name (optional)")
    parser.add_argument("--disease", type=str, default=None, help="Disease name (optional)")
    parser.add_argument("--size", type=str, default="1024x1024",
                        choices=["1024x1024", "1792x1024", "1024x1792"],
                        help="Image size (default: 1024x1024)")
    parser.add_argument("--quality", type=str, default="standard",
                        choices=["standard", "hd"],
                        help="Image quality (default: standard)")
    args = parser.parse_args()

    # Setup
    api_key = load_env()
    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("  반려견 레시피 이미지 생성기 (DALL-E 3)")
    print("=" * 60)
    print(f"\n레시피: {args.food}")
    print(f"재료: {args.ingredients}")

    # Build prompt and generate
    prompt = build_image_prompt(args.food, args.ingredients)
    image_url, revised_prompt = generate_image(client, prompt, args.size, args.quality)

    print(f"\n[DALL-E revised prompt]\n{revised_prompt}")

    # Save image
    filepath = save_image(image_url, args.food)

    print(f"\n{'=' * 60}")
    print(f"완료! 이미지가 저장되었습니다: {filepath}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
