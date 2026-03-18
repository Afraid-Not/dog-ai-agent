"""
Few-shot LLM test for dog breed disease-based recipe recommendation.

Usage:
    python test_few_shot.py                          # interactive mode
    python test_few_shot.py --breed "Beagle"         # single breed
    python test_few_shot.py --breed "Beagle,Poodle"  # mixed breed
    python test_few_shot.py --breed "Pug" --disease "Cataracts"  # specify both

Requires:
    pip install openai python-dotenv
    .env file with OPENAI_API_KEY=sk-...
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data"
FEW_SHOT_PATH = BASE_DIR / "few_shot_samples.json"
CSV_PATH = DATA_DIR / "breed_disease_food_combined.csv"
BREED_DATA_PATH = BASE_DIR.parent / "breed" / "breed_data.json"
FOOD_TRANS_PATH = BASE_DIR / "food_translation.json"
ENV_PATH = BASE_DIR.parent.parent / ".env"

# Serving size guide (per meal)
SERVING_SIZE = {
    "소형": "100~150g",
    "중형": "200~300g",
    "대형": "400~500g",
}


def load_env():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"[ERROR] OPENAI_API_KEY not found. Please set it in {ENV_PATH}")
        sys.exit(1)
    return api_key


def load_breed_info():
    """Load breed_data.json and build English->Korean name and size mappings."""
    with open(BREED_DATA_PATH, "r", encoding="utf-8") as f:
        breeds = json.load(f)
    en_to_ko = {}
    en_to_size = {}
    for b in breeds:
        key = b["en"].lower()
        en_to_ko[key] = b["ko"]
        en_to_size[key] = b["size"]
    return en_to_ko, en_to_size


def get_breed_ko(en_name, en_to_ko):
    """Get Korean breed name with case-insensitive fuzzy matching."""
    key = en_name.lower()
    if key in en_to_ko:
        return en_to_ko[key]
    # Fuzzy: check if CSV breed name contains or is contained by a known name
    for en_key, ko in en_to_ko.items():
        if en_key in key or key in en_key:
            return ko
    return en_name  # fallback to English


def get_breed_size(en_name, en_to_size):
    """Get breed size with case-insensitive fuzzy matching."""
    key = en_name.lower()
    if key in en_to_size:
        return en_to_size[key]
    for en_key, size in en_to_size.items():
        if en_key in key or key in en_key:
            return size
    return None


def load_food_translation():
    """Load food_translation.json for English->Korean food name mapping."""
    with open(FOOD_TRANS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def translate_food(name, food_trans):
    """Translate a food name to Korean. Falls back to original if not found."""
    return food_trans.get(name, name)


def load_few_shot_samples():
    with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_breed_data():
    """Load breed_disease_food_combined.csv into a lookup structure."""
    breed_diseases = {}  # breed_CIDD -> list of {disease, genetic, foods}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            breed = row["breed_CIDD"].strip()
            disease = row["disease"].strip()
            if breed not in breed_diseases:
                breed_diseases[breed] = {}
            if disease and disease not in breed_diseases[breed]:
                breed_diseases[breed][disease] = {
                    "genetic_disease": int(row["genetic_disease"]),
                    "foods": [],
                }
            if disease:
                breed_diseases[breed][disease]["foods"].append(
                    {
                        "food": row["food"].strip(),
                        "ingredients": row["effective_ingredients"].strip(),
                    }
                )
    return breed_diseases


def merge_breed_diseases(breed_data, breeds):
    """Merge diseases from multiple breeds, prioritizing common diseases."""
    disease_breed_map = {}  # disease -> list of breeds that have it

    for breed in breeds:
        diseases = breed_data.get(breed, {})
        for disease, info in diseases.items():
            if disease not in disease_breed_map:
                disease_breed_map[disease] = {"breeds": [], "info": info}
            disease_breed_map[disease]["breeds"].append(breed)
            # Merge foods from all breeds
            existing_foods = {f["food"] for f in disease_breed_map[disease]["info"]["foods"]}
            for food in info["foods"]:
                if food["food"] not in existing_foods:
                    disease_breed_map[disease]["info"]["foods"].append(food)

    # Sort: common diseases (multiple breeds) first, then single-breed diseases
    common = {}
    single = {}
    for disease, data in disease_breed_map.items():
        if len(data["breeds"]) > 1:
            common[disease] = data
        else:
            single[disease] = data

    return common, single


def build_few_shot_messages(samples):
    """Convert few-shot samples into OpenAI chat messages."""
    messages = []
    for sample in samples:
        inp = sample["input"]
        out = sample["output"]

        size = inp.get("size", "")
        size_info = f" ({size})" if size else ""

        if inp["disease"]:
            genetic_str = "유전성" if inp["genetic_disease"] == 1 else "비유전성"
            user_msg = (
                f"우리 강아지 견종은 {inp['breed']}{size_info}입니다. "
                f"{inp['disease']} 위험이 있습니다 ({genetic_str}). "
                f"어떤 음식을 추천하시나요? 추천 이유, 재료, 만드는 법, 급여량도 알려주세요."
            )
        else:
            user_msg = (
                f"우리 강아지 견종은 {inp['breed']}{size_info}입니다. "
                f"어떤 질병 위험이 있고, 어떤 음식을 추천하시나요?"
            )

        assistant_msg = out["message"]
        if out["recommended_recipes"]:
            assistant_msg += "\n\n추천 레시피:\n"
            for i, recipe in enumerate(out["recommended_recipes"], 1):
                assistant_msg += (
                    f"\n  {i}. {recipe['food']}\n"
                    f"     재료: {recipe['ingredients']}\n"
                )
                if "reason" in recipe:
                    assistant_msg += f"     추천 이유: {recipe['reason']}\n"
                if "serving_size" in recipe:
                    assistant_msg += f"     급여량: {recipe['serving_size']}\n"
                if "recipe_steps" in recipe:
                    assistant_msg += "     만드는 법:\n"
                    for step_num, step in enumerate(recipe["recipe_steps"], 1):
                        assistant_msg += f"       {step_num}. {step}\n"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    return messages


def build_system_prompt():
    return (
        "당신은 반려견 영양 상담 전문 수의사입니다. "
        "견종, 질병 위험, 유전 여부를 기반으로 적합한 레시피와 효과적인 재료를 추천합니다.\n\n"
        "중요: 반드시 모든 응답을 한국어로 작성하세요. "
        "아래 few-shot 예시는 참고용 영어이지만, 실제 응답은 반드시 한국어로 해주세요.\n\n"
        "규칙:\n"
        "1. 해당 질병에 대해 간단히 설명하고, 유전성 여부를 언급하세요.\n"
        "2. 최대 3개의 레시피와 재료를 추천하세요.\n"
        "3. 각 레시피마다 '재료: ~~~' 형태로 재료를 나열하고, "
        "그 아래에 번호를 매겨 조리법을 단계별로 알려주세요.\n"
        "4. 각 레시피마다 해당 재료가 질병에 왜 도움이 되는지 이유를 설명하세요.\n"
        "5. 알려진 질병 위험이 없는 견종이라면 보호자를 안심시키고 "
        "균형 잡힌 식단을 유지하도록 권장하세요.\n"
        "6. 믹스견(혼합견)의 경우, 여러 견종에서 공통으로 나타나는 질병을 "
        "우선적으로 설명하고, 해당 질병에 맞는 레시피를 추천하세요. "
        "공통 질병이 없다면 각 견종의 주요 질병을 종합하여 추천하세요.\n"
        "7. 견종의 크기(소형/중형/대형)에 따라 1회 급여량을 안내하세요. "
        "소형: 100~150g, 중형: 200~300g, 대형: 400~500g.\n"
        "8. 견종명과 레시피명은 반드시 한국어로 표기하세요. "
        "사용자 질문에 포함된 한국어 레시피명을 그대로 사용하세요.\n"
        "9. 따뜻하고 전문적인 어조로 응답하세요."
    )


def select_breed(breed_data):
    """Interactive breed selection."""
    breeds = sorted(breed_data.keys())
    print("\n=== Available Breeds ===")
    for i, b in enumerate(breeds):
        print(f"  [{i:3d}] {b}")
    print()

    while True:
        choice = input("Enter breed name or index number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(breeds):
                return breeds[idx]
        else:
            matches = [b for b in breeds if choice.lower() in b.lower()]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                print(f"  Multiple matches: {matches}")
                continue
        print("  Not found. Try again.")


def select_disease(breed_data, breed):
    """Interactive disease selection."""
    diseases = breed_data.get(breed, {})
    if not diseases:
        return None, None

    disease_list = sorted(diseases.keys())
    print(f"\n=== Diseases for {breed} ===")
    for i, d in enumerate(disease_list):
        genetic_label = "genetic" if diseases[d]["genetic_disease"] == 1 else "other"
        print(f"  [{i:2d}] {d} ({genetic_label})")
    print()

    while True:
        choice = input("Enter disease name or index number (or 'all' for all): ").strip()
        if choice.lower() == "all":
            return "all", None
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(disease_list):
                d = disease_list[idx]
                return d, diseases[d]
        else:
            matches = [d for d in disease_list if choice.lower() in d.lower()]
            if len(matches) == 1:
                return matches[0], diseases[matches[0]]
            elif len(matches) > 1:
                print(f"  Multiple matches: {matches}")
                continue
        print("  Not found. Try again.")


def build_user_query(breed, disease_name, disease_info, breed_ko=None, breed_size=None, food_trans=None):
    """Build the user query for the LLM."""
    display_name = breed_ko if breed_ko else breed
    size_info = f" ({breed_size}, 1회 급여량 {SERVING_SIZE[breed_size]})" if breed_size else ""
    ft = food_trans or {}

    if disease_name is None:
        return (
            f"우리 강아지 견종은 {display_name}{size_info}입니다. "
            f"어떤 질병 위험이 있고, 어떤 음식을 추천하시나요?"
        )

    genetic_str = "유전성" if disease_info["genetic_disease"] == 1 else "비유전성"
    foods_context = disease_info["foods"][:5]
    food_list = ", ".join(translate_food(f['food'], ft) for f in foods_context)

    return (
        f"우리 강아지 견종은 {display_name}{size_info}입니다. "
        f"{disease_name} 위험이 있습니다 ({genetic_str}). "
        f"선택 가능한 레시피: {food_list}. "
        f"어떤 음식을 추천하시나요? 추천 이유, 재료, 만드는 법, 급여량도 알려주세요."
    )


def build_mix_query(breeds, common_diseases, single_diseases, en_to_ko, en_to_size, food_trans=None):
    """Build the user query for mixed breed dogs."""
    ft = food_trans or {}
    breed_ko_list = [get_breed_ko(b, en_to_ko) for b in breeds]
    breed_str = " + ".join(breed_ko_list)

    # Determine size for mixed breed (use the largest)
    size_priority = {"대형": 3, "중형": 2, "소형": 1}
    sizes = [get_breed_size(b, en_to_size) for b in breeds]
    sizes = [s for s in sizes if s]
    mix_size = max(sizes, key=lambda s: size_priority.get(s, 0)) if sizes else None
    size_info = f" ({mix_size}, 1회 급여량 {SERVING_SIZE[mix_size]})" if mix_size else ""

    query = f"우리 강아지는 {breed_str} 믹스견{size_info}입니다.\n"

    if common_diseases:
        query += "\n공통 질병 위험 (여러 견종에서 공통):\n"
        for disease, data in common_diseases.items():
            genetic_str = "유전성" if data["info"]["genetic_disease"] == 1 else "비유전성"
            from_breeds = ", ".join(get_breed_ko(b, en_to_ko) for b in data["breeds"])
            foods = data["info"]["foods"][:3]
            food_list = ", ".join(translate_food(f["food"], ft) for f in foods)
            query += f"- {disease} ({genetic_str}, {from_breeds} 공통) / 레시피: {food_list}\n"

    if single_diseases:
        query += "\n개별 질병 위험:\n"
        for disease, data in single_diseases.items():
            genetic_str = "유전성" if data["info"]["genetic_disease"] == 1 else "비유전성"
            from_breed = get_breed_ko(data["breeds"][0], en_to_ko)
            foods = data["info"]["foods"][:3]
            food_list = ", ".join(translate_food(f["food"], ft) for f in foods)
            query += f"- {disease} ({genetic_str}, {from_breed}) / 레시피: {food_list}\n"

    query += "\n공통 질병을 우선으로 추천해주세요. 추천 이유, 재료, 만드는 법, 급여량도 알려주세요."
    return query


def call_llm(client, system_prompt, few_shot_messages, user_query, model="gpt-4o-mini"):
    """Call the OpenAI API with few-shot examples."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_query})

    print(f"\n[INFO] Calling {model}...")
    print(f"[INFO] Total messages: {len(messages)} (1 system + {len(few_shot_messages)} few-shot + 1 query)\n")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Few-shot LLM test for dog recipe recommendation")
    parser.add_argument("--breed", type=str, help="Dog breed name (comma-separated for mixed breeds, e.g. 'Beagle,Poodle')")
    parser.add_argument("--disease", type=str, help="Disease name")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    args = parser.parse_args()

    # Setup
    api_key = load_env()
    client = OpenAI(api_key=api_key)
    samples = load_few_shot_samples()
    breed_data = load_breed_data()
    en_to_ko, en_to_size = load_breed_info()
    food_trans = load_food_translation()
    system_prompt = build_system_prompt()
    few_shot_messages = build_few_shot_messages(samples)

    print("=" * 60)
    print("  반려견 질병 기반 레시피 추천 시스템 (Few-Shot LLM)")
    print("=" * 60)

    # Select breed(s)
    if args.breed:
        breed_inputs = [b.strip() for b in args.breed.split(",")]
    else:
        breed_inputs = [select_breed(breed_data)]

    # Resolve breed names
    resolved_breeds = []
    for b_input in breed_inputs:
        matches = [b for b in breed_data if b_input.lower() in b.lower()]
        if not matches:
            print(f"[ERROR] 견종 '{b_input}'을(를) 찾을 수 없습니다.")
            sys.exit(1)
        resolved_breeds.append(matches[0])

    is_mix = len(resolved_breeds) > 1

    if is_mix:
        # Mixed breed mode
        breed_ko_list = [get_breed_ko(b, en_to_ko) for b in resolved_breeds]
        breed_str = " + ".join(breed_ko_list)
        print(f"\n견종 (믹스): {breed_str}")

        common, single = merge_breed_diseases(breed_data, resolved_breeds)

        if not common and not single:
            print("[INFO] 해당 견종들의 질병 정보가 없습니다.")
        else:
            if common:
                print(f"\n[공통 질병] {len(common)}개 (우선)")
                for d, data in common.items():
                    breeds_ko = ", ".join(get_breed_ko(b, en_to_ko) for b in data["breeds"])
                    print(f"  - {d} ({breeds_ko})")
            if single:
                print(f"\n[개별 질병] {len(single)}개")
                for d, data in single.items():
                    breed_ko = get_breed_ko(data["breeds"][0], en_to_ko)
                    print(f"  - {d} ({breed_ko})")

            query = build_mix_query(resolved_breeds, common, single, en_to_ko, en_to_size, food_trans)
            print(f"\n[Query] {query}")
            result = call_llm(client, system_prompt, few_shot_messages, query, args.model)
            print(f"\n[Response]\n{result}")
    else:
        # Single breed mode
        breed = resolved_breeds[0]
        breed_ko = get_breed_ko(breed, en_to_ko)
        breed_size = get_breed_size(breed, en_to_size)
        size_str = f" ({breed_size})" if breed_size else ""
        print(f"\n견종: {breed_ko}{size_str}")

        # Select disease
        if args.disease:
            diseases = breed_data.get(breed, {})
            matches = [d for d in diseases if args.disease.lower() in d.lower()]
            if not matches:
                print(f"[ERROR] '{args.disease}' 질병을 {breed_ko}에서 찾을 수 없습니다.")
                sys.exit(1)
            disease_name = matches[0]
            disease_info = diseases[disease_name]
            print(f"질병: {disease_name}")
        else:
            disease_name, disease_info = select_disease(breed_data, breed)

        # Handle "all" diseases
        if disease_name == "all":
            diseases = breed_data.get(breed, {})
            for d_name in sorted(diseases.keys()):
                d_info = diseases[d_name]
                print(f"\n{'─' * 60}")
                print(f"질병: {d_name} ({'유전성' if d_info['genetic_disease'] == 1 else '비유전성'})")
                print(f"{'─' * 60}")
                query = build_user_query(breed, d_name, d_info, breed_ko, breed_size, food_trans)
                print(f"[Query] {query}")
                result = call_llm(client, system_prompt, few_shot_messages, query, args.model)
                print(f"\n[Response]\n{result}")
        else:
            query = build_user_query(breed, disease_name, disease_info, breed_ko, breed_size, food_trans)
            print(f"\n[Query] {query}")
            result = call_llm(client, system_prompt, few_shot_messages, query, args.model)
            print(f"\n[Response]\n{result}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
