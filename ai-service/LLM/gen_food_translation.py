"""Generate food_translation.json from CSV food names."""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_PATH = DATA_DIR / "breed_disease_food_combined_v2.csv"
OUT_PATH = Path(__file__).resolve().parent / "food_translation.json"

# Ingredient/term translation map
TRANS = {
    # Proteins
    "Beef": "소고기", "Beef heart": "소 심장", "Beef liver": "소 간",
    "Beef trachea": "소 기관지", "Lean beef": "저지방 소고기", "Lean meat": "저지방 고기",
    "Chicken": "닭고기", "Chicken breast": "닭가슴살", "Chicken feet": "닭발",
    "Chicken heart": "닭 심장", "Chicken liver": "닭 간", "Chicken Liver": "닭 간",
    "Chicken sternum cartilage": "닭 연골", "Chicken cartilage": "닭 연골",
    "Chicken fat": "닭 지방", "Boiled chicken": "삶은 닭고기",
    "Chicken Breast (boiled)": "삶은 닭가슴살", "Chicken Breast (pureed)": "으깬 닭가슴살",
    "Chicken breast (small amounts)": "닭가슴살(소량)", "Pureed Chicken": "으깬 닭고기",
    "Soft Cooked Chicken": "부드럽게 익힌 닭고기", "Cooked Chicken": "익힌 닭고기",
    "Turkey": "칠면조", "Lamb": "양고기", "Duck": "오리고기", "Venison": "사슴고기",
    "Pork": "돼지고기", "Pork cartilage": "돼지 연골", "Pork ribs": "돼지 갈비",
    "Salmon": "연어", "Salmon (small amounts)": "연어(소량)", "Salmon (moderate)": "연어(적당량)",
    "Cooked Salmon": "익힌 연어", "Soft Salmon": "부드러운 연어",
    "Sardines": "정어리", "Sardine": "정어리", "Sardine (small)": "정어리(소형)",
    "Sardines with bones": "뼈째 정어리",
    "Mackerel": "고등어", "Herring": "청어", "Tuna": "참치",
    "Cod": "대구", "White fish": "흰살 생선", "White Fish": "흰살 생선",
    "Fish fillet": "생선 필렛", "Oysters": "굴",
    "Green-lipped mussel": "초록입홍합", "Shark cartilage": "상어 연골",
    "Eggs": "계란", "Egg": "계란", "Egg yolk": "계란 노른자",
    "Egg whites": "계란 흰자", "Egg White": "계란 흰자", "Egg white": "계란 흰자",
    "Egg (soft)": "반숙 계란", "Egg (cooked)": "익힌 계란", "Cooked Egg": "익힌 계란",
    "Bone broth": "사골 육수", "Bone meal": "뼈 분말",
    # Vegetables & Fruits
    "Spinach": "시금치", "Broccoli": "브로콜리", "Kale": "케일",
    "Sweet potato": "고구마", "Pumpkin": "호박", "Carrot": "당근",
    "Bell pepper": "피망", "Asparagus": "아스파라거스", "Cabbage": "양배추",
    "Cucumber": "오이", "Corn": "옥수수", "Parsley": "파슬리",
    "Mushrooms": "버섯", "Bok choy": "청경채", "Peas": "완두콩",
    "Seaweed": "해초", "Apple": "사과", "Banana": "바나나",
    "Blueberries": "블루베리", "Turmeric": "강황", "Watermelon": "수박",
    # Grains & Legumes
    "Brown rice": "현미", "Brown Rice": "현미", "White rice": "백미",
    "Barley": "보리", "Oats": "귀리", "Oat": "귀리",
    "Chickpeas": "병아리콩", "Lentils": "렌틸콩", "Kidney beans": "강낭콩",
    # Supplements & Dairy
    "Cod liver oil": "대구간유", "Sunflower seeds": "해바라기씨",
    "Pumpkin seeds": "호박씨", "Brazil nuts": "브라질너트",
    "Flaxseed oil": "아마씨유", "Wheat germ": "밀배아",
    "Plain yogurt": "플레인 요거트", "Yogurt": "요거트",
    "Cottage cheese": "코티지 치즈", "Kefir": "케피어",
    # Cooking styles
    "Stew": "스튜", "Bowl": "보울", "Porridge": "죽", "Mix": "믹스",
    "Plate": "플레이트", "Dinner": "디너", "Patty": "패티",
}

# Functional/branded name prefixes
FUNC_TRANS = {
    "Airway Care": "기도 케어", "Anti-Inflammatory Wellness": "항염 웰니스",
    "Belly Balance": "장 건강", "Blood Builder": "혈액 보강",
    "Blood Support": "혈액 지원", "Brain Boost": "두뇌 활성",
    "Breathe Easy": "호흡 케어", "Bright Eyes": "눈 건강",
    "Calm Mind": "신경 안정", "Cardiac Support": "심장 지원",
    "Cardio Shield": "심장 보호", "Care Blend": "케어 블렌드",
    "Cartilage Care": "연골 케어", "Clear Air": "호흡기 케어",
    "Clear Sight": "시력 케어", "Coat Shine": "모질 윤기",
    "Defense Boost": "면역 강화", "Derma Care": "피부 케어",
    "Digest Ease": "소화 케어", "Digestive Care": "소화 케어",
    "Endo Care": "내분비 케어", "Eye Health": "눈 건강",
    "Eye Protect": "눈 보호", "Flex & Move": "관절 유연",
    "Fur Renewal": "모질 재생", "Gentle Gut": "장 보호",
    "Gentle Kidney": "신장 보호", "Gentle Liver": "간 보호",
    "Gland Guard": "선 보호", "Guard Up": "면역 업",
    "Gut Heal": "장 치유", "Health Guard": "건강 지킴이",
    "Heart Guard": "심장 지킴이", "Heart Health": "심장 건강",
    "Hepato Guard": "간 보호", "Hormone Balance": "호르몬 균형",
    "Immune Boost": "면역 강화", "Immune Shield": "면역 보호",
    "Iron Boost": "철분 보강", "Itch Relief": "가려움 완화",
    "Joint & Bone Support": "관절 뼈 지원", "Joint Shield": "관절 보호",
    "Joint Support": "관절 지원", "Kidney Kind": "신장 케어",
    "Kidney Shield": "신장 보호", "Lean & Fit": "체중 관리",
    "Liver Love": "간 케어", "Liver Shield": "간 보호",
    "Low-P Comfort": "저인산 케어", "Lung Support": "폐 건강",
    "Metabolic Balance": "대사 균형", "Metabolic Fuel": "대사 연료",
    "Metabolism Boost": "대사 촉진", "Muscle Support": "근육 지원",
    "Nerve Care": "신경 케어", "Neuro Shield": "신경 보호",
    "Neuro Support": "신경 지원", "Nourish": "영양 보충",
    "Renal Relief": "신장 완화", "Renal Support": "신장 지원",
    "Retina Guard": "망막 보호", "Seizure Shield": "발작 예방",
    "Skin & Coat Repair": "피부 모질 회복", "Skin Glow": "피부 윤기",
    "Slim Fuel": "다이어트", "Strong Bones": "뼈 강화",
    "Strong Heart": "심장 강화", "Taurine Boost": "타우린 보강",
    "Thyroid Support": "갑상선 지원", "Tummy Calm": "위장 안정",
    "Vision Shield": "시력 보호", "Vital Blood": "혈액 활력",
    "Vitality": "활력", "Weight Watch": "체중 관리",
    "Wellness": "웰니스",
}


def translate_food(name):
    result = name

    # 1. Try functional prefix first (longer first)
    for en, ko in sorted(FUNC_TRANS.items(), key=lambda x: -len(x[0])):
        if result.startswith(en):
            result = result.replace(en, ko, 1)
            break

    # 2. Translate ingredients (longer phrases first)
    for en, ko in sorted(TRANS.items(), key=lambda x: -len(x[0])):
        result = result.replace(en, ko)

    # 3. Clean up English remnants
    result = result.replace(" with ", " ")
    result = result.replace("Lean ", "저지방 ")

    # 4. Natural Korean formatting
    result = result.replace(" & ", " ")  # remove all &
    result = result.replace("보울", "볼")  # 보울 → 볼
    result = result.replace("플레이트", "플레이트")  # keep as is

    # 5. Fix cooking style at wrong position: "스튜 시금치" → "시금치 스튜"
    import re
    # Move cooking style to end if ingredients follow it
    for style in ["스튜", "볼", "죽", "믹스", "플레이트", "디너", "패티"]:
        pattern = rf"({style})\s+([가-힣\s]+?)(\s*#\d+)?$"
        m = re.search(pattern, result)
        if m:
            suffix = m.group(3) or ""
            result = result[:m.start()] + m.group(2).strip() + " " + style + suffix

    # 6. Remove duplicate spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result


def main():
    foods = set()
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            foods.add(row["food"].strip())

    translation = {}
    for food in sorted(foods):
        translation[food] = translate_food(food)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(translation, f, ensure_ascii=False, indent=2)

    print(f"Total: {len(translation)} foods translated")
    print(f"Saved to: {OUT_PATH}")

    # Show samples
    samples = [
        "Beef & Chickpeas Stew", "Bright Eyes Egg yolk Bowl",
        "Brain Boost Salmon Bowl #1", "Bone broth & Spinach Brown Rice Bowl",
        "Belly Balance Lean Beef Mix", "Joint Shield Salmon Bowl #1",
        "Cardiac Support Bowl", "Sardines with Spinach & Sweet potato",
        "Seizure Shield Egg Plate #5", "Chicken breast & Pumpkin with Plain yogurt",
    ]
    print("\nSamples:")
    for s in samples:
        if s in translation:
            print(f"  {s}  =>  {translation[s]}")


if __name__ == "__main__":
    main()
