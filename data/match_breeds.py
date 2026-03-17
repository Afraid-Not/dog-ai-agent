# -*- coding: utf-8 -*-
"""cidd_breed_disorders.json 견종명과 Annotation 폴더 견종명 매칭 후 CSV 저장"""
import json
import csv
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
JSON_PATH = DATA_DIR / "cidd_breed_disorders.json"
ANNOTATION_DIR = DATA_DIR / "Annotation"
OUTPUT_CSV = DATA_DIR / "breed_match.csv"


def normalize_for_match(name: str) -> str:
    """매칭용 정규화: 소문자, 공백/하이픈을 언더스코어로."""
    if not name:
        return ""
    s = name.lower().strip()
    for c in " -/,":
        s = s.replace(c, "_")
    # 연속 언더스코어 하나로
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def main():
    # 1. JSON에서 견종 목록 로드
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    cidd_breeds = [b["breed"] for b in data["breeds"]]
    cidd_normalized = {normalize_for_match(b): b for b in cidd_breeds}

    # 2. Annotation 폴더에서 견종(폴더명) 목록
    annotation_folders = []
    if ANNOTATION_DIR.exists():
        for item in ANNOTATION_DIR.iterdir():
            if item.is_dir() and item.name.startswith("n") and "-" in item.name:
                # n02116738-African_hunting_dog -> African_hunting_dog
                breed_part = item.name.split("-", 1)[1]
                annotation_folders.append((item.name, breed_part))
    annotation_normalized = {normalize_for_match(p): (folder, p) for folder, p in annotation_folders}

    # 3. 매칭: Annotation 기준으로 CIDD와 매칭 (완전 일치 후 부분 일치)
    rows = []
    matched_cidd = set()

    def find_cidd_match(norm_ann: str):
        if norm_ann in cidd_normalized:
            return cidd_normalized[norm_ann], "yes"
        # 부분 일치: 접두/접미 (airedale <-> airedale_terrier, malamute <-> alaskan_malamute)
        for norm_cidd, cidd_breed in cidd_normalized.items():
            if (
                norm_cidd.startswith(norm_ann + "_")
                or norm_cidd.endswith("_" + norm_ann)
                or norm_ann.startswith(norm_cidd + "_")
                or norm_ann.endswith("_" + norm_cidd)
            ):
                return cidd_breed, "partial"
        return None, "no"

    for norm_ann, (folder, ann_display) in sorted(annotation_normalized.items()):
        cidd_breed, match_status = find_cidd_match(norm_ann)
        if cidd_breed:
            matched_cidd.add(cidd_breed)
            rows.append({
                "annotation_folder": folder,
                "annotation_breed": ann_display,
                "cidd_breed": cidd_breed,
                "matched": match_status,
            })
        else:
            rows.append({
                "annotation_folder": folder,
                "annotation_breed": ann_display,
                "cidd_breed": "",
                "matched": "no",
            })

    # 4. CIDD에만 있는 견종 (Annotation에 없음)
    for cidd in sorted(cidd_breeds):
        if cidd not in matched_cidd:
            rows.append({
                "annotation_folder": "",
                "annotation_breed": "",
                "cidd_breed": cidd,
                "matched": "cidd_only",
            })

    # 5. CSV 저장
    fieldnames = ["annotation_folder", "annotation_breed", "cidd_breed", "matched"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_matched = sum(1 for r in rows if r["matched"] == "yes")
    n_ann_only = sum(1 for r in rows if r["matched"] == "no")
    n_cidd_only = sum(1 for r in rows if r["matched"] == "cidd_only")
    print(f"저장 완료: {OUTPUT_CSV}")
    print(f"  매칭됨(둘 다): {n_matched}, Annotation만: {n_ann_only}, CIDD만: {n_cidd_only}")


if __name__ == "__main__":
    main()
