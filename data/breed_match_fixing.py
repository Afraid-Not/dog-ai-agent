# -*- coding: utf-8 -*-
"""견종 매칭 수정: 일부라도 맞으면 yes, Annotation만 있는 경우 CIDD에서 최대 유사 종으로 매칭"""
import json
import csv
from pathlib import Path
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).resolve().parent
JSON_PATH = DATA_DIR / "cidd_breed_disorders.json"
ANNOTATION_DIR = DATA_DIR / "Annotation"
OUTPUT_CSV = DATA_DIR / "breed_match_fixed.csv"


def normalize_for_match(name: str) -> str:
    """매칭용 정규화: 소문자, 공백/하이픈을 언더스코어로."""
    if not name:
        return ""
    s = name.lower().strip()
    for c in " -/,":
        s = s.replace(c, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def similarity(a: str, b: str) -> float:
    """0~1 유사도. 정규화된 문자열 기준."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    cidd_breeds = [b["breed"] for b in data["breeds"]]
    cidd_normalized = {normalize_for_match(b): b for b in cidd_breeds}

    annotation_folders = []
    if ANNOTATION_DIR.exists():
        for item in ANNOTATION_DIR.iterdir():
            if item.is_dir() and item.name.startswith("n") and "-" in item.name:
                breed_part = item.name.split("-", 1)[1]
                annotation_folders.append((item.name, breed_part))
    annotation_normalized = {normalize_for_match(p): (folder, p) for folder, p in annotation_folders}

    rows = []
    matched_cidd = set()

    def find_cidd_match(norm_ann: str):
        if norm_ann in cidd_normalized:
            return cidd_normalized[norm_ann], "yes"
        for norm_cidd, cidd_breed in cidd_normalized.items():
            if (
                norm_cidd.startswith(norm_ann + "_")
                or norm_cidd.endswith("_" + norm_ann)
                or norm_ann.startswith(norm_cidd + "_")
                or norm_ann.endswith("_" + norm_cidd)
            ):
                return cidd_breed, "yes"  # 일부라도 맞으면 yes
        return None, "no"

    # 1) 매칭된 것 (yes만 사용)
    no_match_rows = []
    for norm_ann, (folder, ann_display) in sorted(annotation_normalized.items()):
        cidd_breed, match_status = find_cidd_match(norm_ann)
        if cidd_breed:
            matched_cidd.add(cidd_breed)
            rows.append({
                "annotation_folder": folder,
                "annotation_breed": ann_display,
                "cidd_breed": cidd_breed,
                "matched": "yes",
            })
        else:
            no_match_rows.append({
                "annotation_folder": folder,
                "annotation_breed": ann_display,
                "norm_ann": norm_ann,
            })

    # 2) CIDD에만 있는 풀 (아직 매칭 안 된 것)
    cidd_only_pool = [c for c in cidd_breeds if c not in matched_cidd]

    # 3) Annotation만 있는 경우 → CIDD 풀에서 최대 유사 종으로 매칭
    for r in no_match_rows:
        norm_ann = r["norm_ann"]
        best_cidd = None
        best_score = -1.0
        for cidd in cidd_only_pool:
            norm_cidd = normalize_for_match(cidd)
            score = similarity(norm_ann, norm_cidd)
            if score > best_score:
                best_score = score
                best_cidd = cidd
        if best_cidd is not None:
            matched_cidd.add(best_cidd)
            cidd_only_pool.remove(best_cidd)
            rows.append({
                "annotation_folder": r["annotation_folder"],
                "annotation_breed": r["annotation_breed"],
                "cidd_breed": best_cidd,
                "matched": "similar",
            })
        else:
            rows.append({
                "annotation_folder": r["annotation_folder"],
                "annotation_breed": r["annotation_breed"],
                "cidd_breed": "",
                "matched": "no",
            })

    # 4) 남은 CIDD만 있는 견종
    for cidd in sorted(cidd_breeds):
        if cidd not in matched_cidd:
            rows.append({
                "annotation_folder": "",
                "annotation_breed": "",
                "cidd_breed": cidd,
                "matched": "cidd_only",
            })

    fieldnames = ["annotation_folder", "annotation_breed", "cidd_breed", "matched"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_yes = sum(1 for r in rows if r["matched"] == "yes")
    n_similar = sum(1 for r in rows if r["matched"] == "similar")
    n_no = sum(1 for r in rows if r["matched"] == "no")
    n_cidd_only = sum(1 for r in rows if r["matched"] == "cidd_only")
    print(f"저장 완료: {OUTPUT_CSV}")
    print(f"  yes(완전/부분): {n_yes}, similar(유사매칭): {n_similar}, no: {n_no}, cidd_only: {n_cidd_only}")


if __name__ == "__main__":
    main()
