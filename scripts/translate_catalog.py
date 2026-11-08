"""Create the final English catalog from the original bilingual catalog.

This is the only script needed for the complete workflow:
- reads catalog_full_en_fr.csv
- translates French titles, descriptions, skills, subjects, and categories
- wraps short fragments before translation to reduce hallucinations
- translates comma-separated terms individually
- applies corrections found during the full v3 audit
- writes catalog_full_en_fr_translated_final.csv

Requirements:
    pip install pandas transformers sentencepiece torch
"""

import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "fra_Latn"
TGT_LANG = "eng_Latn"
INPUT_FILE = "catalog_full_en_fr.csv"
OUTPUT_FILE = "catalog_full_en_fr_translated.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / INPUT_FILE
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / OUTPUT_FILE

FRAGMENT_TEMPLATES = {
    "title": ("Titre du cours : {}", "course title:"),
    "term": ("Terme technique : {}", "technical term:"),
}

print("Loading NLLB model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")


def strip_prefix(text: str, expected_prefix: str) -> str:
    value = str(text).strip()
    lower = value.lower()
    if lower.startswith(expected_prefix):
        return value[len(expected_prefix):].strip()
    if ":" in value[:40]:
        return value.split(":", 1)[1].strip()
    return value


def is_bad_fragment_translation(text: str) -> bool:
    value = str(text).strip().lower()
    if len(value) < 2:
        return True
    suspicious = (
        "europa.eu",
        "official journal of the european union",
        "having regard to",
        "whereas",
        "this regulation shall apply",
        "technical term:",
        "course title:",
    )
    return any(phrase in value for phrase in suspicious)


def translate_fragments(fragments: list[str], template_key: str, batch_size: int = 32) -> dict[str, str]:
    fr_template, en_prefix = FRAGMENT_TEMPLATES[template_key]
    translations = {}
    for start in range(0, len(fragments), batch_size):
        batch = fragments[start:start + batch_size]
        wrapped = [fr_template.format(fragment) for fragment in batch]
        inputs = tokenizer(
            wrapped,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_length=200,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for original, output in zip(batch, decoded):
            translated = strip_prefix(output, en_prefix)
            translations[original] = (
                original if is_bad_fragment_translation(translated) else translated
            )
        if (start // batch_size) % 10 == 0:
            print(f"  [{template_key}] {start + len(batch)} / {len(fragments)}")
    return translations


def translate_descriptions(texts: list[str], batch_size: int = 8) -> dict[str, str]:
    translations = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        min_length = max(10, int(input_lengths.float().mean().item() * 0.7))
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_length=512,
                min_length=min_length,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.update(zip(batch, decoded))
        if (start // batch_size) % 10 == 0:
            print(f"  [description] {start + len(batch)} / {len(texts)}")
    return translations


def build_term_set(series: pd.Series) -> set[str]:
    terms = set()
    for value in series.dropna():
        for term in str(value).split(","):
            term = term.strip()
            if term:
                terms.add(term)
    return terms


def apply_fragment_corrections(text: str) -> str:
    value = str(text).strip()
    replacements = (
        (r"\bcritical spirit\b", "critical thinking"),
        (r"\bmembership\b", "belonging"),
        (r"\bdaily, salary, course\b", "day-to-day work, salary, and career path"),
        (r"\bIT media\b", "IT support"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(r"^The title of the course is\s+", "", value, flags=re.IGNORECASE)
    return value.strip()


def apply_title_correction(source: str, translated: str) -> str:
    exact = {
        "Joomla! : Optimisation du référencement": "Joomla!: SEO optimization",
        "L’essentiel de Pages 7": "The basics of Pages 7",
        "L'essentiel de Tableau": "The basics of Tableau",
        "Acrobat 9 : Les nouveautés": "Acrobat 9: New features",
    }
    return exact.get(str(source), translated)


def map_comma_list(value, term_map: dict[str, str]) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    terms = [term.strip() for term in str(value).split(",") if term.strip()]
    return ", ".join(apply_fragment_corrections(term_map.get(term, term)) for term in terms)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    catalog = pd.read_csv(INPUT_PATH)
    required = {"Library", "Course Title", "Description", "Skills", "Subjects", "Categories"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    french_mask = catalog["Library"].eq("fr-FR")
    french = catalog.loc[french_mask]
    print(f"French rows: {len(french)}")

    unique_titles = sorted(french["Course Title"].dropna().unique().tolist())
    title_map = translate_fragments(unique_titles, "title")

    unique_terms = sorted(
        build_term_set(french["Skills"])
        | build_term_set(french["Subjects"])
        | build_term_set(french["Categories"])
    )
    term_map = translate_fragments(unique_terms, "term")

    unique_descriptions = sorted(french["Description"].dropna().unique().tolist())
    description_map = translate_descriptions(unique_descriptions)

    for column in ["Course Title", "Description", "Skills", "Subjects", "Categories"]:
        target = f"{column}_EN"
        catalog[target] = catalog[column]
        catalog.loc[~french_mask, target] = catalog.loc[~french_mask, column]

    for index in french.index:
        source_title = catalog.at[index, "Course Title"]
        translated_title = title_map.get(source_title, source_title)
        translated_title = apply_fragment_corrections(translated_title)
        catalog.at[index, "Course Title_EN"] = apply_title_correction(
            source_title, translated_title
        )
        catalog.at[index, "Skills_EN"] = map_comma_list(catalog.at[index, "Skills"], term_map)
        catalog.at[index, "Subjects_EN"] = map_comma_list(catalog.at[index, "Subjects"], term_map)
        catalog.at[index, "Categories_EN"] = map_comma_list(catalog.at[index, "Categories"], term_map)
        description = catalog.at[index, "Description"]
        catalog.at[index, "Description_EN"] = (
            description_map.get(description, "") if pd.notna(description) else ""
        )

    known_bad_description_id = "193427"
    correction = (
        "Follow the production phases of a brochure using InDesign, Photoshop, "
        "Illustrator, Bridge, and Acrobat. Your trainer Pierre Ruiz will guide you "
        "through the various steps using the example of a theater program."
    )
    catalog.loc[catalog["Course ID"].astype(str).eq(known_bad_description_id), "Description_EN"] = correction

    catalog.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")

    print("Validation:")
    print(f"  rows: {len(catalog)}")
    print(f"  French rows: {int(french_mask.sum())}")
    for source, target in [
        ("Course Title", "Course Title_EN"),
        ("Skills", "Skills_EN"),
        ("Subjects", "Subjects_EN"),
        ("Categories", "Categories_EN"),
        ("Description", "Description_EN"),
    ]:
        source_values = catalog.loc[french_mask, source].fillna("").astype(str).str.strip()
        target_values = catalog.loc[french_mask, target].fillna("").astype(str).str.strip()
        missing_count = int((source_values.ne("") & target_values.eq("")).sum())
        print(f"  {target} missing for non-empty source: {missing_count}")


if __name__ == "__main__":
    main()
