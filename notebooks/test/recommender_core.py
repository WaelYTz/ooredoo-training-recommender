"""
Core, testable scoring logic for the Ooredoo training recommender.

Extracted from Recommend_V4.ipynb so it can be:
  1. unit-tested (see test_recommender_core.py) without needing the full
     1,059-employee / 10,765-course real dataset or a GPU, and
  2. imported by both the notebook and any future script (the Streamlit
     demo currently reads precomputed recommendations rather than
     re-scoring live, but could call into this module directly instead).

Every function here is pure: it takes its config and data explicitly as
arguments rather than reading module-level/notebook-global variables, which
is what makes it independently testable. `recommender_config.yaml` holds
every tunable constant -- see that file's comments for what each one means
and why its current value was chosen (or, in most cases, not yet tuned).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

KIND_SENTENCES = {
    "field": "it matches your field",
    "pref_strong": "it matches what you said you want to learn",
    "pref_weak": "it's related to what you said you want to learn",
    "lang_hist": "it's in the language you've mostly trained in",
    "lang_default": "it's offered in English by default",
    "level_match": "it fits your seniority level",
    "level_mismatch": "it's a different level than yours",
    "type_match": "it matches your preferred training format",
}
STRONG_KINDS = {"field", "pref_strong", "pref_weak", "type_match"}


def load_config(path: str | Path = "recommender_config.yaml") -> dict[str, Any]:
    """Load the YAML config. Raises FileNotFoundError with a clear message
    if the path doesn't exist -- fail loudly rather than silently falling
    back to hardcoded defaults, since a missing config file is exactly the
    kind of mistake that should surface immediately, not three cells later
    as a confusing NameError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Every boost weight and tunable "
            f"constant lives there now -- see recommender_config.yaml in the "
            f"project root."
        )
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# pandas 3.x compatibility
# ---------------------------------------------------------------------------

def to_plain_array(series: pd.Series) -> np.ndarray:
    """Force a plain numpy object array out of a pandas Series.

    Needed because pandas 3.x defaults to a "future string" dtype for
    object-like columns, whose `.values` returns a `StringArray` (a pandas
    ExtensionArray) rather than a plain ndarray. `StringArray` supports
    `[:, None]` reshaping but NOT full elementwise broadcasting against
    another differently-shaped StringArray (e.g. `(n,1) == (1,m)`), which
    the language/level matching below relies on. On pandas <2.x this is a
    no-op; on pandas 3.x it's required, or these comparisons raise
    `ValueError: Lengths of operands do not match`. This was a real bug
    found via a synthetic smoke test while building Recommend_Model_Comparison.ipynb
    -- see test_to_plain_array_handles_string_dtype below for the
    regression test.
    """
    return series.astype(object).to_numpy()


# ---------------------------------------------------------------------------
# Field match (TF-IDF-weighted department/course text overlap)
# ---------------------------------------------------------------------------

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def field_tokens(bu_new: str, config: dict) -> list[str]:
    fm = config["field_match"]
    stopwords = set(fm["field_stopwords"])
    min_len = fm["min_field_token_len"]
    raw = re.split(r"[^a-zA-Z]+", strip_accents(bu_new).lower())
    return [t for t in raw if len(t) >= min_len and t not in stopwords]


def compute_field_score_norm(
    employees: pd.DataFrame, catalog_pool: pd.DataFrame, course_text_col: str, config: dict
) -> np.ndarray:
    """Returns an (n_employees, n_courses) matrix in [0, 1]: how much of an
    employee's department-name vocabulary (rarity-weighted like TF-IDF)
    appears in each course's text."""
    unique_bu = employees["Bu_new"].unique()
    bu_tokens = {bu: set(field_tokens(bu, config)) for bu in unique_bu}
    all_tokens = sorted(set().union(*bu_tokens.values())) if len(bu_tokens) else []
    token_index = {t: i for i, t in enumerate(all_tokens)}

    n_course = len(catalog_pool)
    course_text_norm = catalog_pool[course_text_col].map(lambda t: strip_accents(t).lower())
    course_token_matrix = np.zeros((n_course, len(all_tokens)), dtype=bool)
    for tok, col in token_index.items():
        pattern = re.compile(r"\b" + re.escape(tok) + r"\b")
        course_token_matrix[:, col] = course_text_norm.map(lambda t: bool(pattern.search(t))).values

    doc_freq = course_token_matrix.sum(axis=0)
    token_idf = np.log((n_course + 1) / (doc_freq + 1)) + 1

    n_emp = len(employees)
    emp_token_weights = np.zeros((n_emp, len(all_tokens)), dtype=np.float32)
    for i, bu in enumerate(to_plain_array(employees["Bu_new"])):
        for tok in bu_tokens[bu]:
            emp_token_weights[i, token_index[tok]] = token_idf[token_index[tok]]

    raw_field_score = emp_token_weights @ course_token_matrix.astype(np.float32).T
    emp_max_score = emp_token_weights.sum(axis=1, keepdims=True)
    emp_max_score_safe = np.where(emp_max_score == 0, 1.0, emp_max_score)
    return np.clip(raw_field_score / emp_max_score_safe, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Preference vocabulary gate
# ---------------------------------------------------------------------------

def build_preference_vocab(catalog: pd.DataFrame, config: dict) -> set[str]:
    blocklist = set(config["preference_gate"]["vocab_blocklist"])
    phrases = set()
    for col in ["Skills_EN", "Subjects_EN"]:
        for val in catalog[col].dropna():
            for term in str(val).split(","):
                term = term.strip().lower()
                if len(term.split()) >= 2 and term not in blocklist:
                    phrases.add(term)
    return phrases


def preference_is_informative(text: str, vocab: set[str]) -> bool:
    cleaned = str(text).strip().lower()
    if cleaned == "":
        return False
    return any(phrase in cleaned for phrase in vocab)


# ---------------------------------------------------------------------------
# Training format
# ---------------------------------------------------------------------------

def course_training_types(catalog_pool: pd.DataFrame, config: dict) -> pd.Series:
    """Each course's delivery format, lowercased so it compares cleanly
    against Preferred_Training_Type. Read from the catalog's optional
    `Training_Type` column (Online / On-site Internal / On-site External);
    rows with no value -- or the whole catalog, if the column doesn't exist,
    as with today's LinkedIn-Learning-only catalog -- fall back to
    `training_type.default_course_type` in the config."""
    default = config["training_type"]["default_course_type"]
    if "Training_Type" in catalog_pool.columns:
        types = catalog_pool["Training_Type"].fillna("").astype(str).str.strip()
    else:
        types = pd.Series("", index=catalog_pool.index)
    return types.where(types != "", default).str.lower()


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def compute_rank_score_matrix(
    sim_matrix: np.ndarray,
    pref_sim_matrix: np.ndarray,
    field_score_norm: np.ndarray,
    has_pref: np.ndarray,
    employees: pd.DataFrame,
    catalog_pool: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """V4's rebalanced scoring function -- see recommender_config.yaml's
    `generic_boost_floor` comment and PROJECT_JOURNAL.md Phase 27 for the
    mechanism this implements. Returns (rank_score_matrix_masked, extras)
    where `extras` holds the intermediate per-boost arrays `describe_pick`
    needs to build a human-readable explanation."""
    w = config["boosts"]
    n_emp, n_course = sim_matrix.shape

    row_min = pref_sim_matrix.min(axis=1, keepdims=True)
    row_max = pref_sim_matrix.max(axis=1, keepdims=True)
    row_range_safe = np.where((row_max - row_min) == 0, 1.0, row_max - row_min)
    pref_norm = (pref_sim_matrix - row_min) / row_range_safe
    pref_contribution_raw = np.where(has_pref[:, None], pref_norm, 0.0)

    topical_signal = np.clip(field_score_norm + pref_contribution_raw, 0.0, 1.0)
    generic_scale = w["generic_boost_floor"] + (1.0 - w["generic_boost_floor"]) * topical_signal

    emp_lang = to_plain_array(employees["Inferred_Language"])
    course_lang = to_plain_array(catalog_pool["_lang"])
    has_lang_pref = (emp_lang != "")[:, None]
    lang_match = has_lang_pref & (emp_lang[:, None] == course_lang[None, :])
    lang_default = (~has_lang_pref) & (course_lang[None, :] == "EN")
    lang_boost_raw = np.where(
        lang_match, w["language_match_boost"], np.where(lang_default, w["language_default_en_boost"], 0.0)
    )
    lang_boost = lang_boost_raw * generic_scale

    emp_level = to_plain_array(employees["Employee_Level"])
    course_level = to_plain_array(catalog_pool["_level_norm"])
    course_has_level = (course_level != "")[None, :]
    level_match = course_has_level & (course_level[None, :] == emp_level[:, None])
    level_mismatch = course_has_level & (course_level[None, :] != emp_level[:, None])
    level_boost = (
        np.where(level_match, w["level_match_boost"] * generic_scale, 0.0)
        + np.where(level_mismatch, -w["level_mismatch_penalty"], 0.0)
    )

    pref_type = to_plain_array(employees["Preferred_Training_Type"].astype(str).str.strip().str.lower())
    course_type = to_plain_array(course_training_types(catalog_pool, config))
    type_match = (pref_type != "")[:, None] & (pref_type[:, None] == course_type[None, :])
    training_boost = np.where(type_match, w["training_type_boost"], 0.0) * generic_scale

    field_boost = w["field_boost"] * field_score_norm
    pref_contribution = w["preference_boost"] * pref_contribution_raw

    rank_score_matrix = sim_matrix + field_boost + pref_contribution + lang_boost + level_boost + training_boost

    course_id_index = {str(cid): idx for idx, cid in enumerate(catalog_pool["Course ID"])}
    exclusion_mask = np.zeros((n_emp, n_course), dtype=bool)
    for i, excluded in enumerate(employees["Excluded_Course_IDs"]):
        for cid in excluded:
            col = course_id_index.get(cid)
            if col is not None:
                exclusion_mask[i, col] = True

    extras = {
        "pref_norm": pref_norm, "lang_match": lang_match, "lang_default": lang_default,
        "level_match": level_match, "level_mismatch": level_mismatch, "type_match": type_match,
        "field_boost": field_boost, "pref_contribution": pref_contribution,
        "generic_scale": generic_scale,
    }
    return np.where(exclusion_mask, -np.inf, rank_score_matrix), extras


def select_top_k(rank_score_matrix_masked: np.ndarray, k: int) -> np.ndarray:
    """Returns an (n_employees, k) matrix of column indices, sorted
    best-first per row. Separated from the scoring function itself so the
    selection/ordering logic can be tested independently of the boost
    arithmetic."""
    n_emp, n_course = rank_score_matrix_masked.shape
    k = min(k, n_course)
    top_idx_unsorted = np.argpartition(-rank_score_matrix_masked, k - 1, axis=1)[:, :k]
    row_indices = np.arange(n_emp)[:, None]
    top_scores_unsorted = rank_score_matrix_masked[row_indices, top_idx_unsorted]
    order = np.argsort(-top_scores_unsorted, axis=1)
    return np.take_along_axis(top_idx_unsorted, order, axis=1)


def sentence_for(items: list[tuple[str, str, float]]) -> str:
    phrases = [KIND_SENTENCES[k] for k, _, _ in items]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def describe_pick(
    i: int, j: int, *, sim_matrix, pref_sim_matrix, rank_score_matrix_masked,
    has_pref, extras, employees, catalog_pool, config, title_col: str = "Course Title_EN",
) -> dict:
    """Builds the reason/explanation/flagged_for_review dict for one
    (employee row i, course column j) pick -- identical mechanism to
    Recommend_V4.ipynb's describe_pick, parameterized instead of closing
    over notebook globals."""
    w = config["boosts"]
    course = catalog_pool.iloc[j]
    boosts: list[tuple[str, str, float]] = []

    if extras["field_boost"][i, j] > 0:
        boosts.append(("field", "field match", float(extras["field_boost"][i, j])))

    if has_pref[i]:
        norm_val = float(extras["pref_norm"][i, j])
        contribution = float(w["preference_boost"] * extras["pref_norm"][i, j])
        raw_sim = float(pref_sim_matrix[i, j])
        if norm_val > 0.7:
            boosts.append(("pref_strong", f"matches stated interest ({raw_sim:.2f})", contribution))
        elif norm_val > 0.3:
            boosts.append(("pref_weak", f"related to stated interest ({raw_sim:.2f})", contribution))

    if extras["lang_match"][i, j]:
        boosts.append(("lang_hist", "language match (from history)", w["language_match_boost"] * extras["generic_scale"][i, j]))
    elif extras["lang_default"][i, j]:
        boosts.append(("lang_default", "default English", w["language_default_en_boost"] * extras["generic_scale"][i, j]))

    if extras["level_match"][i, j]:
        boosts.append(("level_match", "level match", w["level_match_boost"] * extras["generic_scale"][i, j]))
    elif extras["level_mismatch"][i, j]:
        boosts.append(("level_mismatch", "level mismatch", -w["level_mismatch_penalty"]))

    if extras["type_match"][i, j]:
        boosts.append(("type_match", "training type match", w["training_type_boost"] * extras["generic_scale"][i, j]))

    sorted_boosts = sorted(boosts, key=lambda x: -abs(x[2]))
    numeric_reason = ", ".join(f"{l} ({v:+.2f})" for _, l, v in sorted_boosts)

    positive = [x for x in sorted_boosts if x[2] > 0]
    negative = [x for x in sorted_boosts if x[2] < 0]
    pos_sentence = sentence_for(positive[:3])
    neg_sentence = sentence_for(negative)
    natural = f"Recommended because {pos_sentence}." if pos_sentence else "Recommended based on overall profile similarity."
    if neg_sentence:
        natural += f" Note: {neg_sentence}."

    flagged = not any(k_ in STRONG_KINDS for k_, _, _ in positive)

    return {
        "Course_ID": course["Course ID"], "Course_Title": course[title_col],
        "similarity": float(sim_matrix[i, j]), "rank_score": float(rank_score_matrix_masked[i, j]),
        "reason": numeric_reason, "explanation": natural, "flagged_for_review": flagged,
    }
