"""
Pytest suite for recommender_core.py.

Uses small, hand-built synthetic data throughout -- these are unit tests of
the scoring mechanics themselves (does the math do what it's supposed to),
not a re-run of the real 1,059-employee / 10,765-course pipeline. See
Recommend_V4.ipynb's own smoke tests (built the same way, at real scale)
for that.

Run with: pytest test_recommender_core.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import recommender_core as core

CONFIG_PATH = Path(__file__).parent / "recommender_config.yaml"


@pytest.fixture
def config():
    return core.load_config(CONFIG_PATH)


@pytest.fixture
def tiny_employees():
    return pd.DataFrame({
        "Employee_ID": ["E1", "E2", "E3"],
        "Bu_new": ["Finance", "Network Engineering & Architecture", "Finance"],
        "Employee_Level": ["intermediate", "advanced", "intermediate"],
        "Inferred_Language": ["", "EN", "FR"],
        "Preferred_Training_Type": ["Online", "", "On-site Internal"],
        "Excluded_Course_IDs": [set(), {"C2"}, set()],
    })


@pytest.fixture
def tiny_catalog_pool():
    # _text/_level_norm/_lang are the derived columns V4's own candidate-pool
    # step adds -- reproduced by hand here since this test doesn't run the
    # full pool-derivation pipeline.
    return pd.DataFrame({
        "Course ID": ["C1", "C2", "C3"],
        "Course Title_EN": ["Budgeting Essentials", "Radio Planning Basics", "Leadership 101"],
        "_text": [
            "Budgeting Essentials. Budgeting, Finance. Excel, Budgeting. Learn to budget.",
            "Radio Planning Basics. Network Engineering, Radio Planning. Network Engineering. Learn radio planning.",
            "Leadership 101. Leadership. Leadership. Intro to leadership.",
        ],
        "_level_norm": ["intermediate", "advanced", ""],
        "_lang": ["EN", "EN", "FR"],
    })


# ---------------------------------------------------------------------------
# Field match
# ---------------------------------------------------------------------------

def test_field_tokens_filters_stopwords_and_short_tokens(config):
    # "of" (stopword) and "it" (< min_field_token_len=4) should both be dropped.
    tokens = core.field_tokens("Ministry of IT Affairs", config)
    assert "of" not in tokens
    assert "it" not in tokens
    assert "ministry" in tokens
    assert "affairs" in tokens


def test_compute_field_score_norm_rewards_rare_department_tokens(config, tiny_employees, tiny_catalog_pool):
    field_score = core.compute_field_score_norm(tiny_employees, tiny_catalog_pool, "_text", config)
    assert field_score.shape == (3, 3)
    # E1 (Finance) should score higher on C1 (Budgeting, mentions "finance")
    # than on C2 (Radio Planning) or C3 (Leadership), which share no tokens
    # with "Finance" at all.
    assert field_score[0, 0] > field_score[0, 1]
    assert field_score[0, 0] > field_score[0, 2]
    # E2 (Network Engineering & Architecture) should score highest on C2,
    # which literally contains "Network Engineering" in its text.
    assert field_score[1, 1] == field_score[1].max()
    # Scores must stay within [0, 1] -- the normalization's whole point.
    assert field_score.min() >= 0.0
    assert field_score.max() <= 1.0


# ---------------------------------------------------------------------------
# Preference vocabulary gate
# ---------------------------------------------------------------------------

def test_build_preference_vocab_excludes_blocklist_and_single_words(config):
    catalog = pd.DataFrame({
        "Skills_EN": ["Cash Flow Forecasting, Excel", "Leadership Skills, Budgeting"],
        "Subjects_EN": ["Finance, Corporate Law", ""],
    })
    vocab = core.build_preference_vocab(catalog, config)
    assert "cash flow forecasting" in vocab
    # single-word terms should never enter the vocab (gate requires 2+ words)
    assert "excel" not in vocab
    assert "finance" not in vocab
    # explicitly blocklisted even though it's a real, multi-word catalog phrase
    assert "leadership skills" not in vocab


def test_preference_is_informative_matches_substring(config):
    vocab = {"cash flow forecasting"}
    assert core.preference_is_informative("I want to learn cash flow forecasting please", vocab)
    assert not core.preference_is_informative("open to whatever is recommended", vocab)
    assert not core.preference_is_informative("", vocab)


# ---------------------------------------------------------------------------
# Ranking mechanics
# ---------------------------------------------------------------------------

def test_compute_rank_score_matrix_level_match_and_mismatch(config, tiny_employees, tiny_catalog_pool):
    field_score = np.zeros((3, 3))  # isolate the level signal from field match
    sim_matrix = np.zeros((3, 3))
    pref_sim_matrix = np.zeros((3, 3))
    has_pref = np.array([False, False, False])

    rank_score, _ = core.compute_rank_score_matrix(
        sim_matrix, pref_sim_matrix, field_score, has_pref, tiny_employees, tiny_catalog_pool, config
    )
    # E1 is intermediate; C1 is intermediate (match), C2 is advanced (mismatch).
    assert rank_score[0, 0] > rank_score[0, 1]
    # A mismatch should be an actual penalty relative to a course with no
    # level info at all (C3, blank _level_norm == neutral, not a mismatch).
    assert rank_score[0, 1] < rank_score[0, 2]


def test_compute_rank_score_matrix_generic_scale_reduces_boost_without_topical_signal(config, tiny_employees, tiny_catalog_pool):
    """Regression test for V4's core mechanism (PROJECT_JOURNAL.md Phase 27):
    an employee with zero field/preference signal for a pick should get a
    SMALLER language/level boost than one with strong field signal for an
    otherwise-identical pick -- this is the whole point of generic_boost_floor
    replacing the old flat domain-gate penalty."""
    sim_matrix = np.zeros((2, 1))
    pref_sim_matrix = np.zeros((2, 1))
    has_pref = np.array([False, False])

    employees = pd.DataFrame({
        "Employee_ID": ["E1", "E2"],
        "Bu_new": ["X", "X"],
        "Employee_Level": ["intermediate", "intermediate"],
        "Inferred_Language": ["", ""],
        "Preferred_Training_Type": ["", ""],
        "Excluded_Course_IDs": [set(), set()],
    })
    catalog_pool = pd.DataFrame({
        "Course ID": ["C1"],
        "_level_norm": ["intermediate"],
        "_lang": ["EN"],
    })

    field_score_no_signal = np.array([[0.0], [0.9]])  # E1: no field signal, E2: strong field signal

    rank_score, extras = core.compute_rank_score_matrix(
        sim_matrix, pref_sim_matrix, field_score_no_signal, has_pref, employees, catalog_pool, config
    )
    # Both get the same level-match condition, but E1's generic_scale should
    # be at the floor (0.3 by default) while E2's should be much closer to 1.0.
    assert extras["generic_scale"][0, 0] < extras["generic_scale"][1, 0]
    assert extras["generic_scale"][0, 0] == pytest.approx(config["boosts"]["generic_boost_floor"], abs=1e-6)


def test_compute_rank_score_matrix_excludes_completed_courses(config, tiny_employees, tiny_catalog_pool):
    field_score = np.zeros((3, 3))
    sim_matrix = np.ones((3, 3))  # everything otherwise tied
    pref_sim_matrix = np.zeros((3, 3))
    has_pref = np.array([False, False, False])

    rank_score, _ = core.compute_rank_score_matrix(
        sim_matrix, pref_sim_matrix, field_score, has_pref, tiny_employees, tiny_catalog_pool, config
    )
    # E2's Excluded_Course_IDs = {"C2"} (column index 1) -- must be masked to -inf.
    assert rank_score[1, 1] == -np.inf
    # E2's other columns must NOT be masked.
    assert rank_score[1, 0] != -np.inf
    assert rank_score[1, 2] != -np.inf


def test_select_top_k_returns_correct_order_and_count():
    rank_score = np.array([
        [0.1, 0.9, 0.5, -np.inf],
        [0.3, 0.2, 0.1, 0.4],
    ])
    top_idx = core.select_top_k(rank_score, k=2)
    assert top_idx.shape == (2, 2)
    # Row 0's best two, best-first: column 1 (0.9), then column 2 (0.5).
    assert list(top_idx[0]) == [1, 2]
    # Row 1's best two, best-first: column 3 (0.4), then column 0 (0.3).
    assert list(top_idx[1]) == [3, 0]


def test_select_top_k_never_returns_a_fully_excluded_column_ahead_of_an_eligible_one():
    # A column of -inf must never outrank a real score, however negative.
    rank_score = np.array([[-np.inf, -5.0, -0.01]])
    top_idx = core.select_top_k(rank_score, k=3)
    assert list(top_idx[0]) == [2, 1, 0]


# ---------------------------------------------------------------------------
# Training format
# ---------------------------------------------------------------------------

def test_training_type_without_catalog_column_defaults_every_course_to_online(config, tiny_employees, tiny_catalog_pool):
    """Today's catalog has no Training_Type column, so every course must count
    as Online: the boost fires on every course for the "Online" employee and
    on none for the others -- identical to the behaviour before the column
    was supported."""
    zeros = np.zeros((3, 3))
    _, extras = core.compute_rank_score_matrix(
        zeros, zeros, zeros, np.array([False, False, False]), tiny_employees, tiny_catalog_pool, config
    )
    assert extras["type_match"][0].all()      # E1: Online
    assert not extras["type_match"][1].any()  # E2: no preference
    assert not extras["type_match"][2].any()  # E3: On-site Internal


def test_training_type_boost_only_fires_for_courses_in_the_preferred_format(config, tiny_employees):
    # Same level and language on every course, so training format is the only
    # thing separating them. C3's blank Training_Type falls back to the
    # default (Online); C2's odd casing/whitespace must still match.
    catalog_pool = pd.DataFrame({
        "Course ID": ["C1", "C2", "C3"],
        "Course Title_EN": ["Excel Online", "Excel Workshop", "Excel Basics"],
        "_level_norm": ["intermediate"] * 3,
        "_lang": ["EN"] * 3,
        "Training_Type": ["Online", " on-site internal ", ""],
    })
    zeros = np.zeros((3, 3))
    has_pref = np.array([False, False, False])
    rank_score, extras = core.compute_rank_score_matrix(
        zeros, zeros, zeros, has_pref, tiny_employees, catalog_pool, config
    )
    # E1 prefers Online: C1 and C3 (default) match, C2 doesn't.
    assert list(extras["type_match"][0]) == [True, False, True]
    # E2 has no preference: nothing matches.
    assert not extras["type_match"][1].any()
    # E3 prefers On-site Internal: only C2 matches, and it outranks C1.
    assert list(extras["type_match"][2]) == [False, True, False]
    assert rank_score[2, 1] > rank_score[2, 0]

    pick = core.describe_pick(
        2, 1, sim_matrix=zeros, pref_sim_matrix=zeros, rank_score_matrix_masked=rank_score,
        has_pref=has_pref, extras=extras, employees=tiny_employees, catalog_pool=catalog_pool, config=config,
    )
    assert "training type match" in pick["reason"]


# ---------------------------------------------------------------------------
# pandas 3.x regression test
# ---------------------------------------------------------------------------

def test_to_plain_array_handles_string_dtype_without_broadcast_error():
    """Regression test for the real bug a synthetic smoke test caught while
    building Recommend_Model_Comparison.ipynb: on pandas 3.x, a plain
    `.values` call on a text column returns a StringArray that raises
    'Lengths of operands do not match' under [:, None] == [None, :]
    broadcasting. to_plain_array must prevent that."""
    series = pd.Series(["EN", "FR", ""])
    arr = core.to_plain_array(series)
    # The comparison this function exists to make safe:
    result = arr[:, None] == arr[None, :]
    assert result.shape == (3, 3)
    assert result[0, 0]  # "EN" == "EN"
    assert not result[0, 1]  # "EN" != "FR"


# ---------------------------------------------------------------------------
# Explanation building
# ---------------------------------------------------------------------------

def test_describe_pick_flags_picks_with_no_strong_signal(config, tiny_catalog_pool):
    # A dedicated employee with NO training-type preference either -- the
    # shared tiny_employees fixture's E1 has Preferred_Training_Type=="Online",
    # which fires a type_match boost on every course here (tiny_catalog_pool
    # has no Training_Type column, so every course defaults to Online), and
    # type_match IS a STRONG_KINDS member. That's a real,
    # deliberate finding in its own right (PROJECT_JOURNAL.md Phase 24: the
    # training-type boost is a confirmed no-op for *ranking*, since it's
    # identical for every course, but it still counts as "strong" for
    # flagged_for_review purposes) -- worth a dedicated employee here so this
    # test isolates the "genuinely nothing fired" case cleanly.
    employees = pd.DataFrame({
        "Employee_ID": ["E1"],
        "Bu_new": ["Finance"],
        "Employee_Level": ["intermediate"],
        "Inferred_Language": [""],
        "Preferred_Training_Type": [""],
        "Excluded_Course_IDs": [set()],
    })

    field_score = np.zeros((1, 3))
    sim_matrix = np.zeros((1, 3))
    pref_sim_matrix = np.zeros((1, 3))
    has_pref = np.array([False])

    rank_score, extras = core.compute_rank_score_matrix(
        sim_matrix, pref_sim_matrix, field_score, has_pref, employees, tiny_catalog_pool, config
    )
    # E1 x C3 ("Leadership 101"): no field match, no preference, no training-type
    # preference, C3 has no level at all (neutral), and C3's language is FR
    # while E1 has no inferred language -- so lang_default (which requires the
    # course to be EN) doesn't fire either. Nothing should fire at all, so
    # this pick should be flagged.
    pick = core.describe_pick(
        0, 2, sim_matrix=sim_matrix, pref_sim_matrix=pref_sim_matrix,
        rank_score_matrix_masked=rank_score, has_pref=has_pref, extras=extras,
        employees=employees, catalog_pool=tiny_catalog_pool, config=config,
    )
    assert pick["flagged_for_review"] is True
    assert pick["reason"] == ""
    assert pick["Course_Title"] == "Leadership 101"


def test_describe_pick_does_not_flag_a_genuine_field_match(config, tiny_employees, tiny_catalog_pool):
    field_score = np.zeros((3, 3))
    field_score[0, 0] = 0.8  # E1 x C1: a strong, genuine field match
    sim_matrix = np.zeros((3, 3))
    pref_sim_matrix = np.zeros((3, 3))
    has_pref = np.array([False, False, False])

    rank_score, extras = core.compute_rank_score_matrix(
        sim_matrix, pref_sim_matrix, field_score, has_pref, tiny_employees, tiny_catalog_pool, config
    )
    pick = core.describe_pick(
        0, 0, sim_matrix=sim_matrix, pref_sim_matrix=pref_sim_matrix,
        rank_score_matrix_masked=rank_score, has_pref=has_pref, extras=extras,
        employees=tiny_employees, catalog_pool=tiny_catalog_pool, config=config,
    )
    assert pick["flagged_for_review"] is False
    assert "field match" in pick["reason"]
