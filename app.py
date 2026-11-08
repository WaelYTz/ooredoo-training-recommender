"""
Ooredoo Training Recommender -- interactive demo + L&D dashboard.

Two pages (pick in the sidebar):
  - Recommender Demo: pick an employee, see their profile and their real
    top-3 recommendations from Recommend_V4.ipynb's output, with the same
    reason/explanation strings the notebook produces.
  - L&D Dashboard: 6 panels giving an HR-facing view across the whole
    1,059-employee population.

Expects the same directory layout as the notebooks in this project:
    <project_root>/data/processed/employees_final.csv
    <project_root>/data/processed/catalog_full_en_fr_translated.csv
    <project_root>/data/processed/history_cleaned.csv        (optional)
    <project_root>/outputs/v4/recommendations_v4.csv

Run with (from the project root, or from this file's own directory):
    streamlit run app.py

If a file isn't found at its default path, the app offers a manual upload
instead of crashing -- no need to edit paths by hand for a one-off demo.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Ooredoo Training Recommender", layout="wide", page_icon="🎓")

# ---------------------------------------------------------------------------
# 1. Path resolution + flexible loading (falls back to a manual upload
#    instead of crashing if a default path is missing)
# ---------------------------------------------------------------------------

def resolve_base_dir() -> Path:
    base = Path.cwd()
    if (base / "data" / "processed").is_dir():
        return base
    if (base.parent / "data" / "processed").is_dir():
        return base.parent
    return base


BASE = resolve_base_dir()
DATA_PROCESSED = BASE / "data" / "processed"
V4_OUTPUTS = BASE / "outputs" / "v4"

DEFAULT_PATHS = {
    "employees": DATA_PROCESSED / "employees_final.csv",
    "catalog": DATA_PROCESSED / "catalog_full_en_fr_translated.csv",
    "history": DATA_PROCESSED / "history_cleaned.csv",
    "recommendations": V4_OUTPUTS / "recommendations_v4.csv",
}


def get_dataframe(key: str, label: str, required: bool = True):
    default_path = DEFAULT_PATHS[key]
    if default_path.exists():
        return pd.read_csv(default_path).fillna("")
    st.sidebar.warning(f"Couldn't find `{default_path}`.")
    uploaded = st.sidebar.file_uploader(f"Upload {label}", type="csv", key=f"upload_{key}")
    if uploaded is not None:
        return pd.read_csv(uploaded).fillna("")
    if required:
        st.error(
            f"**{label} is required and wasn't found.** Either run this app from the "
            f"project root (so `{default_path}` resolves), or upload it via the sidebar."
        )
        st.stop()
    return None


@st.cache_data(show_spinner="Loading employee data...")
def cached_employees():
    return get_dataframe("employees", "employees_final.csv")


@st.cache_data(show_spinner="Loading course catalog...")
def cached_catalog():
    return get_dataframe("catalog", "catalog_full_en_fr_translated.csv")


@st.cache_data(show_spinner="Loading recommendations (Recommend_V4.ipynb output)...")
def cached_recommendations():
    return get_dataframe(
        "recommendations", "recommendations_v4.csv (from Recommend_V4.ipynb)"
    )


@st.cache_data(show_spinner=False)
def cached_history():
    # Optional -- only used for the language-split dashboard panel.
    return get_dataframe("history", "history_cleaned.csv", required=False)


# ---------------------------------------------------------------------------
# 2. Derived signals reused verbatim from Recommend_V4.ipynb -- cheap, pure
#    pandas, no GPU needed, so safe to recompute live in the demo.
# ---------------------------------------------------------------------------

LANGUAGE_THRESHOLD = 0.6


@st.cache_data(show_spinner=False)
def compute_inferred_language(_employees: pd.DataFrame, _catalog: pd.DataFrame, _history: pd.DataFrame):
    if _history is None or len(_history) == 0:
        return pd.Series("", index=_employees.index)

    id_to_library = dict(zip(_catalog["Course ID"].astype(str), _catalog["Library"]))
    history = _history.copy()
    history["_course_id"] = history["ID de contenu"].astype(str)
    history["_status_norm"] = history["Status"].str.strip().str.lower()

    def derive(group):
        completed = group[group["_status_norm"] == "completed"]
        langs = completed["_course_id"].map(id_to_library).dropna()
        if len(langs) == 0:
            return ""
        counts = Counter(langs)
        top_lang, top_count = counts.most_common(1)[0]
        if top_count / len(langs) > LANGUAGE_THRESHOLD:
            return "EN" if top_lang == "en-US" else "FR" if top_lang == "fr-FR" else ""
        return ""

    signals = history.groupby("email_n").apply(derive).rename("Inferred_Language")
    merged = _employees[["Email"]].merge(signals, left_on="Email", right_index=True, how="left")
    return merged["Inferred_Language"].fillna("")


# ---------------------------------------------------------------------------
# 3. Employee clustering -- structured/categorical features only (no
#    embeddings, no GPU), for the dashboard's "employee clusters" panel.
# ---------------------------------------------------------------------------

LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}


@st.cache_data(show_spinner="Clustering employees...")
def compute_clusters(_employees: pd.DataFrame, n_clusters: int):
    features = _employees[["Bu_new", "Employee_Level", "Skills_Source", "Preferred_Training_Type"]].copy()
    features["Employee_Level_ord"] = features["Employee_Level"].map(LEVEL_ORDER).fillna(1)

    one_hot = pd.get_dummies(features[["Bu_new", "Skills_Source", "Preferred_Training_Type"]])
    X = pd.concat([one_hot, features[["Employee_Level_ord"]]], axis=1).to_numpy(dtype=float)

    k = max(2, min(n_clusters, len(_employees) - 1))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = _employees[["Employee_ID", "Email", "Bu_new", "Employee_Level", "Skills_Source"]].copy()
    out["cluster"] = labels.astype(str)
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    return out


# ---------------------------------------------------------------------------
# 4. Page: Recommender Demo
# ---------------------------------------------------------------------------

def render_recommender_page(employees: pd.DataFrame, catalog: pd.DataFrame, recs: pd.DataFrame):
    st.title("🎓 Training Recommender")
    st.caption(
        "Live view of `Recommend_V4.ipynb`'s output (Qwen3-Embedding-0.6B, "
        "rebalanced field-match/domain-gate) -- not a live re-ranking, this "
        "reads the notebook's precomputed top-3 picks per employee."
    )

    catalog_by_id = catalog.set_index(catalog["Course ID"].astype(str))

    col_filter, col_select = st.columns([1, 2])
    with col_filter:
        departments = ["(all)"] + sorted(employees["Bu_new"].unique().tolist())
        dept_filter = st.selectbox("Filter by department", departments)

    candidates = employees if dept_filter == "(all)" else employees[employees["Bu_new"] == dept_filter]
    candidates = candidates.sort_values(["Firstname", "Lastname"])

    if len(candidates) == 0:
        st.warning("No employees match that filter.")
        return

    with col_select:
        labels = (candidates["Firstname"] + " " + candidates["Lastname"] + "  —  " + candidates["Email"]).tolist()
        picked_label = st.selectbox("Employee", labels)
    picked_email = picked_label.split("  —  ")[-1]
    emp = employees[employees["Email"] == picked_email].iloc[0]

    st.divider()
    profile_col, recs_col = st.columns([1, 2])

    with profile_col:
        st.subheader("Profile")
        st.markdown(f"**{emp['Firstname']} {emp['Lastname']}**")
        st.write(f"**Department:** {emp['Bu_new']}")
        st.write(f"**Role:** {emp['Fonction_EN'] or emp['Fonction']}")
        st.write(f"**Level:** {emp['Employee_Level']}")
        skills_source_label = {
            "real": "real (from completed training)",
            "generated": "generated (simulated)",
            "none": "none on file",
        }.get(emp["Skills_Source"], emp["Skills_Source"])
        st.write(f"**Skills source:** {skills_source_label}")
        st.write(f"**Skills:** {emp['Skills'] or '_(none recorded)_'}")
        st.write(f"**Stated preference:** {emp['Preferred_Formation'] or '_(none stated)_'}")
        st.write(f"**Preferred format:** {emp['Preferred_Training_Type'] or '_(not stated)_'}")

    with recs_col:
        st.subheader("Top-3 recommendations")
        emp_recs = recs[(recs["Email"] == picked_email) & recs["Rank"].notna()].sort_values("Rank")

        if len(emp_recs) == 0:
            st.info("No eligible courses were found for this employee in the saved recommendations.")
        else:
            for _, row in emp_recs.iterrows():
                course_id = str(row["Course_ID"])
                course = catalog_by_id.loc[course_id] if course_id in catalog_by_id.index else None

                flag = "🔴 flagged for review" if row["flagged_for_review"] else "🟢 confident pick"
                with st.container(border=True):
                    st.markdown(f"**#{int(row['Rank'])} — {row['Course_Title']}**  \n{flag}")
                    st.caption(row["explanation"])
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Similarity", f"{row['similarity']:.2f}")
                    metric_cols[1].metric("Rank score", f"{row['rank_score']:.2f}")
                    if course is not None:
                        metric_cols[2].metric("Course level", course["Level"] or "—")
                        with st.expander("Course details"):
                            st.write(f"**Subjects:** {course['Subjects_EN']}")
                            st.write(f"**Skills covered:** {course['Skills_EN']}")
                            st.write(f"**Description:** {course['Description_EN']}")
                    st.caption(f"Boost breakdown: {row['reason'] or '(none — similarity only)'}")


# ---------------------------------------------------------------------------
# 5. Page: L&D Dashboard (6 panels)
# ---------------------------------------------------------------------------

def render_dashboard_page(employees: pd.DataFrame, catalog: pd.DataFrame, recs: pd.DataFrame, history):
    st.title("📊 L&D Dashboard")
    st.caption("Company-wide view across all 1,059 employees, built from `Recommend_V4.ipynb`'s output.")

    top1 = recs[(recs["Rank"] == 1)]
    n_course = catalog[(catalog["Status"].str.lower() == "active") & (catalog["Level"].str.lower() != "general")].shape[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Employees", f"{len(employees):,}")
    m2.metric("Active courses in pool", f"{n_course:,}")
    m3.metric("Distinct courses recommended", f"{top1['Course_Title'].nunique():,}")
    flagged_rate = top1["flagged_for_review"].mean() if len(top1) else float("nan")
    m4.metric("Flagged for review", f"{flagged_rate:.1%}")

    st.divider()
    panel_cols_1 = st.columns(2)
    panel_cols_2 = st.columns(2)
    panel_cols_3 = st.columns(2)

    # --- Panel 1: skill-gap-adjacent -- profile documentation completeness by department ---
    with panel_cols_1[0]:
        st.subheader("1. Skill profile completeness by department")
        st.caption(
            "Proxy for a skill-gap view: what fraction of each department's employees "
            "have a real, generated, or no recorded skill profile -- this is exactly what "
            "the recommender itself has to work with, not a certified competency audit."
        )
        gap = (
            employees.groupby(["Bu_new", "Skills_Source"]).size().reset_index(name="count")
        )
        fig = px.bar(
            gap, x="Bu_new", y="count", color="Skills_Source", barmode="stack",
            category_orders={"Skills_Source": ["real", "generated", "none"]},
            labels={"Bu_new": "Department", "count": "Employees"},
        )
        fig.update_layout(xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Panel 2: recommendation confidence overview ---
    with panel_cols_1[1]:
        st.subheader("2. Recommendation confidence overview")
        st.caption(
            "`flagged_for_review` = none of the pick's positive boosts were a genuine "
            "field/preference/format match -- a proxy for low-confidence picks."
        )
        conf = top1["flagged_for_review"].map({True: "Flagged", False: "Confident"}).value_counts().reset_index()
        conf.columns = ["status", "count"]
        fig = px.pie(conf, names="status", values="count", hole=0.45)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Panel 3: language/delivery preference split ---
    with panel_cols_2[0]:
        st.subheader("3. Language & delivery format split")
        inferred_lang = compute_inferred_language(employees, catalog, history)
        lang_counts = inferred_lang.replace("", "No history / unknown").value_counts().reset_index()
        lang_counts.columns = ["language", "count"]
        type_counts = employees["Preferred_Training_Type"].replace("", "Not stated").value_counts().reset_index()
        type_counts.columns = ["format", "count"]

        sub_cols = st.columns(2)
        with sub_cols[0]:
            fig = px.pie(lang_counts, names="language", values="count", title="Inferred training language")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with sub_cols[1]:
            fig = px.pie(type_counts, names="format", values="count", title="Preferred format")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # --- Panel 4: seniority distribution per department ---
    with panel_cols_2[1]:
        st.subheader("4. Seniority distribution per department")
        seniority = employees.groupby(["Bu_new", "Employee_Level"]).size().reset_index(name="count")
        fig = px.bar(
            seniority, x="Bu_new", y="count", color="Employee_Level", barmode="stack",
            category_orders={"Employee_Level": ["beginner", "intermediate", "advanced"]},
            labels={"Bu_new": "Department", "count": "Employees"},
        )
        fig.update_layout(xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Panel 5: aggregate course demand ---
    with panel_cols_3[0]:
        st.subheader("5. Aggregate course demand (top-1 picks)")
        st.caption(
            "The most-recommended courses company-wide -- worth watching for "
            "concentration/hub effects (see `PROJECT_JOURNAL.md` Phases 26-28)."
        )
        top_courses = top1["Course_Title"].value_counts().head(15).reset_index()
        top_courses.columns = ["course", "times_recommended"]
        fig = px.bar(top_courses.sort_values("times_recommended"), x="times_recommended", y="course", orientation="h")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Panel 6: employee clusters ---
    with panel_cols_3[1]:
        st.subheader("6. Employee clusters")
        st.caption(
            "K-means on structured profile features (department, level, skills source, "
            "preferred format) -- not on embeddings, so this reflects HR-visible "
            "attributes, not semantic/topical similarity."
        )
        n_clusters = st.slider("Number of clusters", min_value=3, max_value=12, value=6, key="n_clusters")
        clustered = compute_clusters(employees, n_clusters)
        fig = px.scatter(
            clustered, x="x", y="y", color="cluster",
            hover_data=["Bu_new", "Employee_Level", "Skills_Source"],
            labels={"x": "PCA component 1", "y": "PCA component 2"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 6. App entry point
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("Ooredoo Training Recommender")
    page = st.sidebar.radio("Page", ["Recommender Demo", "L&D Dashboard"])
    st.sidebar.divider()
    st.sidebar.caption(f"Data root: `{BASE}`")

    employees = cached_employees()
    catalog = cached_catalog()
    recs = cached_recommendations()
    history = cached_history()

    if page == "Recommender Demo":
        render_recommender_page(employees, catalog, recs)
    else:
        render_dashboard_page(employees, catalog, recs, history)


if __name__ == "__main__":
    main()
