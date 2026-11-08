# Data (not included)

The real inputs are company data (the org file and the learning-activity export) and a
licensed LinkedIn Learning catalog export, so none of them is in this repository. To run the
pipeline, put your own files here with the layout and columns below. Save every file as
**UTF-8 CSV** (in Excel: *CSV UTF-8*).

```
data/
├── raw/
│   └── catalog_full_en_fr.csv                  catalog export (input of scripts/translate_catalog.py)
└── processed/
    ├── catalog_full_en_fr_translated.csv       catalog + English columns (output of translate_catalog.py)
    ├── employees_final.csv                     one row per employee
    └── history_cleaned.csv                     LinkedIn Learning course activity
```

## `catalog_full_en_fr.csv`

The LinkedIn Learning catalog export, English and French libraries, active and retired courses.

| Column | Used for |
|---|---|
| `Course ID` | joins with the history's `ID de contenu` |
| `Course Title`, `Description`, `Skills`, `Subjects`, `Categories` | course text (translated to the `_EN` columns) |
| `Status` | only `active` courses are recommended |
| `Level` | seniority match; `general` courses are excluded (onboarding-style, not personal) |
| `Library` | course language (`en-US` / `fr-FR`) |
| `Training_Type` *(optional)* | `Online`, `On-site Internal` or `On-site External`; blank or missing means `Online` |

`scripts/translate_catalog.py` adds `Course Title_EN`, `Description_EN`, `Skills_EN`,
`Subjects_EN` and `Categories_EN` and keeps the French originals.

## `history_cleaned.csv`

Start from the LinkedIn Learning activity export and keep only rows whose content type is
**Course**. Three columns are needed; the others are ignored.

| Column | Content |
|---|---|
| `email_n` | employee email, lowercase |
| `ID de contenu` | LinkedIn course ID as a whole number (`483224`, not `483224.0`) |
| `Status` | from `Pourcentage de terminé`: 75% or more is `completed`, 1–74% is `in-progress`, 0% is `dropped` |

## `employees_final.csv`

| Column | Required | Content |
|---|---|---|
| `Employee_ID` | yes | unique ID |
| `Email` | yes | lowercase, unique, same spelling as `email_n` in the history |
| `Firstname`, `Lastname` | no | shown in the demo app |
| `Bu_new`, `Direction_new` | yes | business unit and direction, in English |
| `Fonction`, `Fonction_EN` | yes | job title, original and English |
| `Field_EN` | yes | `Bu_new. Direction_new. Fonction_EN. Role focus: <role in English>` |
| `Employee_Level` | yes | `beginner`, `intermediate` or `advanced` (see the table below) |
| `Preferred_Formation` | no | what the person wants to learn, free text in English |
| `Preferred_Training_Type` | no | `Online`, `On-site Internal` or `On-site External` |
| `Skills`, `Skills_Source` | filled by script | run `python build_skills_from_history.py` |

`build_skills_from_history.py` fills `Skills` with the skills tagged on the courses each person
**completed**. People with no completed course get blank skills; nothing is generated or guessed.

### `Fonction` → `Employee_Level`

Titles not listed default to `intermediate`. "Senior" individual contributors are
`intermediate`, not `advanced`: seniority is not management.

| Level | Fonction values |
|---|---|
| `advanced` | Chef De Service, Directeur Adjoint, Expert, Superviseur, Responsable, Directeur Opérationnel, Chef De Produit, Directeur, Chef D'Équipe, Lead, Area Manager, Chief, Chef d'équipe Commercial, Channel Manager, Chef De Marché, Office Manager, Directeur Général |
| `intermediate` | Spécialiste Sénior, Spécialiste, Ingénieur Senior, Ingénieur, Conseiller, Coordinateur, Juriste Sénior, Animateur, Juriste, BUSINESS PARTNER, Web Conseiller, Certified Trainer, Senior Designer, Infirmier, Infirmière, Advisor, Gestionnaire |
| `beginner` | Chauffeur, Assistante, Agent, Storekeeper |
