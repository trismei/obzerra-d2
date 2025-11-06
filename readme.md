# Obzerra Fraud Detection Platform

A production-ready prototype for insurance claims analysts to triage suspicious activity using a blend of heuristic rules and machine-learning models. The platform ships with both a Streamlit command center and a Dash executive dashboard so operations, data science, and compliance teams can work from a shared dataset while focusing on the workflows that matter to them.

## Table of Contents
- [Product Overview](#product-overview)
- [Core Capabilities](#core-capabilities)
- [Architecture](#architecture)
- [Data Assets](#data-assets)
- [User Guide](#user-guide)
- [Local Development Guide](#local-development-guide)
  - [Using UV (recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
  - [Opening in VS Code](#opening-in-vs-code)
- [Running the Applications](#running-the-applications)
- [Testing & Quality Checks](#testing--quality-checks)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Briefing Materials](#briefing-materials)

## Product Overview
Obzerra focuses on the fraud-prevention needs of Philippine insurers that must process high volumes of motor and property claims under strict turnaround times. The system enables:

- Rapid ingestion of batched claim files with automatic schema validation.
- Dual scoring via deterministic fraud rules and gradient-boosted ensemble models.
- Explainable insights that expose which policyholder, incident, or financial features drove the decision.
- Historical logging so analysts can audit previous decisions and export evidence packets.

The application is designed to run offline or within restricted networks, making it suitable for on-premise deployments inside regulated institutions.

Recent enhancements concentrate on richer explainability and analytics guardrails: the Streamlit console and Dash dashboard now surface the same blended rule/ML score that powers the claim triage queue, while the new feature importance explainer and statistical normality tests provide context for spikes in suspicious activity.【F:app.py†L19-L47】【F:utils/fraud_engine.py†L15-L110】【F:utils/feature_explainer.py†L1-L124】【F:utils/statistical_tests.py†L1-L88】

## Core Capabilities
- **Portfolio Command Center (Streamlit)** – KPI snapshots, alert queues, and analyst-friendly review tools optimized for widescreen desktops.
- **Executive Dashboard (Dash)** – C-suite view with macro KPIs, trend visualizations, and segment filters for strategic planning.
- **Batch Upload Pipeline** – Drag-and-drop CSV import, automated cleansing, feature engineering, and downloadable risk summaries.
- **Single Claim Scoring** – Guided form for manual case entry with immediate probability output and recommended actions.
- **Explainability Layer** – Combines rule hits, SHAP-style feature importances, and natural-language rationales for each claim. The new `FeatureExplainer` module extracts global and local drivers directly from the ensemble models so analysts can defend decisions in audits.【F:utils/feature_explainer.py†L1-L124】
- **Session Persistence** – Uses a lightweight session manager to cache uploaded data, processed features, and model outputs across tabs.
- **Quantitative Guardrails** – Built-in normality testing validates when Z-score outlier logic is appropriate and recommends fallbacks whenever the incoming data distribution shifts.【F:utils/statistical_tests.py†L1-L88】

## Architecture
```
obzerra-d2/
├── app.py                  # Streamlit command center
├── dash_app.py             # Dash dashboard for executives
├── utils/
│   ├── data_processor.py   # Cleaning, feature engineering, schema validation
│   ├── explanations.py     # SHAP + narrative generation utilities
│   ├── fraud_engine.py     # Rule-based heuristics and score aggregation
│   ├── ml_models.py        # Training, persistence, and inference pipeline
│   ├── feature_explainer.py  # Custom feature importance summarizer for the ensemble
│   ├── session_manager.py  # Simple in-memory caching for user interactions
│   └── statistical_tests.py  # Normality testing utilities that inform anomaly scoring
├── assets/
│   ├── insurance_claims_cleaned_fixed.csv
│   ├── insurance_claims_featured_fixed.csv
│   ├── sample_data.csv
│   └── data_dictionary_engineered.csv
├── pyproject.toml          # Project metadata and dependency definition
├── uv.lock                 # Resolved dependency graph for repeatable installs
└── readme.md               # This guide
```

Both `app.py` and `dash_app.py` automatically train the bundled models on startup by reading `assets/insurance_claims_featured_fixed.csv`. Replace this file with your own engineered dataset to adapt the solution to a different insurance line.

## Data Assets
| File | Purpose | Notes |
| --- | --- | --- |
| `insurance_claims_featured_fixed.csv` | Primary training corpus with engineered features. | Loaded during application startup for model refreshes. |
| `insurance_claims_cleaned_fixed.csv` | Cleaned but un-engineered claims used for demonstrations. | Helpful for validating the preprocessing pipeline. |
| `data_dictionary_engineered.csv` | Column-level documentation for the engineered dataset. | Surface this to analysts when onboarding. |
| `sample_data.csv` | Lightweight sample for UI smoke tests. | Safe to share publicly; contains anonymized values. |

All assets are stored locally in `assets/` to avoid any dependency on third-party storage providers.

## User Guide
The [Obzerra User Guide](docs/user-guide.md) consolidates onboarding instructions for partner companies. It includes the standard CSV template (with required fields highlighted), upload checklists, and tips so every organization can prepare data the same way before using the batch analysis workflow.

## Local Development Guide
### Using UV (recommended)
1. [Install UV](https://docs.astral.sh/uv/getting-started/installation/) if it is not already on your machine.
2. Create the environment and install dependencies:
   ```bash
   uv sync
   ```
3. Activate the virtual environment (UV prints the exact command for your shell).

### Using pip
1. Ensure Python 3.11 or newer is available.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install the project in editable mode:
   ```bash
   pip install -e .
   ```

### Opening in VS Code
1. Launch VS Code and choose **File → Open Folder…** to select the repository root.
2. Install the **Python**, **Pylance**, and **Black Formatter** extensions for linting and autocompletion.
3. When prompted, select the virtual environment created above (`.venv` or the UV environment) as the interpreter.
4. Add a `.vscode/launch.json` entry if you want one-click debugging of `app.py` or `dash_app.py`.

## Running the Applications
After activating your environment, choose the interface that fits your workflow:

- **Streamlit operations console**
  ```bash
  streamlit run app.py
  ```
  The UI becomes available at `http://localhost:8501`.

- **Dash executive dashboard**
  ```bash
  python dash_app.py
  ```
  Access it at `http://127.0.0.1:5000`.

Both scripts will log progress while retraining the embedded LightGBM/XGBoost ensemble. Expect the first run to take a few seconds depending on hardware.

## Testing & Quality Checks
While no automated suites are bundled yet, the following manual checks keep the system healthy:

- **Data validation** – Load `sample_data.csv` through the batch upload flow and confirm the derived metrics match expectations.
- **Model sanity** – Replace the training dataset with a known edge-case sample to verify rule overrides and SHAP explanations fire correctly.
- **Performance** – Monitor terminal logs for processing durations; anything above a few seconds per 1,000 rows warrants profiling the `DataProcessor`.

Consider wiring in pytest or great-expectations for regression checks as you productionize the solution.

## Operational Notes
- The ML models are trained in-memory on startup and are not persisted to disk; schedule warm-up runs before analyst shifts if cold-start latency is a concern.
- All styling assets are implemented via inline CSS and Dash/Streamlit theming, so branding updates can be made centrally inside each app file.
- No vendor-specific services are required—everything runs with local Python dependencies for easier on-premise deployment.

## Troubleshooting
| Symptom | Suggested Fix |
| --- | --- |
| Models fail to train on startup | Confirm `assets/insurance_claims_featured_fixed.csv` exists and that the Python process has read permission. |
| CSV uploads rejected | Validate column names against `data_dictionary_engineered.csv`; update `DataProcessor.REQUIRED_COLUMNS` if you extend the schema. |
| UI loads without charts | Ensure Plotly is installed in the active environment and that no corporate proxy is blocking websocket connections. |
| Slow SHAP explanations | Reduce the number of background samples in `ExplanationEngine.generate_shap_values()` or precompute explanations offline. |

For additional questions, contact the Obzerra engineering team or open an issue in your internal tracker.

## Briefing Materials
- [Panel & Investor Brief](docs/panel-and-investor-brief.md) – Slide-ready narrative covering product thesis, architecture, and demo guidance for stakeholder presentations.
