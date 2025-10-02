# Obzerra Fraud Detection Platform

## Executive Summary
Obzerra equips Philippine insurers with a fraud triage cockpit that fuses deterministic heuristics, machine-learning ensembles, and investor-ready analytics. The platform offers two production-grade user experiences—an analyst-focused Streamlit workspace and a board-level Dash dashboard—each backed by an automated model training pipeline that refreshes on start-up using proprietary claims data.【F:app.py†L17-L50】【F:dash_app.py†L1-L99】

## System Overview
- **Dual-interface delivery** keeps operations and leadership aligned. The Streamlit app centers on analyst queues, while the Dash experience packages strategic KPIs and portfolio signals without duplicating engineering effort.【F:app.py†L17-L122】【F:dash_app.py†L1-L200】
- **Unified data preparation** standardizes batch uploads, enriches them with engineered features (Benford signals, young-driver flags, amount buckets), and guarantees required schema alignment for both the rules engine and ML ensemble.【F:utils/data_processor.py†L6-L195】
- **Hybrid scoring engine** combines human-readable fraud indicators with calibrated model probabilities, ensuring every claim inherits both explainability and statistical rigor.【F:utils/fraud_engine.py†L13-L194】【F:utils/ml_models.py†L1-L200】

## Feature Highlights
### Adaptive Detection Stack
- **Rules tuned for local fraud patterns.** Weighted heuristics elevate suspicious combinations such as high-value thefts without police reports, very late incidents, and Benford anomalies, giving analysts immediate context for escalations.【F:utils/fraud_engine.py†L15-L162】
- **Ensemble machine learning core.** Logistic regression, random forest, KNN, and optional gradient boosting models are trained with SMOTE/SMOTETomek balancing, cross-validation, and calibration to stay resilient in imbalanced datasets.【F:utils/ml_models.py†L36-L200】

### Explainability & Governance
- **Feature attribution at two levels.** The custom FeatureExplainer distills global importance from the ensemble and produces per-claim contribution narratives, supporting regulatory reviews and audit trails.【F:utils/feature_explainer.py†L1-L188】
- **Distribution-aware guardrails.** Normality testing (Shapiro, K-S, and Q-Q diagnostics) checks when Z-score-driven thresholds remain valid and flags departures that require analyst intervention.【F:utils/statistical_tests.py†L1-L190】

### Operational Readiness
- **Session-managed workflows.** Shared session state caches processed data, model outputs, and explanations so teams can switch tabs without recomputation penalties.【F:app.py†L25-L38】
- **Immediate retraining.** Both interfaces automatically reload and retrain from the latest `assets/insurance_claims_featured_fixed.csv` dataset, ensuring demos reflect up-to-date claims behavior.【F:app.py†L40-L52】【F:dash_app.py†L89-L99】

## Architecture & Data Flow
1. **Data Ingestion:** Batch CSVs enter through the Streamlit uploader, flow through the `DataProcessor` for cleaning, feature engineering, and Benford scoring before populating analyst queues.【F:utils/data_processor.py†L13-L172】
2. **Rule Evaluation:** The FraudEngine applies weighted heuristics and combo multipliers to generate an initial risk score and indicator list, maintaining interpretability for every claim.【F:utils/fraud_engine.py†L15-L134】
3. **Model Inference:** The MLModelManager transforms engineered features, balances the dataset, trains/calibrates the ensemble, and outputs fraud probabilities plus performance diagnostics.【F:utils/ml_models.py†L111-L200】
4. **Score Blending:** Rule-based scores and model probabilities are merged into a final risk score for both single-claim and batch analyses, ensuring consistent prioritization logic across touchpoints.【F:utils/fraud_engine.py†L163-L194】
5. **Explainability Delivery:** FeatureExplainer contributions and narrative summaries feed the UI components to justify each recommendation, while NormalityTester metrics inform monitoring panels.【F:utils/feature_explainer.py†L57-L188】【F:utils/statistical_tests.py†L25-L190】

## Design Logic & Differentiators
- **Human-in-the-loop emphasis:** Weighted heuristics reflect on-the-ground fraud cues so analysts recognize institutional knowledge in the scoring output.【F:utils/fraud_engine.py†L23-L162】
- **Model governance baked in:** Calibration, rebalancing, and statistical validation occur during every training run, reducing compliance risk when deploying on-premise within regulated insurers.【F:utils/ml_models.py†L111-L200】【F:utils/statistical_tests.py†L25-L190】
- **Explainability-first visuals:** UI styling highlights risk badges, SHAP-style badges, and contribution narratives so executives see rationale alongside KPI shifts.【F:app.py†L54-L199】【F:dash_app.py†L100-L200】

## Interface Strategy
- **Streamlit Command Center:** Wide-layout design with KPI cards, claim detail grids, and dynamic risk badging streamlines investigator workflows on desktop monitors.【F:app.py†L54-L199】
- **Dash Executive Dashboard:** A themed Bootstrap experience optimized for presentation screens provides trend analyses, scenario toggles, and investor-grade polish with custom typography and gradients.【F:dash_app.py†L76-L200】

## Prototype Demo Guide
1. **Warm start:** Launch the Streamlit console (`streamlit run app.py`) and allow automatic model retraining to complete (spinner confirms completion).【F:app.py†L40-L52】
2. **Portfolio snapshot:** Highlight the KPI header and alert queue that display combined rule/ML risk scores with color-coded severity badges.【F:app.py†L80-L175】
3. **Claim drill-down:** Open a high-risk claim to show rule indicators, blended scoring, and the explainability narrative generated by the FeatureExplainer.【F:utils/fraud_engine.py†L50-L178】【F:utils/feature_explainer.py†L57-L188】
4. **Data ingestion story:** Demonstrate batch upload prep by referencing how DataProcessor enforces schema and constructs fraud-focused features.【F:utils/data_processor.py†L13-L195】
5. **Executive view:** Switch to the Dash dashboard (`python dash_app.py`) to showcase board-level KPIs and emphasize the shared training pipeline between analyst and executive surfaces.【F:dash_app.py†L76-L200】
6. **Governance talking points:** Close with the statistical guardrails and retraining cadence that mitigate model drift concerns for regulators and investors.【F:utils/statistical_tests.py†L25-L190】【F:app.py†L40-L52】

## Investment Rationale
Obzerra’s modular architecture, automated model lifecycle, and explainability-first presentation position it as a scalable anti-fraud platform that can be localized across ASEAN insurers without heavy vendor lock-in. The dual-interface approach broadens stakeholder adoption, while the hybrid detection stack balances transparency and predictive power for rapid ROI.【F:app.py†L17-L199】【F:utils/ml_models.py†L1-L200】【F:utils/fraud_engine.py†L13-L194】
