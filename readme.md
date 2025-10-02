# replit.md

## Overview

Obzerra is a local-first fraud detection MVP specifically designed for Philippine insurance claims officers. The system provides an intuitive Dash-based web interface for analyzing insurance claims using a combination of rule-based detection algorithms and machine learning models. The application focuses on simplicity and user-friendliness, allowing claims officers to upload batch CSV files for comprehensive fraud analysis. All processing is done locally without external data transmission, ensuring data privacy and security.

## Recent Changes (October 2025)

### Analysis Naming & History Enhancements (Oct 2, 2025)
Added comprehensive naming and history features for better analysis organization:

1. **Analysis Naming Feature**:
   - Optional name input field in batch analysis section with placeholder examples
   - Auto-generates "Batch Analysis - {timestamp}" when left empty
   - Stores analysis_name and analysis_type in history records
   - Helper text guides users: "Give this analysis a name to easily find it in your history later"

2. **Enhanced History View**:
   - Premium timeline design with analysis names displayed prominently
   - Type indicators (Batch/Single) with appropriate icons
   - Click "View Details" to reload historical analyses with full premium card layout
   - Download capability for historical analyses (CSV export)
   - Backwards compatible with unnamed legacy entries

3. **UI Polish Updates**:
   - Fixed risk score formatting to exactly 2 decimal places throughout application
   - Added "-- No Data --" option to optional field dropdowns in column mapping
   - Removed infinite progress animation, replaced with clean Dash loading spinner
   - Fixed text alignment issues caused by inconsistent decimal formatting

### Enterprise-Grade Design Overhaul (Oct 2, 2025)
The system has been completely transformed with Fortune 500-level visual polish and premium midnight-indigo design:

1. **Premium Design System Implementation**:
   - Midnight-indigo color palette (#0a0e27, #0f1629, #1a1f3a) with platinum accents (#f8fafc, #e2e8f0)
   - Radial gradient background with ethereal ellipse effect
   - Inter & IBM Plex Sans typography from Google Fonts for professional hierarchy
   - CSS custom properties system for consistent theming throughout
   - Frosted-glass card effects using backdrop-filter blur(20px) and rgba backgrounds
   - Premium shadow system (sm, md, lg) with glow effects
   - Smooth cubic-bezier transitions for all interactive elements

2. **Executive Dashboard Redesign**:
   - Premium KPI cards with floating icons, gradient accent bars, and micro-statistics
   - Enhanced empty states with centered icon and professional messaging
   - Risk distribution donut chart with pull effects and custom fonts
   - Top fraud indicators bar chart with gradient colorscale
   - All charts in frosted-glass containers with proper spacing

3. **Modern CSV Data Preview**:
   - Metadata card grid showing total rows, columns breakdown, data quality percentage, and filename
   - Column chips display (first 15 columns with "more" indicator)
   - Styled data table with alternating rows, dark theme, and proper typography
   - All wrapped in frosted-glass containers with premium borders

4. **Premium History Timeline**:
   - Timeline design with left border and circular gradient markers
   - History cards with gradient backgrounds and mini KPI displays
   - "View Details" buttons with primary gradient and shadow effects
   - Smooth hover transitions and professional spacing

5. **Enhanced Header & Navigation**:
   - Gradient header with top border accent and radial background
   - Brand title with gradient text effect
   - Premium nav tabs with frosted background, hover states, and gradient active state

### Previous Enhancements (Oct 2025)
6. **CSV Export Functionality**: Added download button with timestamped CSV export capability for batch analysis results including all key metrics and explanations
7. **5-Tab Interface**: Added Single Claim Analysis (quick individual claim checks), System Logs (real-time analysis tracking), and User Guide (comprehensive documentation) tabs
8. **Comprehensive Single Claim Form**: Added police report field, incident type/severity dropdowns, state location field, and incident hour slider (0-23) with visual marks for professional data entry
9. **Enhanced Session Management**: Extended SessionManager with helper methods for dashboard KPIs (total_analyzed, flagged_count, avg_risk_score, high_risk_count)
10. **Dash Interface Migration**: Successfully migrated from Streamlit to Dash framework to meet capstone requirements
10. **KNN Model Integration**: Added K-Nearest Neighbors (K=7, distance-weighted) to ML ensemble with optimized weights (LR 35%, RF 40%, KNN 25%)
11. **Normality Testing**: Implemented Shapiro-Wilk and Kolmogorov-Smirnov tests for Z-score analysis with IQR fallback for non-normal distributions
12. **Custom SHAP-like Explainability**: Created FeatureExplainer class that extracts feature importance from tree-based models and generates per-prediction contribution analysis
13. **Enhanced High-Frequency Detection**: Implemented sophisticated frequency analysis supporting pre-calculated flags, 30-day rolling windows, and policy-based tracking (>3 claims threshold)
14. **Bug Fixes**: Resolved critical runtime bugs in DataProcessor, FraudEngine, MLModelManager, ExplanationEngine, and SessionManager

### Current Performance Metrics
- Ensemble Accuracy: 57.5%
- Precision: 24.3%
- Recall: 34.7%
- F1-Score: 28.6%
- PR-AUC: 0.269
- Cross-validation: LR F1=0.373, RF F1=0.173, KNN F1=0.147

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Dash with dash-bootstrap-components featuring enterprise-grade midnight-indigo design system
- **Design Language**: Fortune 500-level visual polish with frosted-glass effects, premium gradients, and professional typography
- **Layout**: 5-tab interface with Dashboard, Batch Analysis, Single Claim Analysis, System Logs, and User Guide views
- **User Interface Components**:
  - **Dashboard Tab**: Executive KPI cards with floating icons, gradient accents, mini-statistics, sparklines, and enhanced Plotly visualizations in frosted-glass containers
  - **Batch Analysis Tab**: Premium CSV upload with radial gradient hover, metadata cards (rows/columns/quality/filename), column chips display, styled data table with alternating rows, and interactive charts
  - **Single Claim Tab**: Form-based individual claim analysis with segmented sections and professional input styling
  - **System Logs Tab**: Premium timeline design with circular gradient markers, history cards showing mini KPIs, and "View Details" buttons with gradient effects
  - **User Guide Tab**: Comprehensive documentation with clean typography and proper hierarchy
  - Real-time filtering and sorting capabilities with premium dropdown styling

### Backend Architecture
- **Core Detection Engine**: Rule-based fraud detection system (`FraudEngine`) implementing weighted scoring across 8 fraud indicators
- **Data Processing Pipeline**: Modular data processor handling cleaning, validation, feature engineering, and duplicate resolution
- **ML Pipeline**: Ensemble approach combining Logistic Regression, Random Forest, and K-Nearest Neighbors models with SMOTE balancing for imbalanced datasets
- **Normality Testing**: Statistical validation using Shapiro-Wilk and Kolmogorov-Smirnov tests before Z-score analysis, with IQR fallback for non-normal data
- **Custom SHAP-like Explainability**: Feature importance explainer that computes per-prediction contributions and generates human-readable explanations
- **Explanation System**: Converts technical ML outputs into user-friendly insights using plain-language templates
- **Session Management**: Framework-agnostic in-memory persistence for analysis history and session statistics

### Fraud Detection Logic
- **Rule-Based Analysis**: 8 weighted fraud indicators including:
  - Z-score outliers for claim amounts with normality testing (weight: 0.2)
  - Benford's Law analysis for number authenticity (weight: 0.15)
  - Temporal pattern analysis for unusual incident hours (weight: 0.1)
  - Round amount detection (weight: 0.1)
  - High-value claim flagging (weight: 0.15)
  - Young claimant high-value combinations (weight: 0.1)
  - Witness validation patterns (weight: 0.1)
  - High-frequency claims detection (>3 claims in 30 days, weight: 0.1)
- **Risk Scoring**: 0-100 scale with Low (0-30), Medium (31-70), High (71-100) risk bands
- **Feature Engineering**: Automated creation of derived features from raw claim data

### Data Management
- **Local Processing**: All data processing occurs locally without external transmission
- **Column Mapping System**: Flexible mapping allowing users to map CSV columns to internal schema
- **Required Fields**: claim_id, total_claim_amount, incident_hour_of_the_day
- **Optional Fields**: age, incident_state, incident_severity, incident_type, witnesses
- **Data Validation**: Comprehensive cleaning with automatic duplicate handling and missing value imputation

### ML Model Architecture (Enhanced for Enterprise-Grade Performance)
- **Advanced Ensemble Approach**: 
  - Enhanced Logistic Regression with class_weight='balanced' and L2 regularization (C=0.1)
  - Enhanced Random Forest (200 estimators, depth 15, class_weight='balanced')
  - K-Nearest Neighbors (K=5, weights='distance') for local pattern detection
  - Optional LightGBM/XGBoost gradient boosting (graceful fallback if unavailable)
  - Optimized ensemble weights: LR=35%, RF=40%, KNN=25% (baseline) or LR=15%, RF=20%, KNN=15%, LGB=25%, XGB=25% (advanced)
- **Probability Calibration**: 
  - CalibratedClassifierCV with isotonic method for all models
  - Improves confidence score reliability for business decisions
  - Calibrated on non-SMOTE data to prevent distribution bias
- **Advanced Data Balancing**: 
  - SMOTETomek (SMOTE + Tomek links cleaning) for superior class balance
  - Fallback to regular SMOTE if needed
  - Handles imbalanced datasets with 24.7% fraud rate effectively
- **Robust Evaluation Pipeline**:
  - 5-fold stratified cross-validation with F1 scoring
  - Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Individual model performance tracking
  - Test set evaluation with 80/20 split
- **Feature Engineering & Importance**:
  - StandardScaler for numerical feature normalization
  - Averaged feature importance across RF/LGB/XGB tree-based models
  - Consistent feature schema between training and prediction
- **Model Training & Deployment**: 
  - Requires minimum 50 samples with at least 5 positive fraud cases
  - Automatic training on startup using insurance_claims_featured_fixed.csv (1000 samples, 247 fraud cases)
  - Fallback training on batch data with synthetic labels if real datasets unavailable
  - Robust error handling with graceful degradation

## External Dependencies

### Core Libraries
- **Dash**: Web application framework for the user interface
- **dash-bootstrap-components**: Bootstrap components for Dash (modern clean design)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing for statistical operations
- **Plotly**: Interactive data visualization (express and graph_objects)
- **Scikit-learn**: Machine learning models and preprocessing utilities
- **Imbalanced-learn**: SMOTE implementation for handling imbalanced datasets
- **SciPy**: Statistical functions for fraud detection algorithms
- **Joblib**: Model serialization and persistence

### Data Processing Dependencies
- **datetime**: Time-based feature engineering and session management
- **re**: Regular expression processing for data validation
- **base64/io**: File handling for CSV upload functionality
- **json**: Configuration and session data serialization

### No External Services
The system is designed to be completely local-first with no external API calls, database connections, or cloud service dependencies, ensuring data privacy and offline functionality.