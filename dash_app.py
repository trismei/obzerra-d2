import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import io
import json
from utils.data_processor import DataProcessor
from utils.fraud_engine import FraudEngine
from utils.ml_models import MLModelManager
from utils.explanations import ExplanationEngine
from utils.session_manager import SessionManager

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Obzerra - Fraud Detection"
)

data_processor = DataProcessor()
fraud_engine = FraudEngine()
ml_manager = MLModelManager()
explanation_engine = ExplanationEngine()
session_manager = SessionManager()

try:
    import os
    training_file = 'assets/insurance_claims_featured_fixed.csv'
    if os.path.exists(training_file):
        print('🤖 Training ML models...')
        success = ml_manager.train_from_real_data(training_file)
        if success:
            print('✅ Models trained successfully!')
except Exception as e:
    print(f"Training failed: {str(e)}")

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #5b7bff 0%, #8f5fff 100%);
                --primary: #6366f1;
                --primary-light: #818cf8;
                --primary-dark: #4f46e5;
                --accent: #8b5cf6;
                --accent-light: #a78bfa;
                --success: #10b981;
                --success-light: #34d399;
                --warning: #f59e0b;
                --warning-light: #fbbf24;
                --danger: #ef4444;
                --danger-light: #f87171;
                --platinum: #e5e7eb;
                --platinum-dark: #d1d5db;
                --bg-midnight: #0a0e27;
                --bg-deep-navy: #0f1629;
                --bg-navy: #1a1f3a;
                --bg-card: #1e2a47;
                --bg-card-light: #2a3654;
                --bg-frosted: rgba(30, 42, 71, 0.7);
                --text-platinum: #f8fafc;
                --text-silver: #e2e8f0;
                --text-muted: #94a3b8;
                --text-dim: #64748b;
                --border-subtle: rgba(148, 163, 184, 0.1);
                --border-medium: rgba(148, 163, 184, 0.2);
                --border-strong: rgba(148, 163, 184, 0.3);
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
                --shadow-glow: 0 0 40px rgba(91, 123, 255, 0.15);
                
                /* Legacy variable aliases for backward compatibility */
                --bg-dark: #0f1629;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --border: rgba(148, 163, 184, 0.2);
            }
            
            * {
                box-sizing: border-box;
            }
            
            body {
                background: radial-gradient(ellipse at top, #1a1f3a 0%, #0a0e27 50%, #000000 100%);
                background-attachment: fixed;
                color: var(--text-platinum);
                font-family: 'IBM Plex Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 15px;
                line-height: 1.6;
                letter-spacing: -0.01em;
                min-height: 100vh;
            }
            
            .header-section {
                background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%);
                padding: 2.5rem 3rem;
                margin-bottom: 3rem;
                border-bottom: 1px solid var(--border-medium);
                box-shadow: var(--shadow-md), var(--shadow-glow);
                backdrop-filter: blur(20px);
                position: relative;
            }
            
            .header-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--primary-gradient);
            }
            
            .brand-title {
                font-family: 'Inter', sans-serif;
                font-size: 2.75rem;
                font-weight: 800;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 1rem;
                letter-spacing: -0.03em;
            }
            
            .brand-subtitle {
                color: var(--text-muted);
                font-size: 1.05rem;
                margin-top: 0.75rem;
                font-weight: 400;
                letter-spacing: 0.01em;
            }
            
            .nav-tabs {
                border-bottom: 1px solid var(--border-medium);
                margin-bottom: 2.5rem;
                background: var(--bg-frosted);
                padding: 0.5rem 1rem;
                border-radius: 12px;
                backdrop-filter: blur(10px);
            }
            
            .nav-tabs .nav-link {
                color: var(--text-muted);
                border: none;
                background: transparent;
                padding: 0.875rem 1.5rem;
                font-weight: 500;
                font-size: 0.95rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border-radius: 8px;
                margin: 0 0.25rem;
            }
            
            .nav-tabs .nav-link:hover {
                color: var(--primary-light);
                background: rgba(99, 102, 241, 0.1);
                transform: translateY(-1px);
            }
            
            .nav-tabs .nav-link.active {
                color: var(--text-platinum);
                background: var(--primary-gradient);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            }
            
            .card-modern {
                background: var(--bg-frosted);
                backdrop-filter: blur(20px);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: var(--shadow-md);
                border: 1px solid var(--border-medium);
                margin-bottom: 2rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .kpi-card {
                background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%);
                border-radius: 16px;
                padding: 2rem;
                border: 1px solid var(--border-medium);
                box-shadow: var(--shadow-md);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .kpi-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: var(--primary-gradient);
            }
            
            .kpi-card:hover {
                transform: translateY(-4px);
                box-shadow: var(--shadow-lg);
                border-color: var(--border-strong);
            }
            
            .kpi-label {
                color: var(--text-muted);
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 0.75rem;
            }
            
            .kpi-value {
                font-family: 'Inter', sans-serif;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.5rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                line-height: 1;
            }
            
            .btn-primary-custom {
                background: var(--primary-gradient);
                border: none;
                padding: 1rem 2.5rem;
                border-radius: 12px;
                font-weight: 600;
                font-size: 0.95rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                letter-spacing: 0.02em;
            }
            
            .btn-primary-custom:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
            }
            
            .risk-low {
                background: linear-gradient(135deg, var(--success), #059669);
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.875rem;
                font-weight: 600;
                display: inline-block;
            }
            
            .risk-medium {
                background: linear-gradient(135deg, var(--warning), #d97706);
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.875rem;
                font-weight: 600;
                display: inline-block;
            }
            
            .risk-high {
                background: linear-gradient(135deg, var(--danger), #dc2626);
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.875rem;
                font-weight: 600;
                display: inline-block;
            }
            
            .section-title {
                font-family: 'Inter', sans-serif;
                font-size: 1.75rem;
                font-weight: 700;
                color: var(--text-platinum);
                margin-bottom: 2rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                letter-spacing: -0.02em;
            }
            
            .input-field {
                background: var(--bg-deep-navy);
                border: 1px solid var(--border-medium);
                border-radius: 10px;
                color: var(--text-platinum);
                padding: 0.875rem 1rem;
                font-size: 0.95rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .input-field:focus {
                border-color: var(--primary);
                outline: none;
                box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
                background: var(--bg-navy);
            }
            
            .upload-zone {
                background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%);
                border: 2px dashed var(--border-medium);
                border-radius: 16px;
                padding: 4rem 2rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .upload-zone::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 300px;
                height: 300px;
                background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
                transform: translate(-50%, -50%);
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .upload-zone:hover::before {
                opacity: 1;
            }
            
            .upload-zone:hover {
                border-color: var(--primary);
                border-style: solid;
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }
            
            .log-entry {
                background: var(--bg-dark);
                border-left: 3px solid var(--primary);
                padding: 1rem;
                margin-bottom: 0.75rem;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 0.875rem;
            }
            
            .guide-section {
                background: var(--bg-card);
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 4px solid var(--primary);
            }
            
            .guide-section h4 {
                color: var(--primary);
                margin-bottom: 1rem;
            }
            
            table {
                background: var(--bg-card) !important;
            }
            
            .dash-table-container .dash-spreadsheet-container {
                background: var(--bg-card);
            }
            
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background: var(--bg-card) !important;
                color: var(--text-primary) !important;
                border-color: var(--border) !important;
            }
            
            /* Dropdown fixes - proper dark theme styling */
            .Select-control {
                background-color: var(--bg-dark) !important;
                border: 1px solid var(--border) !important;
                border-radius: 8px !important;
            }
            
            .Select-control:hover {
                border-color: var(--primary) !important;
            }
            
            .Select-placeholder,
            .Select--single > .Select-control .Select-value {
                color: var(--text-primary) !important;
            }
            
            .Select-input > input {
                color: var(--text-primary) !important;
            }
            
            .Select-menu-outer {
                background-color: var(--bg-dark) !important;
                border: 1px solid var(--border) !important;
                border-radius: 8px !important;
                margin-top: 4px !important;
            }
            
            .Select-menu {
                background-color: var(--bg-dark) !important;
            }
            
            .Select-option {
                background-color: var(--bg-dark) !important;
                color: var(--text-primary) !important;
                padding: 10px 12px !important;
            }
            
            .Select-option:hover {
                background-color: rgba(79, 70, 229, 0.2) !important;
                color: var(--text-primary) !important;
            }
            
            .Select-option.is-selected {
                background-color: var(--primary) !important;
                color: white !important;
            }
            
            .Select-option.is-focused {
                background-color: rgba(79, 70, 229, 0.3) !important;
                color: var(--text-primary) !important;
            }
            
            /* Slider styling */
            .rc-slider {
                margin: 2rem 0 !important;
            }
            
            .rc-slider-track {
                background: linear-gradient(90deg, var(--primary), var(--primary-dark)) !important;
                height: 6px !important;
            }
            
            .rc-slider-rail {
                background: var(--border) !important;
                height: 6px !important;
            }
            
            .rc-slider-handle {
                background: white !important;
                border: 3px solid var(--primary) !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4) !important;
            }
            
            .rc-slider-handle:hover,
            .rc-slider-handle:active {
                border-color: var(--primary-dark) !important;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.6) !important;
            }
            
            .rc-slider-mark-text {
                color: var(--text-secondary) !important;
                font-size: 0.75rem !important;
            }
            
            .rc-slider-mark-text-active {
                color: var(--primary) !important;
            }
            
            /* Input field improvements */
            input[type="text"],
            input[type="number"],
            textarea {
                background: var(--bg-dark) !important;
                border: 1px solid var(--border) !important;
                border-radius: 8px !important;
                color: var(--text-primary) !important;
                padding: 0.75rem !important;
                transition: all 0.3s !important;
            }
            
            input[type="text"]:focus,
            input[type="number"]:focus,
            textarea:focus {
                border-color: var(--primary) !important;
                outline: none !important;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
            }
            
            /* Label improvements */
            label {
                color: var(--text-secondary);
                font-weight: 500;
                margin-bottom: 0.5rem;
                display: block;
            }
            
            /* Card enhancements */
            .result-card {
                background: linear-gradient(135deg, var(--bg-card) 0%, rgba(79, 70, 229, 0.05) 100%);
                border-radius: 12px;
                padding: 1.5rem;
                border: 1px solid var(--border);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 1rem;
                transition: all 0.3s;
            }
            
            .result-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            }
            
            .claim-result-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 24px rgba(79, 70, 229, 0.3) !important;
            }
            
            @keyframes progress {
                0% { width: 0%; }
                50% { width: 70%; }
                100% { width: 100%; }
            }
            
            #download-results-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6) !important;
            }
            
            /* Better label visibility */
            label, .Select-placeholder {
                color: #e2e8f0 !important;
                font-weight: 500 !important;
            }
            
            /* Dropdown text visibility */
            .Select-control {
                background-color: #1e293b !important;
                border-color: #334155 !important;
            }
            
            .Select-menu-outer {
                background-color: #1e293b !important;
                border-color: #334155 !important;
            }
            
            .Select-option {
                background-color: #1e293b !important;
                color: #e2e8f0 !important;
            }
            
            .Select-option:hover {
                background-color: #334155 !important;
            }
            
            .Select-value-label {
                color: #e2e8f0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    html.Div([
        html.H1("🛡️ Obzerra", className="brand-title"),
        html.P("AI-Powered Insurance Fraud Detection System", className="brand-subtitle")
    ], className="header-section"),
    
    dbc.Tabs([
        dbc.Tab(label="📊 Dashboard", tab_id="dashboard"),
        dbc.Tab(label="📦 Batch Analysis", tab_id="batch"),
        dbc.Tab(label="🔍 Single Claim", tab_id="single"),
        dbc.Tab(label="📋 System Logs", tab_id="logs"),
        dbc.Tab(label="📖 User Guide", tab_id="guide"),
    ], id="main-tabs", active_tab="dashboard"),
    
    html.Div(id="tab-content", style={'marginTop': '2rem'}),
    
    dcc.Store(id='uploaded-data-store'),
    dcc.Store(id='analysis-results-store'),
    dcc.Store(id='batch-results-persist', data=None),
    dcc.Store(id='history-view-index', data=None),
    dcc.Store(id='analysis-history-store', data=[]),
    dcc.Store(id='session-stats-store', data={
        'total_analyzed': 0,
        'flagged_count': 0,
        'avg_risk_score': 0,
        'high_risk_count': 0
    }),
    dcc.Store(id='system-logs-store', data=[]),
    dcc.Store(id='download-data-store', data=None),
    dcc.Store(id='batch-progress-store', data={'current': 0, 'total': 0, 'message': ''}),
    dcc.Interval(id='progress-interval', interval=500, disabled=True),
    
], fluid=True, style={'backgroundColor': 'transparent', 'maxWidth': '1400px'})

def create_dashboard_layout(stats):
    total = stats.get('total_analyzed', 0)
    flagged = stats.get('flagged_count', 0)
    avg_risk = stats.get('avg_risk_score', 0)
    high_risk = stats.get('high_risk_count', 0)
    
    if total == 0:
        return html.Div([
            html.Div([
                html.I(className="bi bi-bar-chart-line", style={
                    'fontSize': '3rem',
                    'background': 'var(--primary-gradient)',
                    '-webkit-background-clip': 'text',
                    '-webkit-text-fill-color': 'transparent',
                    'marginBottom': '1rem'
                }),
                html.H3("Executive Dashboard", style={
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '2rem',
                    'fontWeight': '700',
                    'color': 'var(--text-platinum)',
                    'marginBottom': '0.5rem'
                }),
                html.P("No claims analyzed yet. Upload data to begin fraud detection analytics.", style={
                    'color': 'var(--text-muted)',
                    'fontSize': '1.05rem'
                })
            ], style={
                'textAlign': 'center',
                'padding': '4rem 2rem',
                'background': 'var(--bg-frosted)',
                'borderRadius': '16px',
                'border': '1px solid var(--border-medium)'
            })
        ])
    
    fraud_rate = (flagged / total * 100) if total > 0 else 0
    
    kpi_cards = dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-files", style={'fontSize': '2rem', 'color': 'var(--primary-light)', 'opacity': '0.7'}),
                ], style={'position': 'absolute', 'top': '1.5rem', 'right': '1.5rem'}),
                html.Div("Total Claims", className="kpi-label"),
                html.Div(f"{total:,}", className="kpi-value"),
                html.Div([
                    html.Span("Analyzed in session", style={'fontSize': '0.75rem', 'color': 'var(--text-dim)'})
                ], style={'marginTop': '0.5rem'})
            ], className="kpi-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle-fill", style={'fontSize': '2rem', 'color': 'var(--warning-light)', 'opacity': '0.7'}),
                ], style={'position': 'absolute', 'top': '1.5rem', 'right': '1.5rem'}),
                html.Div("Flagged Claims", className="kpi-label"),
                html.Div(f"{flagged:,}", className="kpi-value"),
                html.Div([
                    html.Span(f"{fraud_rate:.1f}% fraud rate", style={
                        'fontSize': '0.8rem',
                        'color': 'var(--warning)',
                        'fontWeight': '600',
                        'background': 'rgba(245, 158, 11, 0.1)',
                        'padding': '0.25rem 0.75rem',
                        'borderRadius': '12px',
                        'display': 'inline-block'
                    })
                ], style={'marginTop': '0.5rem'})
            ], className="kpi-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-speedometer", style={'fontSize': '2rem', 'color': 'var(--accent-light)', 'opacity': '0.7'}),
                ], style={'position': 'absolute', 'top': '1.5rem', 'right': '1.5rem'}),
                html.Div("Avg Risk Score", className="kpi-label"),
                html.Div(f"{avg_risk:.2f}", className="kpi-value"),
                html.Div([
                    html.Div(style={
                        'height': '4px',
                        'background': 'var(--border-medium)',
                        'borderRadius': '2px',
                        'marginTop': '0.5rem',
                        'position': 'relative',
                        'overflow': 'hidden'
                    }, children=[
                        html.Div(style={
                            'height': '100%',
                            'width': f'{min(avg_risk, 100)}%',
                            'background': 'var(--primary-gradient)',
                            'borderRadius': '2px',
                            'transition': 'width 0.3s'
                        })
                    ])
                ], style={'marginTop': '0.5rem'})
            ], className="kpi-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-shield-fill-exclamation", style={'fontSize': '2rem', 'color': 'var(--danger-light)', 'opacity': '0.7'}),
                ], style={'position': 'absolute', 'top': '1.5rem', 'right': '1.5rem'}),
                html.Div("High Risk", className="kpi-label"),
                html.Div(f"{high_risk:,}", className="kpi-value"),
                html.Div([
                    html.Span("Requires immediate review", style={'fontSize': '0.75rem', 'color': 'var(--text-dim)'})
                ], style={'marginTop': '0.5rem'})
            ], className="kpi-card")
        ], md=3),
    ], style={'marginBottom': '3rem'})
    
    results_df = session_manager.get_all_results()
    charts = []
    
    if len(results_df) > 0:
        risk_counts = results_df['risk_level'].value_counts()
        fig_risk = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.5,
            marker=dict(
                colors=['#10b981', '#f59e0b', '#ef4444'],
                line=dict(color='#1e2a47', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='#f8fafc', family='IBM Plex Sans'),
            pull=[0.05, 0.05, 0.1]
        )])
        fig_risk.update_layout(
            title=dict(
                text="Risk Distribution Analysis",
                font=dict(size=18, color='#f8fafc', family='Inter', weight=600),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='rgba(30, 42, 71, 0.7)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='IBM Plex Sans'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            margin=dict(t=60, b=60, l=40, r=40),
            height=400
        )
        
        charts.append(
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=fig_risk, config={'displayModeBar': False})
                ], style={
                    'background': 'var(--bg-frosted)',
                    'backdropFilter': 'blur(20px)',
                    'borderRadius': '16px',
                    'padding': '1.5rem',
                    'border': '1px solid var(--border-medium)',
                    'boxShadow': 'var(--shadow-md)'
                })
            ], md=6)
        )
        
        if 'top_indicators' in results_df.columns:
            all_indicators = []
            for indicators in results_df['top_indicators']:
                if isinstance(indicators, list):
                    all_indicators.extend(indicators)
            
            if all_indicators:
                from collections import Counter
                indicator_counts = Counter(all_indicators)
                top_5 = dict(indicator_counts.most_common(5))
                
                fig_indicators = go.Figure(data=[go.Bar(
                    x=list(top_5.values()),
                    y=list(top_5.keys()),
                    orientation='h',
                    marker=dict(
                        color=list(top_5.values()),
                        colorscale=[[0, '#5b7bff'], [0.5, '#8f5fff'], [1, '#c74fff']],
                        line=dict(color='#1e2a47', width=1)
                    ),
                    text=list(top_5.values()),
                    textposition='outside',
                    textfont=dict(size=12, color='#f8fafc')
                )])
                fig_indicators.update_layout(
                    title=dict(
                        text="Top Fraud Indicators",
                        font=dict(size=18, color='#f8fafc', family='Inter', weight=600),
                        x=0.5,
                        xanchor='center'
                    ),
                    paper_bgcolor='rgba(30, 42, 71, 0.7)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', family='IBM Plex Sans'),
                    xaxis=dict(
                        gridcolor='rgba(148, 163, 184, 0.1)',
                        showgrid=True,
                        zeroline=False,
                        color='#94a3b8'
                    ),
                    yaxis=dict(
                        gridcolor='rgba(148, 163, 184, 0.1)',
                        showgrid=False,
                        color='#94a3b8'
                    ),
                    showlegend=False,
                    margin=dict(t=60, b=40, l=20, r=80),
                    height=400
                )
                
                charts.append(
                    dbc.Col([
                        html.Div([
                            dcc.Graph(figure=fig_indicators, config={'displayModeBar': False})
                        ], style={
                            'background': 'var(--bg-frosted)',
                            'backdropFilter': 'blur(20px)',
                            'borderRadius': '16px',
                            'padding': '1.5rem',
                            'border': '1px solid var(--border-medium)',
                            'boxShadow': 'var(--shadow-md)'
                        })
                    ], md=6)
                )
    
    return html.Div([
        html.Div([
            html.I(className="bi bi-bar-chart-line-fill me-3", style={'fontSize': '1.75rem'}),
            html.Span("Executive Dashboard", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '1.75rem', 'fontWeight': '700'})
        ], className="section-title"),
        kpi_cards,
        dbc.Row(charts, style={'marginTop': '1rem'}) if charts else None
    ])

def create_batch_layout(existing_results=None):
    upload_section = html.Div([
        html.Div([
            html.H5("Step 1: Upload Claims Data", style={'marginBottom': '1rem'}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="bi bi-cloud-upload", style={'fontSize': '3rem', 'color': '#4f46e5'}),
                    html.Div("Drag and drop CSV file or click to browse", style={'marginTop': '1rem', 'fontSize': '1.1rem'}),
                    html.Div("Supported format: CSV files", style={'marginTop': '0.5rem', 'color': '#94a3b8', 'fontSize': '0.875rem'})
                ], className="upload-zone"),
                multiple=False
            ),
            html.Div(id='upload-status', style={'marginTop': '1rem'}),
        ], className="card-modern"),
        
        html.Div(id='data-preview-section', style={'marginTop': '2rem'}),
        html.Div(id='column-mapping-section', style={'marginTop': '2rem'}),
        
        html.Div(id='progress-bar-section', children=[], style={'marginTop': '2rem'}),
    ])
    
    if existing_results:
        return html.Div([
            html.Div([
                html.H3("📦 Batch Claims Analysis", className="section-title", style={'display': 'inline-block', 'marginRight': '2rem'}),
                dbc.Button(
                    [html.I(className="bi bi-plus-circle me-2"), "Start New Analysis"],
                    id='clear-batch-btn',
                    color="primary",
                    outline=True,
                    style={'display': 'inline-block'}
                )
            ], style={'marginBottom': '2rem'}),
            html.Div(id='progress-bar-section', children=[], style={'marginTop': '1rem'}),
            html.Div(id='batch-results-section', children=existing_results, style={'marginTop': '2rem'})
        ])
    
    return html.Div([
        html.H3("📦 Batch Claims Analysis", className="section-title"),
        upload_section,
        dcc.Loading(
            id="batch-loading",
            type="circle",
            color="var(--primary)",
            children=html.Div(id='batch-results-section', style={'marginTop': '2rem'})
        )
    ])

def create_historical_batch_layout(history_item):
    """Build historical results display from a history item"""
    results_json = history_item.get('results_data')
    analysis_name = history_item.get('analysis_name', 'Unnamed Analysis')
    timestamp = history_item.get('timestamp', 'N/A')
    download_data = history_item.get('download_data', None)
    
    if not results_json:
        return html.Div("No results data available", className="card-modern")
    
    try:
        results_df = pd.read_json(io.StringIO(results_json))
        
        claim_cards = []
        for _, row in results_df.iterrows():
            risk_level = row['risk_level']
            
            if risk_level == 'High':
                badge_bg = 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)'
                badge_text = 'HIGH RISK'
            elif risk_level == 'Medium':
                badge_bg = 'linear-gradient(135deg, #ea580c 0%, #f97316 100%)'
                badge_text = 'MEDIUM RISK'
            else:
                badge_bg = 'linear-gradient(135deg, #059669 0%, #10b981 100%)'
                badge_text = 'LOW RISK'
            
            risk_score_val = row['final_risk_score']
            ml_confidence_val = row.get('ml_fraud_probability', 0) * 100
            
            fraud_indicators = row.get('top_indicators', [])
            if isinstance(fraud_indicators, str):
                fraud_indicators = [indicator.strip() for indicator in fraud_indicators.split(',') if indicator.strip()]
            elif isinstance(fraud_indicators, (tuple, set)):
                fraud_indicators = [str(indicator).strip() for indicator in fraud_indicators if str(indicator).strip()]
            elif fraud_indicators is None:
                fraud_indicators = []

            fraud_indicators_html = []
            if fraud_indicators:
                for indicator in fraud_indicators[:5]:
                    fraud_indicators_html.append(
                        html.Li(indicator, style={
                            'color': '#cbd5e1',
                            'marginBottom': '0.4rem',
                            'fontSize': '0.9rem'
                        })
                    )
            else:
                fraud_indicators_html = [html.Li("No specific fraud patterns detected", style={'color': '#10b981', 'fontSize': '0.9rem'})]
            
            if risk_level == 'High':
                next_step = "⛔ IMMEDIATE ACTION: Escalate to senior fraud investigator. Do not approve without thorough investigation."
                next_step_color = '#ef4444'
            elif risk_level == 'Medium':
                next_step = "⚠️ VERIFICATION REQUIRED: Request additional documentation and verify all witness statements before proceeding."
                next_step_color = '#f97316'
            else:
                next_step = "✅ STANDARD PROCESS: Proceed with routine documentation review. Claim shows legitimate patterns."
                next_step_color = '#10b981'
            
            card = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5([
                                html.I(className="bi bi-file-text-fill me-2", style={'color': '#8b5cf6'}),
                                f"Claim #{row['claim_id']}"
                            ], style={'marginBottom': '0.75rem', 'color': '#f1f5f9', 'fontWeight': '600'}),
                            html.Div([
                                html.Span("₱", style={'fontSize': '1.5rem', 'fontWeight': '700', 'color': '#94a3b8'}),
                                html.Span(f"{row['total_claim_amount']:,.2f}", style={'fontSize': '1.8rem', 'fontWeight': '700', 'color': '#e2e8f0', 'marginLeft': '0.25rem'})
                            ])
                        ], width=8),
                        dbc.Col([
                            html.Div(badge_text, style={
                                'background': badge_bg,
                                'color': 'white',
                                'padding': '0.6rem 1.2rem',
                                'borderRadius': '20px',
                                'textAlign': 'center',
                                'fontSize': '0.85rem',
                                'fontWeight': '700',
                                'letterSpacing': '0.5px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.3)'
                            })
                        ], width=4, style={'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'flex-end'})
                    ], style={'marginBottom': '1.5rem'}),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div("Risk Score", style={'fontSize': '0.8rem', 'color': '#94a3b8', 'marginBottom': '0.3rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                html.Div(f"{risk_score_val:.2f}/100", style={'fontSize': '1.4rem', 'fontWeight': '700', 'color': '#e2e8f0'})
                            ], style={'textAlign': 'center', 'padding': '0.75rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'})
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Div("Prediction", style={'fontSize': '0.8rem', 'color': '#94a3b8', 'marginBottom': '0.3rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                html.Div(row['fraud_prediction'], style={
                                    'fontSize': '1.1rem',
                                    'fontWeight': '700',
                                    'color': '#ef4444' if row['fraud_prediction'] == 'Fraud' else '#10b981'
                                })
                            ], style={'textAlign': 'center', 'padding': '0.75rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'})
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Div("ML Confidence", style={'fontSize': '0.8rem', 'color': '#94a3b8', 'marginBottom': '0.3rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                html.Div(f"{ml_confidence_val:.2f}%", style={'fontSize': '1.4rem', 'fontWeight': '700', 'color': '#e2e8f0'})
                            ], style={'textAlign': 'center', 'padding': '0.75rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'})
                        ], width=4)
                    ], style={'marginBottom': '1.5rem'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-exclamation-triangle-fill me-2", style={'color': '#f59e0b'}),
                            html.Span("Fraud Indicators Detected", style={'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#e2e8f0'})
                        ], style={'marginBottom': '0.75rem'}),
                        html.Ul(fraud_indicators_html, style={'paddingLeft': '1.5rem', 'marginBottom': '0'})
                    ], style={'marginBottom': '1.5rem', 'padding': '1rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-lightbulb-fill me-2", style={'color': '#eab308'}),
                            html.Span("Detailed Analysis", style={'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#e2e8f0'})
                        ], style={'marginBottom': '0.75rem'}),
                        html.P(row['explanation'], style={
                            'fontSize': '0.9rem',
                            'color': '#cbd5e1',
                            'lineHeight': '1.7',
                            'marginBottom': '0'
                        })
                    ], style={'marginBottom': '1.5rem', 'padding': '1rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-arrow-right-circle-fill me-2", style={'color': next_step_color}),
                            html.Span("Recommended Action", style={'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#e2e8f0'})
                        ], style={'marginBottom': '0.75rem'}),
                        html.P(next_step, style={
                            'fontSize': '0.95rem',
                            'color': '#e2e8f0',
                            'lineHeight': '1.6',
                            'fontWeight': '500',
                            'marginBottom': '0'
                        })
                    ], style={'padding': '1rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px', 'borderLeft': f'4px solid {next_step_color}'})
                ])
            ], style={
                'marginBottom': '1.5rem',
                'backgroundColor': '#0f172a',
                'border': 'none',
                'boxShadow': '0 4px 16px rgba(0,0,0,0.4)',
                'transition': 'all 0.3s ease'
            }, className="claim-result-card")
            claim_cards.append(card)
        
        return html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H3([
                            html.I(className="bi bi-history me-3"),
                            "Historical Analysis"
                        ], className="section-title"),
                    ], width=6),
                    dbc.Col([
                        dbc.Button([
                            html.I(className="bi bi-plus-circle me-2"),
                            "Start New Analysis"
                        ], id='clear-history-view-btn', color="primary", outline=True,
                        style={'float': 'right'})
                    ], width=6)
                ], style={'marginBottom': '1.5rem'}),
                
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Div([
                        html.Strong(analysis_name, style={'fontSize': '1.05rem'}),
                        html.Br(),
                        html.Small(f"Analyzed on {timestamp}", style={'fontSize': '0.85rem'})
                    ])
                ], color="info", style={'marginBottom': '1.5rem'}),
                
                html.Div(claim_cards)
            ], className="card-modern")
        ])
    except Exception as e:
        return html.Div(f"Error loading historical results: {str(e)}", className="card-modern")

def create_single_claim_layout():
    return html.Div([
        html.H3("🔍 Single Claim Analysis", className="section-title"),
        
        html.Div([
            html.H5("Comprehensive Claim Information", style={'marginBottom': '1.5rem', 'color': '#6366f1'}),
            
            html.Div([
                html.H6("Basic Claim Details", style={'color': '#cbd5e1', 'marginBottom': '1rem', 'borderBottom': '2px solid #334155', 'paddingBottom': '0.5rem'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Claim ID / Policy Number", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dbc.Input(id='single-claim-id', type='text', placeholder='e.g., CLM-2024-001', className="input-field")
                    ], md=6),
                    dbc.Col([
                        html.Label("Total Claim Amount (₱)", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dbc.Input(id='single-amount', type='number', placeholder='e.g., 150000', className="input-field")
                    ], md=6),
                ], style={'marginBottom': '1.5rem'}),
                
                html.H6("Incident Information", style={'color': '#cbd5e1', 'marginBottom': '1rem', 'marginTop': '2rem', 'borderBottom': '2px solid #334155', 'paddingBottom': '0.5rem'}),
                
                html.Div([
                    html.Label("Time of Incident (24-hour format)", style={'marginBottom': '0.75rem', 'display': 'block', 'color': '#e2e8f0', 'fontWeight': '500'}),
                    dcc.Slider(
                        id='single-hour',
                        min=0,
                        max=23,
                        step=1,
                        value=12,
                        marks={i: f'{i:02d}:00' if i % 3 == 0 else str(i) for i in range(0, 24)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': '2rem'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Incident Type", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dcc.Dropdown(
                            id='single-incident-type',
                            options=[
                                {'label': 'Select type...', 'value': ''},
                                {'label': 'Vehicle Collision', 'value': 'Vehicle Collision'},
                                {'label': 'Multi-vehicle Collision', 'value': 'Multi-vehicle Collision'},
                                {'label': 'Single Vehicle Collision', 'value': 'Single Vehicle Collision'},
                                {'label': 'Vehicle Theft', 'value': 'Vehicle Theft'},
                                {'label': 'Parked Car', 'value': 'Parked Car'},
                                {'label': 'Break-in', 'value': 'Break-in'},
                            ],
                            placeholder='Select incident type',
                            className="input-field"
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Incident Severity", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dcc.Dropdown(
                            id='single-severity',
                            options=[
                                {'label': 'Select severity...', 'value': ''},
                                {'label': 'Trivial Damage', 'value': 'Trivial Damage'},
                                {'label': 'Minor Damage', 'value': 'Minor Damage'},
                                {'label': 'Major Damage', 'value': 'Major Damage'},
                                {'label': 'Total Loss', 'value': 'Total Loss'},
                            ],
                            placeholder='Select severity',
                            className="input-field"
                        )
                    ], md=6),
                ], style={'marginBottom': '1.5rem'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Incident State / Location", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dbc.Input(id='single-state', type='text', placeholder='e.g., Metro Manila', className="input-field")
                    ], md=12),
                ], style={'marginBottom': '1.5rem'}),
                
                html.H6("Insured & Witness Details", style={'color': '#cbd5e1', 'marginBottom': '1rem', 'marginTop': '2rem', 'borderBottom': '2px solid #334155', 'paddingBottom': '0.5rem'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Insured Age", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dbc.Input(id='single-age', type='number', placeholder='e.g., 35', className="input-field")
                    ], md=6),
                    dbc.Col([
                        html.Label("Number of Witnesses", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dbc.Input(id='single-witnesses', type='number', placeholder='e.g., 2', min=0, className="input-field")
                    ], md=6),
                ], style={'marginBottom': '1.5rem'}),
                
                html.H6("Police & Documentation", style={'color': '#cbd5e1', 'marginBottom': '1rem', 'marginTop': '2rem', 'borderBottom': '2px solid #334155', 'paddingBottom': '0.5rem'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Police Report Filed?", style={'color': '#e2e8f0', 'fontWeight': '500'}),
                        dcc.Dropdown(
                            id='single-police-report',
                            options=[
                                {'label': 'Select...', 'value': ''},
                                {'label': 'Yes - Police report filed', 'value': 'YES'},
                                {'label': 'No - No police report', 'value': 'NO'},
                                {'label': 'Unknown / Not Applicable', 'value': 'UNKNOWN'},
                            ],
                            placeholder='Select police report status',
                            className="input-field"
                        )
                    ], md=12),
                ], style={'marginBottom': '2rem'}),
            ]),
            
            dbc.Button(
                [html.I(className="bi bi-shield-check me-2"), "Analyze Claim for Fraud"],
                id='analyze-single-btn',
                className="btn-primary-custom",
                style={'width': '100%', 'padding': '1rem', 'fontSize': '1.1rem', 'fontWeight': '600'}
            ),
        ], className="card-modern"),
        
        html.Div(id='single-analysis-result', style={'marginTop': '2rem'})
    ])

def create_logs_layout(logs, history):
    
    history_section = None
    if history and len(history) > 0:
        history_cards = []
        for idx, analysis in enumerate(reversed(history[-10:])):
            timestamp = analysis.get('timestamp', 'N/A')
            analysis_name = analysis.get('analysis_name', 'Unnamed Analysis')
            analysis_type = analysis.get('analysis_type', 'batch')
            claims_count = analysis.get('claims_count', 0)
            high_risk = analysis.get('high_risk_count', 0)
            avg_risk = analysis.get('avg_risk_score', 0)
            fraud_rate = (high_risk / claims_count * 100) if claims_count > 0 else 0
            
            type_icon = "bi bi-stack" if analysis_type == 'batch' else "bi bi-file-earmark-text"
            type_label = "Batch" if analysis_type == 'batch' else "Single"
            
            history_cards.append(
                html.Div([
                    html.Div([
                        html.Div(style={
                            'position': 'absolute',
                            'left': '-20px',
                            'top': '20px',
                            'width': '12px',
                            'height': '12px',
                            'background': 'var(--primary-gradient)',
                            'borderRadius': '50%',
                            'border': '3px solid var(--bg-midnight)',
                            'zIndex': '2'
                        }),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className=type_icon + " me-2", style={'color': 'var(--accent)', 'fontSize': '1.3rem'}),
                                    html.Span(analysis_name, style={
                                        'fontWeight': '700',
                                        'color': 'var(--text-platinum)',
                                        'fontSize': '1.15rem',
                                        'fontFamily': 'Inter, sans-serif'
                                    })
                                ], style={'marginBottom': '0.5rem'}),
                                
                                html.Div([
                                    html.Span(type_label, style={
                                        'fontSize': '0.75rem',
                                        'color': 'var(--text-muted)',
                                        'marginRight': '1rem',
                                        'textTransform': 'uppercase',
                                        'letterSpacing': '0.05em'
                                    }),
                                    html.I(className="bi bi-clock me-1", style={'color': 'var(--primary-light)', 'fontSize': '0.85rem'}),
                                    html.Span(timestamp, style={
                                        'fontWeight': '500',
                                        'color': 'var(--text-silver)',
                                        'fontSize': '0.85rem'
                                    })
                                ], style={'marginBottom': '1rem'}),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Div("Claims Analyzed", style={
                                                'fontSize': '0.7rem',
                                                'color': 'var(--text-dim)',
                                                'textTransform': 'uppercase',
                                                'letterSpacing': '0.05em',
                                                'marginBottom': '0.25rem'
                                            }),
                                            html.Div(f"{claims_count:,}", style={
                                                'fontSize': '1.5rem',
                                                'fontWeight': '800',
                                                'color': 'var(--text-silver)',
                                                'fontFamily': 'Inter, sans-serif'
                                            })
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Div("High Risk", style={
                                                'fontSize': '0.7rem',
                                                'color': 'var(--text-dim)',
                                                'textTransform': 'uppercase',
                                                'letterSpacing': '0.05em',
                                                'marginBottom': '0.25rem'
                                            }),
                                            html.Div(f"{high_risk}", style={
                                                'fontSize': '1.5rem',
                                                'fontWeight': '800',
                                                'color': 'var(--danger)',
                                                'fontFamily': 'Inter, sans-serif'
                                            })
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Div("Avg Risk", style={
                                                'fontSize': '0.7rem',
                                                'color': 'var(--text-dim)',
                                                'textTransform': 'uppercase',
                                                'letterSpacing': '0.05em',
                                                'marginBottom': '0.25rem'
                                            }),
                                            html.Div(f"{avg_risk:.1f}", style={
                                                'fontSize': '1.5rem',
                                                'fontWeight': '800',
                                                'color': 'var(--accent)',
                                                'fontFamily': 'Inter, sans-serif'
                                            })
                                        ])
                                    ], width=4),
                                ], style={'marginBottom': '1rem'}),
                            ], md=9),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="bi bi-eye-fill me-2"),
                                    "View Details"
                                ],
                                id={'type': 'view-history-btn', 'index': len(history) - idx - 1},
                                style={
                                    'background': 'var(--primary-gradient)',
                                    'border': 'none',
                                    'fontWeight': '600',
                                    'padding': '0.75rem 1.5rem',
                                    'borderRadius': '10px',
                                    'boxShadow': '0 4px 12px rgba(99, 102, 241, 0.3)',
                                    'width': '100%'
                                })
                            ], md=3, className="d-flex align-items-center"),
                        ], align="center"),
                    ], style={
                        'background': 'linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%)',
                        'padding': '1.5rem',
                        'borderRadius': '12px',
                        'border': '1px solid var(--border-medium)',
                        'position': 'relative',
                        'marginLeft': '20px',
                        'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        'cursor': 'pointer'
                    })
                ], style={
                    'position': 'relative',
                    'marginBottom': '2rem',
                    'paddingLeft': '10px',
                    'borderLeft': '2px solid var(--border-medium)'
                })
            )
        
        history_section = html.Div([
            html.Div([
                html.I(className="bi bi-clock-history me-3", style={'fontSize': '1.75rem'}),
                html.Span("Analysis History", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '1.75rem', 'fontWeight': '700'})
            ], className="section-title"),
            html.Div(history_cards, style={'marginTop': '2rem'})
        ], className="card-modern")
    
    if not logs or len(logs) == 0:
        return html.Div([
            html.H3("📋 System Logs & History", className="section-title"),
            dbc.Alert("No logs or history yet. Start analyzing claims to see activity here.", color="info"),
            history_section if history_section else None
        ])
    
    log_entries = []
    for log in logs[-20:]:
        timestamp = log.get('timestamp', '')
        message = log.get('message', '')
        log_type = log.get('type', 'info')
        
        color = '#4f46e5' if log_type == 'info' else '#f59e0b' if log_type == 'warning' else '#ef4444'
        
        log_entries.append(
            html.Div([
                html.Div([
                    html.Span(timestamp, style={'color': '#94a3b8', 'fontSize': '0.75rem', 'marginRight': '1rem'}),
                    html.Span(log_type.upper(), style={'color': color, 'fontWeight': '600', 'marginRight': '1rem'}),
                ]),
                html.Div(message, style={'marginTop': '0.5rem'})
            ], className="log-entry")
        )
    
    log_display = html.Div([
        html.H4("📋 Real-Time Activity Log", style={'color': '#4f46e5', 'marginBottom': '1rem'}),
        html.Div(log_entries, style={'maxHeight': '400px', 'overflowY': 'auto'})
    ], className="card-modern")
    
    return html.Div([
        html.H3("📋 System Logs & History", className="section-title"),
        log_display,
        history_section if history_section else None
    ])

def create_guide_layout():
    return html.Div([
        html.H3("📖 User Guide", className="section-title"),
        
        html.Div([
            html.H4("🚀 Getting Started"),
            html.P("Obzerra helps insurance claims officers detect potentially fraudulent claims using AI and statistical analysis."),
            html.Ul([
                html.Li("Upload CSV files with claim data for batch analysis"),
                html.Li("Analyze individual claims for quick fraud assessment"),
                html.Li("View risk scores, explanations, and recommendations"),
            ])
        ], className="guide-section"),
        
        html.Div([
            html.H4("📊 Dashboard"),
            html.P("View overall statistics and trends:"),
            html.Ul([
                html.Li("Total claims analyzed in current session"),
                html.Li("Number of claims flagged for review"),
                html.Li("Average risk score across all claims"),
                html.Li("Visualizations of risk distribution and fraud indicators"),
            ])
        ], className="guide-section"),
        
        html.Div([
            html.H4("📦 Batch Analysis"),
            html.P("Analyze multiple claims at once:"),
            html.Ol([
                html.Li("Upload a CSV file with your claims data"),
                html.Li("Map your columns to system fields (Claim ID, Amount, Hour, etc.)"),
                html.Li("Click 'Analyze Claims for Fraud'"),
                html.Li("Review results table with risk scores and explanations"),
            ]),
            html.P("Required fields: Claim ID, Total Amount, Incident Hour", style={'color': '#f59e0b', 'fontWeight': '500'})
        ], className="guide-section"),
        
        html.Div([
            html.H4("🔍 Single Claim Analysis"),
            html.P("Quickly analyze one claim:"),
            html.Ul([
                html.Li("Enter claim details manually"),
                html.Li("Click 'Analyze Claim' to get instant results"),
                html.Li("See detailed fraud risk assessment and recommendations"),
            ])
        ], className="guide-section"),
        
        html.Div([
            html.H4("🎯 Understanding Risk Scores"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("LOW RISK", className="risk-low"),
                        html.P("Score 0-30: Standard processing recommended", style={'marginTop': '0.5rem', 'color': '#94a3b8'})
                    ])
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Span("MEDIUM RISK", className="risk-medium"),
                        html.P("Score 31-70: Additional verification needed", style={'marginTop': '0.5rem', 'color': '#94a3b8'})
                    ])
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Span("HIGH RISK", className="risk-high"),
                        html.P("Score 71-100: Immediate investigation required", style={'marginTop': '0.5rem', 'color': '#94a3b8'})
                    ])
                ], md=4),
            ])
        ], className="guide-section"),
        
        html.Div([
            html.H4("🛡️ Fraud Detection Rules"),
            html.P("The system uses 8 fraud indicators:"),
            html.Ul([
                html.Li([html.Strong("Z-Score Outliers:"), " Unusual claim amounts"]),
                html.Li([html.Strong("Benford's Law:"), " Number pattern analysis"]),
                html.Li([html.Strong("Unusual Hour:"), " Late night/early morning incidents"]),
                html.Li([html.Strong("Round Amounts:"), " Suspiciously exact amounts"]),
                html.Li([html.Strong("High Amount:"), " Claims exceeding thresholds"]),
                html.Li([html.Strong("Young High Claim:"), " Young drivers with large claims"]),
                html.Li([html.Strong("No Witnesses:"), " High claims without witnesses"]),
                html.Li([html.Strong("High Frequency:"), " Multiple claims in short period"]),
            ], style={'columnCount': 2})
        ], className="guide-section"),
        
        html.Div([
            html.H4("❓ Need Help?"),
            html.P("For technical support or questions, contact your system administrator."),
        ], className="guide-section"),
    ])

@callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab'),
     Input('history-view-index', 'data')],
    [State('session-stats-store', 'data'),
     State('system-logs-store', 'data'),
     State('batch-results-persist', 'data'),
     State('analysis-history-store', 'data')]
)
def render_tab_content(active_tab, history_index, stats, logs, batch_results, history):
    if active_tab == "dashboard":
        return create_dashboard_layout(stats)
    elif active_tab == "batch":
        if history_index is not None and history and history_index < len(history):
            return create_historical_batch_layout(history[history_index])
        else:
            return create_batch_layout(batch_results)
    elif active_tab == "single":
        return create_single_claim_layout()
    elif active_tab == "logs":
        return create_logs_layout(logs, history or [])
    elif active_tab == "guide":
        return create_guide_layout()
    return html.Div("Select a tab")

@callback(
    [Output('upload-status', 'children'),
     Output('uploaded-data-store', 'data'),
     Output('data-preview-section', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return None, None, None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        text_cols = len(df.select_dtypes(include=['object']).columns)
        
        metadata_cards = dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="bi bi-table", style={'fontSize': '1.5rem', 'color': 'var(--primary-light)', 'opacity': '0.7'})
                    ], style={'position': 'absolute', 'top': '1rem', 'right': '1rem'}),
                    html.Div("Total Rows", style={
                        'fontSize': '0.75rem',
                        'color': 'var(--text-muted)',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.1em',
                        'marginBottom': '0.5rem',
                        'fontWeight': '600'
                    }),
                    html.Div(f"{len(df):,}", style={
                        'fontSize': '2rem',
                        'fontWeight': '800',
                        'background': 'var(--primary-gradient)',
                        '-webkit-background-clip': 'text',
                        '-webkit-text-fill-color': 'transparent',
                        'fontFamily': 'Inter, sans-serif'
                    })
                ], style={
                    'background': 'linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%)',
                    'padding': '1.5rem',
                    'borderRadius': '12px',
                    'border': '1px solid var(--border-medium)',
                    'position': 'relative'
                })
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="bi bi-columns", style={'fontSize': '1.5rem', 'color': 'var(--accent-light)', 'opacity': '0.7'})
                    ], style={'position': 'absolute', 'top': '1rem', 'right': '1rem'}),
                    html.Div("Columns", style={
                        'fontSize': '0.75rem',
                        'color': 'var(--text-muted)',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.1em',
                        'marginBottom': '0.5rem',
                        'fontWeight': '600'
                    }),
                    html.Div(f"{len(df.columns)}", style={
                        'fontSize': '2rem',
                        'fontWeight': '800',
                        'color': 'var(--accent)',
                        'fontFamily': 'Inter, sans-serif'
                    }),
                    html.Div(f"{numeric_cols} numeric • {text_cols} text", style={
                        'fontSize': '0.7rem',
                        'color': 'var(--text-dim)',
                        'marginTop': '0.25rem'
                    })
                ], style={
                    'background': 'linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%)',
                    'padding': '1.5rem',
                    'borderRadius': '12px',
                    'border': '1px solid var(--border-medium)',
                    'position': 'relative'
                })
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="bi bi-check-circle-fill" if missing_pct < 5 else "bi bi-exclamation-circle-fill", 
                               style={'fontSize': '1.5rem', 'color': 'var(--success)' if missing_pct < 5 else 'var(--warning)', 'opacity': '0.7'})
                    ], style={'position': 'absolute', 'top': '1rem', 'right': '1rem'}),
                    html.Div("Data Quality", style={
                        'fontSize': '0.75rem',
                        'color': 'var(--text-muted)',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.1em',
                        'marginBottom': '0.5rem',
                        'fontWeight': '600'
                    }),
                    html.Div(f"{100-missing_pct:.1f}%", style={
                        'fontSize': '2rem',
                        'fontWeight': '800',
                        'color': 'var(--success)' if missing_pct < 5 else 'var(--warning)',
                        'fontFamily': 'Inter, sans-serif'
                    }),
                    html.Div(f"{missing_pct:.1f}% missing values", style={
                        'fontSize': '0.7rem',
                        'color': 'var(--text-dim)',
                        'marginTop': '0.25rem'
                    })
                ], style={
                    'background': 'linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%)',
                    'padding': '1.5rem',
                    'borderRadius': '12px',
                    'border': '1px solid var(--border-medium)',
                    'position': 'relative'
                })
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="bi bi-file-earmark-check", style={'fontSize': '1.5rem', 'color': 'var(--success-light)', 'opacity': '0.7'})
                    ], style={'position': 'absolute', 'top': '1rem', 'right': '1rem'}),
                    html.Div("File Name", style={
                        'fontSize': '0.75rem',
                        'color': 'var(--text-muted)',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.1em',
                        'marginBottom': '0.5rem',
                        'fontWeight': '600'
                    }),
                    html.Div(filename[:20] + ('...' if len(filename) > 20 else ''), style={
                        'fontSize': '1.1rem',
                        'fontWeight': '700',
                        'color': 'var(--text-silver)',
                        'fontFamily': 'IBM Plex Sans, sans-serif'
                    }),
                    html.Div("CSV Format", style={
                        'fontSize': '0.7rem',
                        'color': 'var(--text-dim)',
                        'marginTop': '0.25rem'
                    })
                ], style={
                    'background': 'linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-light) 100%)',
                    'padding': '1.5rem',
                    'borderRadius': '12px',
                    'border': '1px solid var(--border-medium)',
                    'position': 'relative'
                })
            ], md=3),
        ], style={'marginBottom': '2rem'})
        
        column_chips = html.Div([
            html.Div("Dataset Columns", style={
                'fontSize': '0.9rem',
                'fontWeight': '600',
                'color': 'var(--text-silver)',
                'marginBottom': '1rem'
            }),
            html.Div([
                html.Span(col, style={
                    'display': 'inline-block',
                    'background': 'var(--bg-card)',
                    'color': 'var(--text-silver)',
                    'padding': '0.4rem 0.9rem',
                    'borderRadius': '20px',
                    'fontSize': '0.8rem',
                    'margin': '0.25rem',
                    'border': '1px solid var(--border-medium)',
                    'fontWeight': '500'
                }) for col in df.columns[:15]
            ] + ([html.Span(f"+ {len(df.columns) - 15} more", style={
                'display': 'inline-block',
                'color': 'var(--text-muted)',
                'padding': '0.4rem 0.9rem',
                'fontSize': '0.8rem',
                'fontStyle': 'italic'
            })] if len(df.columns) > 15 else []))
        ], style={'marginBottom': '2rem'})
        
        preview = html.Div([
            html.Div([
                html.I(className="bi bi-eye-fill me-2", style={'fontSize': '1.5rem'}),
                html.Span("Data Preview", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '1.5rem', 'fontWeight': '700'})
            ], style={'marginBottom': '2rem', 'color': 'var(--text-platinum)'}),
            
            metadata_cards,
            column_chips,
            
            html.Div([
                html.Div("Sample Data (First 10 Rows)", style={
                    'fontSize': '0.9rem',
                    'fontWeight': '600',
                    'color': 'var(--text-silver)',
                    'marginBottom': '1rem'
                }),
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '12px',
                        'backgroundColor': 'var(--bg-deep-navy)',
                        'color': 'var(--text-silver)',
                        'border': '1px solid var(--border-subtle)',
                        'fontSize': '0.85rem'
                    },
                    style_header={
                        'backgroundColor': 'var(--bg-card)',
                        'fontWeight': '700',
                        'textTransform': 'uppercase',
                        'fontSize': '0.75rem',
                        'letterSpacing': '0.05em',
                        'color': 'var(--text-muted)',
                        'border': '1px solid var(--border-medium)'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'var(--bg-navy)',
                        }
                    ],
                    page_size=10
                )
            ], style={
                'background': 'var(--bg-frosted)',
                'backdropFilter': 'blur(20px)',
                'padding': '1.5rem',
                'borderRadius': '12px',
                'border': '1px solid var(--border-medium)'
            })
        ], className="card-modern")
        
        return None, df.to_json(date_format='iso', orient='split'), preview
    
    except Exception as e:
        error_msg = dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            f"Error: {str(e)}"
        ], color="danger")
        return error_msg, None, None

@callback(
    Output('column-mapping-section', 'children'),
    Input('uploaded-data-store', 'data')
)
def show_column_mapping(data_json):
    if not data_json:
        return None
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    columns = df.columns.tolist()
    
    required_options = [{'label': c, 'value': c} for c in columns]
    optional_options = [{'label': '-- No Data --', 'value': ''}] + [{'label': c, 'value': c} for c in columns]
    
    return html.Div([
        html.H5("Step 3: Map Your Columns", style={'marginBottom': '1.5rem'}),
        
        html.Div([
            html.H6("Required Fields", style={'color': '#f59e0b', 'marginBottom': '1rem'}),
            dbc.Row([
                dbc.Col([
                    html.Label("Claim ID / Policy Number"),
                    dcc.Dropdown(id='map-claim-id', options=required_options, className="input-field")
                ], md=4),
                dbc.Col([
                    html.Label("Total Claim Amount (₱)"),
                    dcc.Dropdown(id='map-amount', options=required_options, className="input-field")
                ], md=4),
                dbc.Col([
                    html.Label("Incident Hour (0-23)"),
                    dcc.Dropdown(id='map-hour', options=required_options, className="input-field")
                ], md=4),
            ], style={'marginBottom': '1.5rem'}),
            
            html.H6("Optional Fields", style={'color': '#94a3b8', 'marginBottom': '1rem'}),
            dbc.Row([
                dbc.Col([
                    html.Label("Age"),
                    dcc.Dropdown(id='map-age', options=optional_options, className="input-field")
                ], md=3),
                dbc.Col([
                    html.Label("Incident Type"),
                    dcc.Dropdown(id='map-incident-type', options=optional_options, className="input-field")
                ], md=3),
                dbc.Col([
                    html.Label("Incident Severity"),
                    dcc.Dropdown(id='map-severity', options=optional_options, className="input-field")
                ], md=3),
                dbc.Col([
                    html.Label("Witnesses"),
                    dcc.Dropdown(id='map-witnesses', options=optional_options, className="input-field")
                ], md=3),
            ]),
        ], style={'marginBottom': '2rem'}),
        
        html.Div([
            html.Label("Analysis Name (Optional)", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'color': '#e2e8f0'}),
            dbc.Input(
                id='batch-analysis-name',
                type='text',
                placeholder='e.g., "Weekly Review - Jan 2024" or "Q1 High-Value Claims"',
                className="input-field",
                style={'marginBottom': '1rem'}
            ),
            html.Small("Give this analysis a name to easily find it in your history later", 
                      style={'color': '#94a3b8', 'fontSize': '0.85rem'})
        ], style={'marginBottom': '1.5rem'}),
        
        dbc.Button(
            [html.I(className="bi bi-shield-check me-2"), "Analyze Claims for Fraud"],
            id='analyze-batch-btn',
            className="btn-primary-custom",
            style={'width': '100%'}
        ),
    ], className="card-modern")

@callback(
    [Output('batch-results-section', 'children'),
     Output('session-stats-store', 'data'),
     Output('system-logs-store', 'data'),
     Output('batch-results-persist', 'data'),
     Output('analysis-history-store', 'data'),
     Output('download-data-store', 'data'),
     Output('progress-bar-section', 'children')],
    Input('analyze-batch-btn', 'n_clicks'),
    [State('uploaded-data-store', 'data'),
     State('map-claim-id', 'value'),
     State('map-amount', 'value'),
     State('map-hour', 'value'),
     State('map-age', 'value'),
     State('map-incident-type', 'value'),
     State('map-severity', 'value'),
     State('map-witnesses', 'value'),
     State('batch-analysis-name', 'value'),
     State('session-stats-store', 'data'),
     State('system-logs-store', 'data'),
     State('analysis-history-store', 'data')],
    prevent_initial_call=True
)
def analyze_batch(n_clicks, data_json, claim_id_col, amount_col, hour_col, age_col, type_col, severity_col, witnesses_col, analysis_name, stats, logs, history):
    if not data_json or not claim_id_col or not amount_col or not hour_col:
        return dbc.Alert("Please map all required fields", color="warning"), stats, logs, None, history, None, []
    
    try:
        logs = logs or []
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'info',
            'message': f'Started batch analysis'
        })
        
        df = pd.read_json(io.StringIO(data_json), orient='split')
        
        mapping = {
            'claim_id': claim_id_col,
            'total_claim_amount': amount_col,
            'incident_hour_of_the_day': hour_col
        }
        if age_col and age_col != '': mapping['age'] = age_col
        if type_col and type_col != '': mapping['incident_type'] = type_col
        if severity_col and severity_col != '': mapping['incident_severity'] = severity_col
        if witnesses_col and witnesses_col != '': mapping['witnesses'] = witnesses_col
        
        processed_df = data_processor.prepare_data(df, mapping)
        
        results = []
        for idx, row in processed_df.iterrows():
            rule_analysis = fraud_engine.analyze_single_claim(row.to_dict())
            ml_fraud_prob = ml_manager.predict_single(row.to_dict())
            
            combined_result = {
                'claim_id': row.get('claim_id', 'N/A'),
                'total_claim_amount': row.get('total_claim_amount', 0),
                'rule_risk_score': rule_analysis['risk_score'],
                'ml_fraud_probability': ml_fraud_prob,
                'final_risk_score': (rule_analysis['risk_score'] + ml_fraud_prob * 100) / 2,
                'risk_level': '',
                'fraud_prediction': 'Fraud' if ml_fraud_prob > 0.5 else 'Legitimate',
                'risk_score': (rule_analysis['risk_score'] + ml_fraud_prob * 100) / 2,
                'triggered_rules': rule_analysis.get('triggered_rules', ''),
                'ml_fraud_prob': ml_fraud_prob,
                'top_indicators': rule_analysis.get('top_indicators', []),
                'explanation': ''
            }
            
            final_score = combined_result['final_risk_score']
            if final_score < 30:
                combined_result['risk_level'] = 'Low'
            elif final_score < 70:
                combined_result['risk_level'] = 'Medium'
            else:
                combined_result['risk_level'] = 'High'
            
            combined_result = explanation_engine.add_single_explanation(combined_result)
            results.append(combined_result)
            session_manager.add_analysis(combined_result)
        
        results_df = pd.DataFrame(results)
        
        stats['total_analyzed'] = session_manager.get_total_analyzed()
        stats['flagged_count'] = session_manager.get_flagged_count()
        stats['avg_risk_score'] = session_manager.get_avg_risk_score()
        stats['high_risk_count'] = session_manager.get_high_risk_count()
        
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'info',
            'message': f'Completed analysis: {len(results)} claims processed'
        })
        
        display_df = results_df[['claim_id', 'total_claim_amount', 'final_risk_score', 'risk_level', 'fraud_prediction', 'explanation']].copy()
        display_df.columns = ['Claim ID', 'Amount (₱)', 'Risk Score', 'Risk Level', 'Prediction', 'Explanation']
        display_df['Amount (₱)'] = display_df['Amount (₱)'].apply(lambda x: f"₱{x:,.2f}")
        display_df['Risk Score'] = display_df['Risk Score'].apply(lambda x: f"{x:.1f}")
        
        claim_cards = []
        for _, row in results_df.iterrows():
            risk_level = row['risk_level']
            
            if risk_level == 'High':
                badge_bg = 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)'
                badge_text = 'HIGH RISK'
            elif risk_level == 'Medium':
                badge_bg = 'linear-gradient(135deg, #ea580c 0%, #f97316 100%)'
                badge_text = 'MEDIUM RISK'
            else:
                badge_bg = 'linear-gradient(135deg, #059669 0%, #10b981 100%)'
                badge_text = 'LOW RISK'
            
            risk_score_val = row['final_risk_score']
            ml_confidence_val = row['ml_fraud_probability'] * 100
            
            fraud_indicators = row.get('top_indicators', [])
            if isinstance(fraud_indicators, str):
                fraud_indicators = [indicator.strip() for indicator in fraud_indicators.split(',') if indicator.strip()]
            elif isinstance(fraud_indicators, (tuple, set)):
                fraud_indicators = [str(indicator).strip() for indicator in fraud_indicators if str(indicator).strip()]
            elif fraud_indicators is None:
                fraud_indicators = []

            fraud_indicators_html = []
            if fraud_indicators:
                for indicator in fraud_indicators[:5]:
                    fraud_indicators_html.append(
                        html.Li(indicator, style={
                            'color': '#cbd5e1',
                            'marginBottom': '0.4rem',
                            'fontSize': '0.9rem'
                        })
                    )
            else:
                fraud_indicators_html = [html.Li("No specific fraud patterns detected", style={'color': '#10b981', 'fontSize': '0.9rem'})]
            
            if risk_level == 'High':
                next_step = "⛔ IMMEDIATE ACTION: Escalate to senior fraud investigator. Do not approve without thorough investigation."
                next_step_color = '#ef4444'
            elif risk_level == 'Medium':
                next_step = "⚠️ VERIFICATION REQUIRED: Request additional documentation and verify all witness statements before proceeding."
                next_step_color = '#f97316'
            else:
                next_step = "✅ STANDARD PROCESS: Proceed with routine documentation review. Claim shows legitimate patterns."
                next_step_color = '#10b981'
            
            explanation_text = row['explanation']
            explanation_formatted = explanation_text.replace('%.', '%.').replace('%', '%').replace('.0%', '.00%')
            
            card = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5([
                                html.I(className="bi bi-file-text-fill me-2", style={'color': '#8b5cf6'}),
                                f"Claim #{row['claim_id']}"
                            ], style={'marginBottom': '0.75rem', 'color': '#f1f5f9', 'fontWeight': '600'}),
                            html.Div([
                                html.Span("₱", style={'fontSize': '1.5rem', 'fontWeight': '700', 'color': '#94a3b8'}),
                                html.Span(f"{row['total_claim_amount']:,.2f}", style={'fontSize': '1.8rem', 'fontWeight': '700', 'color': '#e2e8f0', 'marginLeft': '0.25rem'})
                            ])
                        ], width=8),
                        dbc.Col([
                            html.Div(badge_text, style={
                                'background': badge_bg,
                                'color': 'white',
                                'padding': '0.6rem 1.2rem',
                                'borderRadius': '20px',
                                'textAlign': 'center',
                                'fontSize': '0.85rem',
                                'fontWeight': '700',
                                'letterSpacing': '0.5px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.3)'
                            })
                        ], width=4, style={'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'flex-end'})
                    ], style={'marginBottom': '1.5rem'}),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div("Risk Score", style={'fontSize': '0.8rem', 'color': '#94a3b8', 'marginBottom': '0.3rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                html.Div(f"{risk_score_val:.2f}/100", style={'fontSize': '1.4rem', 'fontWeight': '700', 'color': '#e2e8f0'})
                            ], style={'textAlign': 'center', 'padding': '0.75rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'})
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Div("Prediction", style={'fontSize': '0.8rem', 'color': '#94a3b8', 'marginBottom': '0.3rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                html.Div(row['fraud_prediction'], style={
                                    'fontSize': '1.1rem',
                                    'fontWeight': '700',
                                    'color': '#ef4444' if row['fraud_prediction'] == 'Fraud' else '#10b981'
                                })
                            ], style={'textAlign': 'center', 'padding': '0.75rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'})
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Div("ML Confidence", style={'fontSize': '0.8rem', 'color': '#94a3b8', 'marginBottom': '0.3rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                html.Div(f"{ml_confidence_val:.2f}%", style={'fontSize': '1.4rem', 'fontWeight': '700', 'color': '#e2e8f0'})
                            ], style={'textAlign': 'center', 'padding': '0.75rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'})
                        ], width=4)
                    ], style={'marginBottom': '1.5rem'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-exclamation-triangle-fill me-2", style={'color': '#f59e0b'}),
                            html.Span("Fraud Indicators Detected", style={'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#e2e8f0'})
                        ], style={'marginBottom': '0.75rem'}),
                        html.Ul(fraud_indicators_html, style={'paddingLeft': '1.5rem', 'marginBottom': '0'})
                    ], style={'marginBottom': '1.5rem', 'padding': '1rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-lightbulb-fill me-2", style={'color': '#eab308'}),
                            html.Span("Detailed Analysis", style={'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#e2e8f0'})
                        ], style={'marginBottom': '0.75rem'}),
                        html.P(explanation_formatted, style={
                            'fontSize': '0.9rem',
                            'color': '#cbd5e1',
                            'lineHeight': '1.7',
                            'marginBottom': '0'
                        })
                    ], style={'marginBottom': '1.5rem', 'padding': '1rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-arrow-right-circle-fill me-2", style={'color': next_step_color}),
                            html.Span("Recommended Action", style={'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#e2e8f0'})
                        ], style={'marginBottom': '0.75rem'}),
                        html.P(next_step, style={
                            'fontSize': '0.95rem',
                            'color': '#e2e8f0',
                            'lineHeight': '1.6',
                            'fontWeight': '500',
                            'marginBottom': '0'
                        })
                    ], style={'padding': '1rem', 'backgroundColor': '#1e293b', 'borderRadius': '8px', 'borderLeft': f'4px solid {next_step_color}'})
                ])
            ], style={
                'marginBottom': '1.5rem',
                'backgroundColor': '#0f172a',
                'border': 'none',
                'boxShadow': '0 4px 16px rgba(0,0,0,0.4)',
                'transition': 'all 0.3s ease'
            }, className="claim-result-card")
            claim_cards.append(card)
        
        results_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className="bi bi-clipboard-check me-2"),
                        "Analysis Results"
                    ], style={'marginBottom': '0', 'color': '#e2e8f0'}),
                ], width=6),
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-download me-2"),
                        "DOWNLOAD CSV REPORT"
                    ], id='download-results-btn', color="success", size="lg",
                    style={
                        'float': 'right',
                        'background': 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                        'border': 'none',
                        'fontWeight': '700',
                        'padding': '0.75rem 1.5rem',
                        'boxShadow': '0 4px 12px rgba(16, 185, 129, 0.4)',
                        'transition': 'all 0.3s ease'
                    })
                ], width=6)
            ], style={'marginBottom': '1.5rem'}),
            
            dcc.Download(id='download-results'),
            
            dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                f"Successfully analyzed {len(results)} claims"
            ], color="success", style={'marginBottom': '1.5rem'}),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by Risk Level", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'color': '#e2e8f0', 'fontSize': '0.95rem'}),
                    dcc.Dropdown(
                        id='filter-risk',
                        options=[
                            {'label': '🔍 All Claims', 'value': 'all'},
                            {'label': '🔴 High Risk Only', 'value': 'High'},
                            {'label': '🟡 Medium Risk Only', 'value': 'Medium'},
                            {'label': '🟢 Low Risk Only', 'value': 'Low'}
                        ],
                        value='all',
                        style={'backgroundColor': '#1e293b', 'color': '#e2e8f0'}
                    )
                ], md=6),
                dbc.Col([
                    html.Label("Sort by", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'color': '#e2e8f0', 'fontSize': '0.95rem'}),
                    dcc.Dropdown(
                        id='sort-by',
                        options=[
                            {'label': '⬇️ Risk Score (High to Low)', 'value': 'score_desc'},
                            {'label': '⬆️ Risk Score (Low to High)', 'value': 'score_asc'},
                            {'label': '💰 Amount (High to Low)', 'value': 'amount_desc'}
                        ],
                        value='score_desc',
                        style={'backgroundColor': '#1e293b', 'color': '#e2e8f0'}
                    )
                ], md=6),
            ], style={'marginBottom': '2rem'}),
            
            html.Div(id='results-cards-container', children=claim_cards)
        ], className="card-modern")
        
        history = history or []
        if not analysis_name or analysis_name.strip() == '':
            analysis_name = f"Batch Analysis - {datetime.now().strftime('%b %d, %Y %I:%M %p')}"
        
        download_df = results_df[['claim_id', 'total_claim_amount', 'final_risk_score', 'risk_level', 'fraud_prediction', 'ml_fraud_probability', 'explanation']].copy()
        download_data = download_df.to_json(orient='split')
        
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_name': analysis_name,
            'analysis_type': 'batch',
            'claims_count': len(results),
            'high_risk_count': len([r for r in results if r['risk_level'] == 'High']),
            'avg_risk_score': sum([r['final_risk_score'] for r in results]) / len(results),
            'results_data': results_df.to_json(),
            'download_data': download_data
        })
        
        return results_section, stats, logs, results_section, history, download_data, []
    
    except Exception as e:
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'error',
            'message': f'Analysis error: {str(e)}'
        })
        return dbc.Alert(f"Analysis error: {str(e)}", color="danger"), stats, logs, None, history, None, []

@callback(
    [Output('single-analysis-result', 'children'),
     Output('session-stats-store', 'data', allow_duplicate=True),
     Output('system-logs-store', 'data', allow_duplicate=True)],
    Input('analyze-single-btn', 'n_clicks'),
    [State('single-claim-id', 'value'),
     State('single-amount', 'value'),
     State('single-hour', 'value'),
     State('single-age', 'value'),
     State('single-witnesses', 'value'),
     State('single-incident-type', 'value'),
     State('single-severity', 'value'),
     State('single-state', 'value'),
     State('single-police-report', 'value'),
     State('session-stats-store', 'data'),
     State('system-logs-store', 'data')],
    prevent_initial_call=True
)
def analyze_single_claim(n_clicks, claim_id, amount, hour, age, witnesses, incident_type, severity, state, police_report, stats, logs):
    if not claim_id or amount is None or hour is None:
        return dbc.Alert("Please fill in all required fields (Claim ID, Amount, Hour)", color="warning"), stats, logs
    
    try:
        logs = logs or []
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'info',
            'message': f'Analyzing single claim: {claim_id}'
        })
        
        claim_data = {
            'claim_id': claim_id,
            'total_claim_amount': float(amount),
            'incident_hour_of_the_day': int(hour)
        }
        
        if age: claim_data['age'] = int(age)
        if witnesses is not None: claim_data['witnesses'] = int(witnesses)
        if incident_type: claim_data['incident_type'] = incident_type
        if severity: claim_data['incident_severity'] = severity
        if state: claim_data['incident_state'] = state
        if police_report: claim_data['police_report'] = police_report
        
        df = pd.DataFrame([claim_data])
        mapping = {
            'claim_id': 'claim_id',
            'total_claim_amount': 'total_claim_amount',
            'incident_hour_of_the_day': 'incident_hour_of_the_day'
        }
        if age: mapping['age'] = 'age'
        if witnesses is not None: mapping['witnesses'] = 'witnesses'
        if incident_type: mapping['incident_type'] = 'incident_type'
        if severity: mapping['incident_severity'] = 'incident_severity'
        if state: mapping['incident_state'] = 'incident_state'
        
        processed_df = data_processor.prepare_data(df, mapping)
        
        row = processed_df.iloc[0]
        rule_analysis = fraud_engine.analyze_single_claim(row.to_dict())
        ml_fraud_prob = ml_manager.predict_single(row.to_dict())
        
        combined_result = {
            'claim_id': claim_id,
            'total_claim_amount': amount,
            'rule_risk_score': rule_analysis['risk_score'],
            'ml_fraud_probability': ml_fraud_prob,
            'final_risk_score': (rule_analysis['risk_score'] + ml_fraud_prob * 100) / 2,
            'risk_score': (rule_analysis['risk_score'] + ml_fraud_prob * 100) / 2,
            'triggered_rules': rule_analysis.get('triggered_rules', ''),
            'ml_fraud_prob': ml_fraud_prob,
            'fraud_prediction': 'Fraud' if ml_fraud_prob > 0.5 else 'Legitimate',
            'top_indicators': rule_analysis.get('top_indicators', []),
        }
        
        final_score = combined_result['final_risk_score']
        if final_score < 30:
            combined_result['risk_level'] = 'Low'
        elif final_score < 70:
            combined_result['risk_level'] = 'Medium'
        else:
            combined_result['risk_level'] = 'High'
        
        combined_result = explanation_engine.add_single_explanation(combined_result)
        session_manager.add_analysis(combined_result)
        
        stats['total_analyzed'] = session_manager.get_total_analyzed()
        stats['flagged_count'] = session_manager.get_flagged_count()
        stats['avg_risk_score'] = session_manager.get_avg_risk_score()
        stats['high_risk_count'] = session_manager.get_high_risk_count()
        
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'info',
            'message': f'Completed: {claim_id} - Risk: {combined_result["risk_level"]}'
        })
        
        risk_class = f"risk-{combined_result['risk_level'].lower()}"
        
        result_display = html.Div([
            html.H4("🎯 Fraud Detection Analysis Complete", style={'marginBottom': '2rem', 'color': '#4f46e5'}),
            
            html.Div([
                html.H6("Claim Summary", style={'marginBottom': '1rem'}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Claim ID: ", style={'color': '#94a3b8'}),
                            html.Span(claim_id, style={'color': '#f1f5f9', 'fontWeight': '600'})
                        ])
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.Span("Amount: ", style={'color': '#94a3b8'}),
                            html.Span(f"₱{amount:,.2f}", style={'color': '#f1f5f9', 'fontWeight': '600'})
                        ])
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Incident Time: ", style={'color': '#94a3b8'}),
                            html.Span(f"{hour:02d}:00", style={'color': '#f1f5f9', 'fontWeight': '600'})
                        ], style={'marginTop': '0.5rem'})
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.Span("Police Report: ", style={'color': '#94a3b8'}),
                            html.Span(police_report if police_report else "Not specified", style={'color': '#f1f5f9', 'fontWeight': '600'})
                        ], style={'marginTop': '0.5rem'})
                    ], md=6) if police_report else html.Div(),
                ])
            ], className="result-card", style={'marginBottom': '2rem'}),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Final Risk Score", className="kpi-label"),
                        html.Div(f"{final_score:.1f}/100", className="kpi-value")
                    ], className="kpi-card")
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Div("Risk Level", className="kpi-label"),
                        html.Span(combined_result['risk_level'], className=risk_class, style={'fontSize': '1.5rem'})
                    ], className="kpi-card")
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Div("Prediction", className="kpi-label"),
                        html.Div(combined_result['fraud_prediction'], className="kpi-value",
                                style={'color': '#ef4444' if combined_result['fraud_prediction'] == 'Fraud' else '#10b981'})
                    ], className="kpi-card")
                ], md=4),
            ], style={'marginBottom': '2rem'}),
            
            html.Div([
                html.H6("🔍 Detailed Explanation", style={'color': '#4f46e5', 'marginBottom': '1rem'}),
                html.P(combined_result.get('explanation', 'No explanation available'), style={'lineHeight': '1.8'})
            ], className="result-card"),
            
            html.Div([
                html.H6("⚠️ Fraud Indicators Detected", style={'color': '#f59e0b', 'marginBottom': '1rem'}),
                html.Ul([html.Li(indicator, style={'marginBottom': '0.5rem'}) for indicator in combined_result.get('top_indicators', [])] if combined_result.get('top_indicators') else [html.Li("No specific indicators detected", style={'color': '#10b981'})])
            ], className="result-card"),
            
            html.Div([
                html.H6("📋 Action Recommendation", style={'color': '#10b981' if combined_result['risk_level'] == 'Low' else '#f59e0b' if combined_result['risk_level'] == 'Medium' else '#ef4444', 'marginBottom': '1rem'}),
                html.P(
                    "⛔ IMMEDIATE INVESTIGATION REQUIRED: Escalate to senior fraud investigator. Do not process this claim without thorough review." if combined_result['risk_level'] == 'High'
                    else "⚠️ ADDITIONAL VERIFICATION NEEDED: Request supporting documentation before processing. Verify witness statements and police reports." if combined_result['risk_level'] == 'Medium'
                    else "✅ STANDARD PROCESSING APPROVED: Routine documentation review recommended. Claim appears legitimate.",
                    style={'lineHeight': '1.8', 'fontWeight': '500'}
                )
            ], className="result-card")
        ])
        
        return result_display, stats, logs
    
    except Exception as e:
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'error',
            'message': f'Single claim analysis error: {str(e)}'
        })
        return dbc.Alert(f"Analysis error: {str(e)}", color="danger"), stats, logs

@callback(
    Output('batch-results-persist', 'data', allow_duplicate=True),
    Input('clear-batch-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_batch_results(n_clicks):
    return None

@callback(
    Output('history-view-index', 'data', allow_duplicate=True),
    Input('clear-history-view-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_history_view(n_clicks):
    return None

@callback(
    Output('download-data-store', 'data', allow_duplicate=True),
    Input('history-view-index', 'data'),
    State('analysis-history-store', 'data'),
    prevent_initial_call=True
)
def update_download_data_from_history(history_index, history):
    if history_index is not None and history and history_index < len(history):
        return history[history_index].get('download_data', None)
    return None

@callback(
    Output('download-results', 'data'),
    Input('download-results-btn', 'n_clicks'),
    State('download-data-store', 'data'),
    prevent_initial_call=True
)
def download_results(n_clicks, download_data):
    if not download_data:
        raise dash.exceptions.PreventUpdate
    
    df = pd.read_json(io.StringIO(download_data), orient='split')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"fraud_analysis_results_{timestamp}.csv"
    
    return dcc.send_data_frame(df.to_csv, filename, index=False)

@callback(
    [Output('history-view-index', 'data'),
     Output('main-tabs', 'active_tab')],
    Input({'type': 'view-history-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
    State('analysis-history-store', 'data'),
    prevent_initial_call=True
)
def view_historical_results(n_clicks_list, history):
    if not any(n_clicks_list) or not history:
        raise dash.exceptions.PreventUpdate
    
    ctx_triggered = ctx.triggered[0]
    if not ctx_triggered['value']:
        raise dash.exceptions.PreventUpdate
    
    button_id = ctx_triggered['prop_id'].split('.')[0]
    import json
    button_data = json.loads(button_id)
    index = button_data['index']
    
    if index < len(history):
        return index, "batch"
    
    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
