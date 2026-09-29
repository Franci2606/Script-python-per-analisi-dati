# app.py
# Dashboard interattiva per l'analisi del questionario MANILA

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from Utils.llm_async import crea_client_async
from Utils.async_helpers import esegui_coroutine
import base64
import warnings
import os
import re
import json

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

from dotenv import load_dotenv
load_dotenv()

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.models").setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', category=RuntimeWarning)

from config import get_config
from Utils.analisi_likert import AnalisiLikert
from Utils.analisi_testo import AnalisiTesto
from Utils.analisi_profilo import AnalisiProfilo
from Utils.grafici import GeneratoreGrafici
from Utils.tabelle import GeneratoreTabelle
from Utils.report_generator import ReportGenerator
from Utils.analisi_tematica_llm import AnalisiTematicaLLM, nome_tema
from Utils.llm_clients import crea_client, PROVIDER_DISPONIBILI
from mapping_config import MAPPING_GOOGLE_FORMS


# ============================================================================
# CACHE GLOBALE DEL MODELLO DI SENTIMENT
# ============================================================================
@st.cache_resource(show_spinner=False)
def carica_sentiment_pipeline():
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
        device=device,
        truncation=True,
        max_length=512
    )
    print(f"🧠 Modello sentiment caricato in cache (device={'GPU' if device == 0 else 'CPU'})")
    return pipe


# ============================================================================
# CSS PERSONALIZZATO
# ============================================================================
def load_css():
    st.markdown("""
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f7fa; }
        .main-header {
            background: linear-gradient(135deg, #1E88E5, #1565C0);
            padding: 30px 40px; border-radius: 16px; margin-bottom: 30px;
            color: white; box-shadow: 0 4px 20px rgba(30, 136, 229, 0.3);
        }
        .main-header h1 { margin: 0; font-size: 32px; font-weight: 700; }
        .main-header p { margin: 8px 0 0 0; opacity: 0.9; font-size: 16px; }
        .sidebar-header {
            background: linear-gradient(135deg, #1E88E5, #1565C0);
            padding: 20px; border-radius: 12px; color: white;
            text-align: center; margin-bottom: 20px;
        }
        .sidebar-header h3 { margin: 0; font-size: 20px; }
        .sidebar-header p { margin: 5px 0 0 0; opacity: 0.8; font-size: 13px; }
        .metric-card {
            background: white; padding: 20px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center;
            height: 100%; transition: transform 0.2s, box-shadow 0.2s;
            border-top: 4px solid #1E88E5;
        }
        .metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.10); }
        .metric-icon { font-size: 28px; margin-bottom: 8px; }
        .metric-value { font-size: 28px; font-weight: 700; color: #1E88E5; }
        .metric-label { color: #666; font-size: 14px; margin-top: 4px; }
        .metric-sub { color: #999; font-size: 12px; margin-top: 2px; }
        .chart-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin: 10px 0; }
        .chart-card h4 { color: #333; margin-bottom: 16px; font-weight: 600; }
        .custom-tabs .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f0f2f6; border-radius: 10px; padding: 4px; }
        .custom-tabs .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; background-color: transparent; font-weight: 500; color: #555; }
        .custom-tabs .stTabs [aria-selected="true"] { background-color: white; color: #1E88E5; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
        .streamlit-expanderHeader { font-weight: 600 !important; background-color: #f8f9fa !important; border-radius: 8px !important; }
        .stButton > button {
            border-radius: 8px !important; font-weight: 500 !important;
            transition: all 0.3s ease !important;
            background: linear-gradient(135deg, #1E88E5, #1565C0) !important;
            color: white !important; border: none !important; padding: 8px 24px !important;
        }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(30, 136, 229, 0.4) !important; }
        .stSelectbox > div { border-radius: 8px !important; }
        .stDataFrame { border-radius: 12px !important; overflow: hidden !important; box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important; }
        .stAlert { border-radius: 12px !important; border-left: 4px solid #1E88E5 !important; }
        .footer { text-align: center; padding: 20px 0; color: #999; font-size: 13px; border-top: 1px solid #eee; margin-top: 40px; }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #1E88E5, #43A047) !important; }
        @media (max-width: 768px) {
            .main-header { padding: 20px; }
            .main-header h1 { font-size: 24px; }
            .metric-card { padding: 12px; }
            .metric-value { font-size: 22px; }
        }
        .nielsen-legend {
            background: #f8f9fa; border-left: 4px solid #1E88E5;
            padding: 12px 18px; margin: 15px 0 25px 0; border-radius: 4px;
            font-size: 0.9em;
        }
        .nielsen-legend ul { margin: 0; padding-left: 20px; columns: 2; column-gap: 30px; list-style: none; }
        .nielsen-legend li { margin: 3px 0; break-inside: avoid; }
        .nielsen-legend li strong { color: #1E88E5; display: inline-block; min-width: 32px; }
        .nielsen-tag {
            display: inline-block; background: #eaf4fc; color: #2c3e50;
            border: 1px solid #b8d8ea; border-radius: 3px;
            padding: 2px 8px; font-size: 0.85em; margin: 1px 0; white-space: nowrap;
        }
        @media (max-width: 768px) {
            .nielsen-legend ul { columns: 1; }
        }
    </style>
    """, unsafe_allow_html=True)


def crea_metrica_card(titolo, valore, icona, colore, subtitolo=None, colore_bordo=None):
    colore_bordo = colore_bordo or colore
    return f"""
        <div class="metric-card" style="border-top-color: {colore_bordo};">
            <div class="metric-icon">{icona}</div>
            <div class="metric-value" style="color: {colore};">{valore}</div>
            <div class="metric-label">{titolo}</div>
            {f'<div class="metric-sub">{subtitolo}</div>' if subtitolo else ''}
        </div>
    """


def crea_container_grafico(titolo):
    return st.markdown(f'<div class="chart-card"><h4>{titolo}</h4>', unsafe_allow_html=True)


def chiudi_container_grafico():
    st.markdown("</div>", unsafe_allow_html=True)


def mostra_risposte_aperte(risposte_con_sentiment, max_mostra=10, lingua='it'):
    colori = {
        'positivo': {'bg': '#E8F5E9', 'border': '#43A047',
                     'label': '😊 Positivo' if lingua == 'it' else '😊 Positive',
                     'text': '#2E7D32'},
        'neutro': {'bg': '#FFF3E0', 'border': '#FB8C00',
                   'label': '😐 Neutro' if lingua == 'it' else '😐 Neutral',
                   'text': '#E65100'},
        'negativo': {'bg': '#FFEBEE', 'border': '#E53935',
                     'label': '😞 Negativo' if lingua == 'it' else '😞 Negative',
                     'text': '#C62828'},
    }
    for i, (risposta, sentiment) in enumerate(risposte_con_sentiment[:max_mostra], 1):
        c = colori.get(sentiment, colori['neutro'])
        st.markdown(f"""
            <div style="background: {c['bg']}; padding: 12px 16px; border-radius: 8px;
                        margin: 6px 0; border-left: 4px solid {c['border']};">
                <div style="display: flex; justify-content: space-between;
                            align-items: center; margin-bottom: 6px;">
                    <strong style="color: {c['border']};">#{i}</strong>
                    <span style="background: {c['border']}; color: white;
                                 padding: 2px 10px; border-radius: 12px;
                                 font-size: 12px; font-weight: 500;">
                        {c['label']}
                    </span>
                </div>
                <div style="color: #333; line-height: 1.5;">{risposta}</div>
            </div>
        """, unsafe_allow_html=True)
    if len(risposte_con_sentiment) > max_mostra:
        st.info(f"... e altre {len(risposte_con_sentiment) - max_mostra} risposte")


# ============================================================================
# MAPPATURA FLESSIBILE
# ============================================================================
def _normalizza_header(s):
    s = str(s).lower().strip()
    s = re.sub(r'^\s*\d+[\.\)]\s*', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\.\?\!\:\;]+\s*$', '', s)
    return s


def applica_mappatura_flessibile(df, mapping):
    mapping_norm = {_normalizza_header(k): v for k, v in mapping.items()}
    nuove_colonne = {}
    colonne_non_mappate = []
    for col in df.columns:
        col_norm = _normalizza_header(col)
        if col_norm in mapping_norm:
            nuove_colonne[col] = mapping_norm[col_norm]
        elif col in mapping.values():
            continue
        else:
            colonne_non_mappate.append(col)
    df = df.rename(columns=nuove_colonne)
    if colonne_non_mappate:
        print(f"⚠️ Colonne non mappate ({len(colonne_non_mappate)}): {colonne_non_mappate}")
    return df


# ============================================================================
# CARICA CSS E CONFIG PAGINA
# ============================================================================
load_css()
st.set_page_config(
    page_title="MANILA Questionnaire Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'lingua' not in st.session_state:
    st.session_state.lingua = 'it'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analisi_tematica' not in st.session_state:
    st.session_state.analisi_tematica = None

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h3>📊 MANILA</h3>
            <p>Questionnaire Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌐 Language")
    lingua_sel = st.radio(
        "Lingua", ["🇮🇹 Italiano", "🇬🇧 English"],
        index=0 if st.session_state.lingua == 'it' else 1,
        horizontal=True, key="lingua_radio",
        label_visibility="collapsed"
    )
    nuova_lingua = 'it' if lingua_sel == "🇮🇹 Italiano" else 'en'
    if nuova_lingua != st.session_state.lingua:
        st.session_state.lingua = nuova_lingua
        # Reset degli stati dipendenti dalla lingua
        if st.session_state.analisi_tematica is not None:
            st.session_state.analisi_tematica.lingua = nuova_lingua
            st.session_state.analisi_tematica.config = get_config(nuova_lingua)
        st.session_state.problemi_priorita = None
        st.session_state.priorita_domanda_corrente = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV/Excel file" if st.session_state.lingua == 'en' else "Carica il file CSV/Excel",
        type=['csv', 'xlsx', 'xls'],
        key="file_uploader"
    )


    if st.session_state.lingua == 'it':
        testo_bottone_genera = "📊 Genera Report HTML"
        testo_bottone_download = "📥 Scarica Report HTML"
        testo_successo = "✅ Report generato con successo!"
        testo_errore = "❌ Errore: "
        testo_info = "Assicurati che il file utils/report_generator.py esista."
        testo_spinner = "Generando report..."
    else:
        testo_bottone_genera = "📊 Generate HTML Report"
        testo_bottone_download = "📥 Download HTML Report"
        testo_successo = "✅ Report generated successfully!"
        testo_errore = "❌ Error: "
        testo_info = "Make sure the file utils/report_generator.py exists."
        testo_spinner = "Generating report..."

config = get_config(st.session_state.lingua)

# ============================================================================
# TRADUZIONI
# ============================================================================
if st.session_state.lingua == 'it':
    T = {
        'titolo': 'Analisi Questionario MANILA',
        'sottotitolo': "Valutazione dell'applicazione MANILA - Dashboard Interattiva",
        'totale_risposte': 'Totale risposte',
        'domande_likert': 'Domande Likert',
        'domande_aperte': 'Domande Aperte',
        'completamento': 'Completamento medio',
        'profilo': '👤 Profilo utenti',
        'panoramica': '📊 Panoramica',
        'analisi_likert': '📈 Analisi Likert',
        'analisi_testo': '📝 Analisi Testo',
        'analisi_tematica': '🤖 Analisi Tematica',
        'dati_grezzi': '📋 Dati Grezzi',
        'statistiche': 'Statistiche Descrittive',
        'distribuzione': 'Distribuzione delle Risposte',
        'alpha_cronbach': "Alpha di Cronbach",
        'correlazione': 'Matrice di Correlazione',
        'seleziona_domanda': 'Seleziona una domanda:',
        'wordcloud': '☁️ Nuvola di Parole',
        'parole_frequenti': '📊 Parole più Frequenti',
        'sentiment': '🎭 Analisi del Sentiment',
        'positivo': '😊 Positivo',
        'neutro': '😐 Neutro',
        'negativo': '😞 Negativo',
        'scarica_stats': '📥 Scarica statistiche (CSV)',
        'genera_grafico': '🔄 Genera Grafico',
        'mostra_correlazione': '🔄 Mostra Matrice di Correlazione',
        'nessun_dato': 'Nessun dato disponibile per questa domanda.',
        'nessuna_risposta': 'Nessuna risposta aperta trovata.',
        'selezione_scala': 'Seleziona una scala:',
        'caricamento_dati': 'Caricamento dati...',
        'elaborazione': 'Elaborazione in corso...',
        'caricato': '✅ Dati caricati con successo!',
        'errore': '❌ Errore nel caricamento del file',
        'consenso': 'Consenso',
        'posizione': 'Posizione attuale',
        'esperienza': 'Esperienza ML',
        'conoscenza': 'Conoscenza Fairness',
    }
else:
    T = {
        'titolo': 'MANILA Questionnaire Analysis',
        'sottotitolo': "Evaluation of the MANILA application - Interactive Dashboard",
        'totale_risposte': 'Total responses',
        'domande_likert': 'Likert questions',
        'domande_aperte': 'Open questions',
        'completamento': 'Average completion',
        'profilo': '👤 Users profile',
        'panoramica': '📊 Overview',
        'analisi_likert': '📈 Likert Analysis',
        'analisi_testo': '📝 Text Analysis',
        'analisi_tematica': '🤖 Thematic Analysis',
        'dati_grezzi': '📋 Raw Data',
        'statistiche': 'Descriptive Statistics',
        'distribuzione': 'Response Distribution',
        'alpha_cronbach': "Cronbach's Alpha",
        'correlazione': 'Correlation Matrix',
        'seleziona_domanda': 'Select a question:',
        'wordcloud': '☁️ Word Cloud',
        'parole_frequenti': '📊 Most Frequent Words',
        'sentiment': '🎭 Sentiment Analysis',
        'positivo': '😊 Positive',
        'neutro': '😐 Neutral',
        'negativo': '😞 Negative',
        'scarica_stats': '📥 Download statistics (CSV)',
        'genera_grafico': '🔄 Generate Chart',
        'mostra_correlazione': '🔄 Show Correlation Matrix',
        'nessun_dato': 'No data available for this question.',
        'nessuna_risposta': 'No open responses found.',
        'selezione_scala': 'Select a scale:',
        'caricamento_dati': 'Loading data...',
        'elaborazione': 'Processing...',
        'caricato': '✅ Data loaded successfully!',
        'errore': '❌ Error loading file',
        'consenso': 'Consent',
        'posizione': 'Position',
        'esperienza': 'ML Experience',
        'conoscenza': 'Fairness Knowledge',
    }


# ============================================================================
# RICREA ANALISI
# ============================================================================
def ricrea_analisi(df, config, lingua, sentiment_pipeline=None):
    analisi_likert = AnalisiLikert(
        df,
        config['domande_likert'],
        config['mappatura_scale'],
        config['scale']
    )
    analisi_testo = AnalisiTesto(
        df,
        config['domande_aperte'],
        lingua,
        sentiment_pipeline=sentiment_pipeline
    )
    analisi_profilo = AnalisiProfilo(
        df,
        config.get('domande_profilo', {}),
        config.get('opzioni_multipla', {}),
        config.get('labels_profilo', {}),
        config.get('mapping_valori_profilo', {})
    )
    return analisi_likert, analisi_testo, analisi_profilo


# ============================================================================
# HEADER
# ============================================================================
st.markdown(f"""
    <div class="main-header">
        <h1>📊 {T['titolo']}</h1>
        <p>{T['sottotitolo']}</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# CARICAMENTO DATI
# ============================================================================
if uploaded_file is not None:
    try:
        with st.spinner(T['caricamento_dati']):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df = applica_mappatura_flessibile(df, MAPPING_GOOGLE_FORMS)
            if 'Timestamp' in df.columns:
                df = df.drop(columns=['Timestamp'])

            st.session_state.df = df
            st.success(f"{T['caricato']} {len(df)} {T['totale_risposte'].lower()}.")

    except Exception as e:
        st.error(f"{T['errore']}: {str(e)}")

# ============================================================================
# DASHBOARD
# ============================================================================
if st.session_state.df is not None:
    df = st.session_state.df

    with st.spinner(T['elaborazione']):
        sentiment_pipeline = carica_sentiment_pipeline()
        analisi_likert, analisi_testo, analisi_profilo = ricrea_analisi(
            df, config, st.session_state.lingua,
            sentiment_pipeline=sentiment_pipeline
        )

    generatore_grafici = GeneratoreGrafici()
    generatore_tabelle = GeneratoreTabelle()

    # ---- METRICHE ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(crea_metrica_card(T['totale_risposte'], len(df), "📋", "#1E88E5"), unsafe_allow_html=True)
    with col2:
        st.markdown(crea_metrica_card(T['domande_likert'], len(config['domande_likert']), "📊", "#43A047"), unsafe_allow_html=True)
    with col3:
        st.markdown(crea_metrica_card(T['domande_aperte'], len(config['domande_aperte']), "📝", "#FB8C00"), unsafe_allow_html=True)
    with col4:
        completamento = df.notna().mean().mean() * 100
        st.markdown(crea_metrica_card(T['completamento'], f"{completamento:.1f}%", "✅", "#8E24AA"), unsafe_allow_html=True)

    st.markdown("---")

    # ---- TABS (6 tab ora) ----
    st.markdown('<div class="custom-tabs">', unsafe_allow_html=True)
    tab_profilo, tab1, tab2, tab3, tab_tematica, tab4 = st.tabs([
        f" {T['profilo']}",
        f" {T['panoramica']}",
        f" {T['analisi_likert']}",
        f" {T['analisi_testo']}",
        f" {T['analisi_tematica']}",
        f" {T['dati_grezzi']}"
    ])

    # ============================================================
    # TAB PROFILO (Q1-Q4)
    # ============================================================
    with tab_profilo:
        st.header(T['profilo'])

        if 'Q1' in analisi_profilo.df_profilo.columns:
            st.subheader(f"Q1 — {T['consenso']}")
            freq_q1 = analisi_profilo.frequenze('Q1', lingua=st.session_state.lingua)
            if freq_q1 is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(freq_q1, hide_index=True, use_container_width=True)
                with col2:
                    fig = analisi_profilo.grafico_categorico('Q1', 'Q1', lingua=st.session_state.lingua)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

        st.markdown("---")

        if 'Q2' in analisi_profilo.df_profilo.columns:
            st.subheader(f"Q2 — {T['posizione']}")
            freq_q2 = analisi_profilo.frequenze('Q2', lingua=st.session_state.lingua)
            if freq_q2 is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(freq_q2, hide_index=True, use_container_width=True)
                with col2:
                    fig = analisi_profilo.grafico_categorico('Q2', 'Q2', lingua=st.session_state.lingua)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

        st.markdown("---")

        col_q3, col_q4 = st.columns(2)

        with col_q3:
            if 'Q3' in analisi_profilo.df_profilo.columns:
                st.subheader(f"Q3 — {T['esperienza']}")
                labels_q3 = config.get('labels_profilo', {}).get('Q3', {})
                fig = analisi_profilo.grafico_ordinale('Q3', 'Q3', labels=labels_q3, lingua=st.session_state.lingua)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

        with col_q4:
            if 'Q4' in analisi_profilo.df_profilo.columns:
                st.subheader(f"Q4 — {T['conoscenza']}")
                labels_q4 = config.get('labels_profilo', {}).get('Q4', {})
                fig = analisi_profilo.grafico_ordinale('Q4', 'Q4', labels=labels_q4, lingua=st.session_state.lingua)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

    # ============================================================
    # TAB 1: PANORAMICA
    # ============================================================
    with tab1:
        st.header(T['panoramica'])
        st.subheader("📊 " + ("Mean ratings by section" if st.session_state.lingua == 'en' else "Medie delle valutazioni per sezione"))

        spinner_graf = ('Generating chart...' 
                        if st.session_state.lingua == 'en' 
                        else "Generando grafico...")
        with st.spinner(spinner_graf):
            crea_container_grafico("")
            fig = generatore_grafici.grafico_medie_per_sezione(
                analisi_likert.df_likert,
                config['sezioni'],
                config['domande_likert'],
                salva=False,
                lingua=st.session_state.lingua
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(T['nessun_dato'])
            chiudi_container_grafico()

        with st.expander(
            "📊 " + ("Quick statistics" if st.session_state.lingua == 'en' 
                     else "Statistiche rapide"),
            expanded=False
        ):
            stats = analisi_likert.statistiche_descrittive()
            if stats is not None and not stats.empty:
                # Rinomina le colonne per la visualizzazione
                rename_cols = {
                    'media': 'Mean' if st.session_state.lingua == 'en' else 'media',
                    'mediana': 'Median' if st.session_state.lingua == 'en' else 'mediana',
                    'moda': 'Mode' if st.session_state.lingua == 'en' else 'moda',
                    'std': 'Std' if st.session_state.lingua == 'en' else 'std',
                    'min': 'Min' if st.session_state.lingua == 'en' else 'min',
                    'max': 'Max' if st.session_state.lingua == 'en' else 'max',
                    'n': 'N' if st.session_state.lingua == 'en' else 'n',
                    'mancanti': 'Missing' if st.session_state.lingua == 'en' else 'mancanti',
                }
                stats_display = stats.rename(columns=rename_cols)
                media_col = 'Mean' if st.session_state.lingua == 'en' else 'media'
                std_col = 'Std' if st.session_state.lingua == 'en' else 'std'
                n_col = 'N' if st.session_state.lingua == 'en' else 'n'
                st.dataframe(
                    stats_display.style.background_gradient(subset=[media_col], cmap='RdYlGn', vmin=1, vmax=5)
                    .format({media_col: '{:.2f}', std_col: '{:.2f}', n_col: '{:.0f}'}),
                    use_container_width=True, height=300
                )
            else:
                st.info(T['nessun_dato'])

    # ============================================================
    # TAB 2: ANALISI LIKERT
    # ============================================================
    with tab2:
        st.header(T['analisi_likert'])

        scale_disponibili = list(config['scale'].keys())
        scala_selezionata = st.selectbox(
            T['selezione_scala'], scale_disponibili,
            format_func=lambda x: config['scale'][x]['title'],
            key=f"scala_select_{st.session_state.lingua}"   # <-- key dinamica
        )

        if scala_selezionata:
            domande_scala = [col for col, s in analisi_likert.df_scale_info.items()
                             if s == scala_selezionata and col in config['domande_likert']]

            if domande_scala:
                with st.expander(T['statistiche'], expanded=True):
                    stats = analisi_likert.statistiche_descrittive()
                    stats_scala = stats[stats.index.isin([config['domande_likert'][d] for d in domande_scala])]

                    if not stats_scala.empty:
                        rename_cols = {
                            'media': 'Mean' if st.session_state.lingua == 'en' else 'media',
                            'mediana': 'Median' if st.session_state.lingua == 'en' else 'mediana',
                            'moda': 'Mode' if st.session_state.lingua == 'en' else 'moda',
                            'std': 'Std' if st.session_state.lingua == 'en' else 'std',
                            'min': 'Min' if st.session_state.lingua == 'en' else 'min',
                            'max': 'Max' if st.session_state.lingua == 'en' else 'max',
                            'n': 'N' if st.session_state.lingua == 'en' else 'n',
                            'mancanti': 'Missing' if st.session_state.lingua == 'en' else 'mancanti',
                        }
                        stats_scala_display = stats_scala.rename(columns=rename_cols)
                        media_col = 'Mean' if st.session_state.lingua == 'en' else 'media'
                        std_col = 'Std' if st.session_state.lingua == 'en' else 'std'
                        n_col = 'N' if st.session_state.lingua == 'en' else 'n'
                        st.dataframe(
                            stats_scala_display.style.background_gradient(subset=[media_col], cmap='RdYlGn', vmin=1, vmax=5)
                            .format({media_col: '{:.2f}', std_col: '{:.2f}', n_col: '{:.0f}'}),
                            use_container_width=True, height=250
                        )
                        csv = stats_scala.to_csv()
                        
                        st.download_button(
                            label=T['scarica_stats'], data=csv,
                            file_name=f"statistiche_{scala_selezionata}.csv",
                            mime="text/csv", key="download_stats"
                        )
                    else:
                        st.info(T['nessun_dato'])

                st.subheader(T['distribuzione'])
                if st.button(f"{T['genera_grafico']} - {config['scale'][scala_selezionata]['title']}", key="btn_likert"):
                    spinner_graf = ('Generating chart...' 
                                    if st.session_state.lingua == 'en' 
                                    else "Generando grafico...")
                    with st.spinner(spinner_graf):
                        domande_filtrate = {k: v for k, v in config['domande_likert'].items() if k in domande_scala}
                        crea_container_grafico("")
                        fig = generatore_grafici.grafico_likert(
                            analisi_likert.df_likert, domande_filtrate,
                            config['scale'][scala_selezionata], salva=False,
                            lingua=st.session_state.lingua
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info(T['nessun_dato'])
                        chiudi_container_grafico()

                st.subheader(T['alpha_cronbach'])
                col1, col2 = st.columns([1, 2])
                with col1:
                    alpha = analisi_likert.calcola_alpha_cronbach(scala_selezionata)
                    if alpha is None:
                        st.metric(T['alpha_cronbach'], "N/A")
                        st.warning("⚠️ " + ("Non calcolabile: dati insufficienti" if st.session_state.lingua == 'it' else "Not calculable: insufficient data"))
                    else:
                        st.metric(T['alpha_cronbach'], f"{alpha:.3f}")
                        if alpha >= 0.9:
                            st.success("✅ " + ("Eccellente!" if st.session_state.lingua == 'it' else "Excellent!"))
                        elif alpha >= 0.8:
                            st.success("✅ " + ("Buona" if st.session_state.lingua == 'it' else "Good"))
                        elif alpha >= 0.7:
                            st.info("ℹ️ " + ("Accettabile" if st.session_state.lingua == 'it' else "Acceptable"))
                        elif alpha >= 0.6:
                            st.warning("⚠️ " + ("Dubbia" if st.session_state.lingua == 'it' else "Questionable"))
                        else:
                            st.warning("⚠️ " + ("Da migliorare" if st.session_state.lingua == 'it' else "Needs improvement"))

                st.subheader(T['correlazione'])
                if st.button(T['mostra_correlazione'], key="btn_correlation"):
                    spinner_corr = ('Calculating correlation matrix...' 
                                    if st.session_state.lingua == 'en' 
                                    else "Calcolando la matrice di correlazione...")
                    with st.spinner(spinner_corr):
                        crea_container_grafico("")
                        titolo_corr = 'Correlation Matrix' if st.session_state.lingua == 'en' else 'Matrice di Correlazione'
                        fig = generatore_grafici.matrice_correlazione(
                            analisi_likert.df_likert,
                            {k: v for k, v in config['domande_likert'].items() if k in domande_scala},
                            titolo=titolo_corr,
                            salva=False,
                            lingua=st.session_state.lingua
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info(T['nessun_dato'])
                        chiudi_container_grafico()
            else:
                st.info(T['nessun_dato'])

    # ============================================================
    # TAB 3: ANALISI TESTO
    # ============================================================
    with tab3:
        st.header(T['analisi_testo'])

        if analisi_testo.df_testo:
            domanda_selezionata = st.selectbox(
                T['seleziona_domanda'],
                list(config['domande_aperte'].keys()),
                format_func=lambda x: config['domande_aperte'][x],
                key=f"domanda_aperta_select_{st.session_state.lingua}"   # <-- key dinamica
            )

            if domanda_selezionata in analisi_testo.df_testo:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(T['wordcloud'])
                    if st.button(f"{T['genera_grafico']} WordCloud", key="btn_wordcloud"):
                        with st.spinner("Generando word cloud..."):
                            crea_container_grafico("")
                            fig = analisi_testo.crea_wordcloud(domanda_selezionata, salva=False)
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info(T['nessun_dato'])
                            chiudi_container_grafico()

                with col2:
                    st.subheader(T['parole_frequenti'])
                    with st.spinner("Analizzando le parole..."):
                        parole = analisi_testo.parole_piu_frequenti(domanda_selezionata, n=15)
                        if parole:
                            df_parole = pd.DataFrame(
                                parole,
                                columns=['Parola' if st.session_state.lingua == 'it' else 'Word',
                                         'Frequenza' if st.session_state.lingua == 'it' else 'Frequency']
                            )
                            st.bar_chart(df_parole.set_index(df_parole.columns[0]))
                        else:
                            st.info(T['nessun_dato'])

                st.subheader(T['sentiment'])
                with st.spinner("Analizzando il sentiment..."):
                    sentimenti = analisi_testo.analisi_sentiment(domanda_selezionata)
                    if sentimenti:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(crea_metrica_card(T['positivo'], sentimenti.get('positivo', 0), "😊", "#43A047"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(crea_metrica_card(T['neutro'], sentimenti.get('neutro', 0), "😐", "#FB8C00"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(crea_metrica_card(T['negativo'], sentimenti.get('negativo', 0), "😞", "#E53935"), unsafe_allow_html=True)

                
                # --- Tabella di Priorità ---
                st.markdown("---")
                st.subheader("📋 " + ("Priority Table" if st.session_state.lingua == 'en' else "Tabella di Priorità"))
                

                if 'problemi_priorita' not in st.session_state:
                    st.session_state.problemi_priorita = None
                if 'priorita_domanda_corrente' not in st.session_state:
                    st.session_state.priorita_domanda_corrente = None

                if st.session_state.priorita_domanda_corrente != domanda_selezionata:
                    st.session_state.problemi_priorita = None
                    st.session_state.priorita_domanda_corrente = domanda_selezionata
                    
                st.info("ℹ️ " + (
                                                    "Questa tabella usa keyword matching sulle risposte aperte. "
                                                    "Per una versione basata sull'analisi tematica (più oggettiva), "
                                                    "vedi il tab '🤖 Analisi Tematica'."
                                                    if st.session_state.lingua == 'it' else
                                                    "This table uses keyword matching on open responses. "
                                                    "For a thematic-analysis-based version (more objective), "
                                                    "see the '🤖 Thematic Analysis' tab."
                                                ))

                if st.button(
                    "🔍 " + ("Analizza priorità" if st.session_state.lingua == 'it' else "Analyze priorities"),
                    key="btn_priorita"
                ):
                    with st.spinner("Analizzando i problemi..."):
                        temp_gen = ReportGenerator(
                            analisi_likert, analisi_testo, generatore_grafici,
                            config, st.session_state.lingua,
                            analisi_profilo=analisi_profilo
                        )
                        st.session_state.problemi_priorita = temp_gen._estrai_problemi_da_risposte()

                problemi = st.session_state.problemi_priorita

                if problemi is not None:
                    if problemi:
                        if st.session_state.lingua == 'it':
                            legenda_html = """
                            <div class="nielsen-legend">
                            <strong>📚 Legenda euristiche di Nielsen</strong>
                            <ul>
                                <li><strong>H2</strong> Corrispondenza sistema-mondo reale</li>
                                <li><strong>H5</strong> Prevenzione degli errori</li>
                                <li><strong>H6</strong> Riconoscimento invece che ricordo</li>
                                <li><strong>H7</strong> Flessibilità ed efficienza d'uso</li>
                                <li><strong>H8</strong> Design estetico e minimalista</li>
                                <li><strong>H9</strong> Recupero dagli errori</li>
                                <li><strong>H10</strong> Aiuto e documentazione</li>
                            </ul>
                            </div>
                            """
                        else:
                            legenda_html = """
                            <div class="nielsen-legend">
                            <strong>📚 Nielsen heuristics legend</strong>
                            <ul>
                                <li><strong>H2</strong> Match Between System and Real World</li>
                                <li><strong>H5</strong> Error Prevention</li>
                                <li><strong>H6</strong> Recognition Rather than Recall</li>
                                <li><strong>H7</strong> Flexibility and Efficiency of Use</li>
                                <li><strong>H8</strong> Aesthetic and Minimalist Design</li>
                                <li><strong>H9</strong> Help Users Recognize, Diagnose, and Recover from Errors</li>
                                <li><strong>H10</strong> Help and Documentation</li>
                            </ul>
                            </div>
                            """
                        st.markdown(legenda_html, unsafe_allow_html=True)

                        df_priorita = pd.DataFrame(problemi)
                        df_priorita['euristiche_str'] = df_priorita['euristiche_labels'].apply(
                            lambda lst: ' | '.join(lst) if isinstance(lst, list) else ''
                        )

                        def _label_priorita(p):
                            if p >= 7:
                                return '🔴 ' + ('Critica' if st.session_state.lingua == 'it' else 'Critical')
                            elif p >= 5:
                                return '🟠 ' + ('Alta' if st.session_state.lingua == 'it' else 'High')
                            elif p >= 4:
                                return '🟡 ' + ('Media' if st.session_state.lingua == 'it' else 'Medium')
                            else:
                                return '🟢 ' + ('Bassa' if st.session_state.lingua == 'it' else 'Low')

                        df_priorita['priorita_label'] = df_priorita['punteggio'].apply(_label_priorita)

                        rename_map = {
                            'problema': 'Problema' if st.session_state.lingua == 'it' else 'Issue',
                            'euristiche_str': 'Euristiche Nielsen',
                            'severita': 'Severità' if st.session_state.lingua == 'it' else 'Severity',
                            'frequenza_valore': 'Frequenza' if st.session_state.lingua == 'it' else 'Frequency',
                            'frequenza_pct': 'Freq. %',
                            'punteggio': 'Punteggio' if st.session_state.lingua == 'it' else 'Score',
                            'priorita_label': 'Priorità' if st.session_state.lingua == 'it' else 'Priority',
                        }
                        df_priorita = df_priorita.rename(columns=rename_map)

                        col_problema = 'Problema' if st.session_state.lingua == 'it' else 'Issue'
                        col_sev = 'Severità' if st.session_state.lingua == 'it' else 'Severity'
                        col_freq = 'Frequenza' if st.session_state.lingua == 'it' else 'Frequency'
                        col_punt = 'Punteggio' if st.session_state.lingua == 'it' else 'Score'
                        col_prio = 'Priorità' if st.session_state.lingua == 'it' else 'Priority'

                        colonne_mostrate = [
                            col_problema, 'Euristiche Nielsen',
                            col_sev, col_freq, 'Freq. %',
                            col_punt, col_prio,
                        ]

                        def _color_severita(val):
                            if val <= 1:
                                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif val == 2:
                                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                            elif val == 3:
                                return 'background-color: #ffe5b4; color: #8a4b00; font-weight: bold;'
                            else:
                                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'

                        def _color_frequenza(val):
                            if val == 1:
                                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif val == 2:
                                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                            elif val == 3:
                                return 'background-color: #ffe5b4; color: #8a4b00; font-weight: bold;'
                            else:
                                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'

                        def _color_punteggio(val):
                            if val <= 3:
                                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif val <= 5:
                                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                            elif val <= 6:
                                return 'background-color: #ffe5b4; color: #8a4b00; font-weight: bold;'
                            else:
                                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'

                        styled = (
                            df_priorita[colonne_mostrate]
                            .style
                            .map(_color_severita, subset=[col_sev])
                            .map(_color_frequenza, subset=[col_freq])
                            .map(_color_punteggio, subset=[col_punt])
                        )

                        st.dataframe(styled, hide_index=True, use_container_width=True)

                        csv_priorita = df_priorita[colonne_mostrate].to_csv(index=False)
                                                # --- Legenda "Come si legge il punteggio" ---
                        if st.session_state.lingua == 'it':
                            legenda_punteggio = """
                            <div style="background: #f8f9fa; border-left: 4px solid #1E88E5;
                                        padding: 14px 18px; margin: 12px 0; border-radius: 4px;
                                        font-size: 0.9em; line-height: 1.6;">
                                <strong>📖 Come si legge il punteggio di priorità</strong><br>
                                Il punteggio è calcolato come:
                                <strong>Severità + Frequenza</strong>.
                                Le risposte a tutte le domande aperte sono analizzate con pesi differenziati.
                                <br><br>
                                <strong>Severità</strong> (quanto è grave il problema):<br>
                                <span style="background:#d4edda; padding:2px 8px; border-radius:3px; color:#155724;">0–1 = Bassa</span>
                                <span style="background:#fff3cd; padding:2px 8px; border-radius:3px; color:#856404;">2 = Media</span>
                                <span style="background:#ffe5b4; padding:2px 8px; border-radius:3px; color:#8a4b00;">3 = Alta</span>
                                <span style="background:#f8d7da; padding:2px 8px; border-radius:3px; color:#721c24;">4 = Critica</span><br><br>
                                <strong>Frequenza</strong> (quanto è diffuso il problema tra le risposte):<br>
                                <span style="background:#d4edda; padding:2px 8px; border-radius:3px; color:#155724;">1 = Rara (&lt; 11%)</span>
                                <span style="background:#fff3cd; padding:2px 8px; border-radius:3px; color:#856404;">2 = Occasionale (11–50%)</span>
                                <span style="background:#ffe5b4; padding:2px 8px; border-radius:3px; color:#8a4b00;">3 = Frequente (51–89%)</span>
                                <span style="background:#f8d7da; padding:2px 8px; border-radius:3px; color:#721c24;">4 = Molto frequente (≥ 90%)</span><br><br>
                                <strong>Interpretazione del punteggio finale (range 1–8):</strong><br>
                                <span style="background:#d4edda; padding:2px 8px; border-radius:3px; color:#155724;">🟢 1–3 = Priorità bassa</span>
                                <span style="background:#fff3cd; padding:2px 8px; border-radius:3px; color:#856404;">🟡 4-5 = Priorità media</span>
                                <span style="background:#ffe5b4; padding:2px 8px; border-radius:3px; color:#8a4b00;">🟠 6 = Priorità alta</span>
                                <span style="background:#f8d7da; padding:2px 8px; border-radius:3px; color:#721c24;">🔴 7–8 = Priorità critica</span>
                            </div>
                            """
                        else:
                            legenda_punteggio = """
                            <div style="background: #f8f9fa; border-left: 4px solid #1E88E5;
                                        padding: 14px 18px; margin: 12px 0; border-radius: 4px;
                                        font-size: 0.9em; line-height: 1.6;">
                                <strong>📖 How to read the priority score</strong><br>
                                The score is computed as:
                                <code style="background:#eaf4fc; padding:2px 6px; border-radius:3px;">
                                    Score = Severity + Frequency
                                </code><br><br>
                                <strong>Severity</strong> (how serious the issue is):<br>
                                <span style="background:#d4edda; padding:2px 8px; border-radius:3px; color:#155724;">0–1 = Low</span>
                                <span style="background:#fff3cd; padding:2px 8px; border-radius:3px; color:#856404;">2 = Medium</span>
                                <span style="background:#ffe5b4; padding:2px 8px; border-radius:3px; color:#8a4b00;">3 = High</span>
                                <span style="background:#f8d7da; padding:2px 8px; border-radius:3px; color:#721c24;">4 = Critical</span><br><br>
                                <strong>Frequency</strong> (how widespread the issue is):<br>
                                <span style="background:#d4edda; padding:2px 8px; border-radius:3px; color:#155724;">1 = Rare (&lt; 11%)</span>
                                <span style="background:#fff3cd; padding:2px 8px; border-radius:3px; color:#856404;">2 = Occasional (11–50%)</span>
                                <span style="background:#ffe5b4; padding:2px 8px; border-radius:3px; color:#8a4b00;">3 = Frequent (51–89%)</span>
                                <span style="background:#f8d7da; padding:2px 8px; border-radius:3px; color:#721c24;">4 = Very frequent (≥ 90%)</span><br><br>
                                <strong>Interpretation of the final score (range 1–8):</strong><br>
                                <span style="background:#d4edda; padding:2px 8px; border-radius:3px; color:#155724;">🟢 1–3 = Low priority</span>
                                <span style="background:#fff3cd; padding:2px 8px; border-radius:3px; color:#856404;">🟡 4-5 = Medium priority</span>
                                <span style="background:#ffe5b4; padding:2px 8px; border-radius:3px; color:#8a4b00;">🟠 6 = High priority</span>
                                <span style="background:#f8d7da; padding:2px 8px; border-radius:3px; color:#721c24;">🔴 7–8 = Critical priority</span>
                            </div>
                            """
                        st.markdown(legenda_punteggio, unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 " + ("Scarica tabella (CSV)" if st.session_state.lingua == 'it' else "Download table (CSV)"),
                            data=csv_priorita,
                            file_name=f"priorita_manila_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="download_priorita"
                        )
                    else:
                        st.info("Nessun problema ricorrente identificato nelle risposte aperte.")

                # --- Visualizzazione risposte aperte ---
                st.markdown("---")
                with st.expander(
                    "📝 " + ("View all responses" if st.session_state.lingua == 'en' else "Visualizza tutte le risposte"),
                    expanded=False
                ):
                    risposte_con_sent = analisi_testo.risposte_con_sentiment(domanda_selezionata)
                    mostra_risposte_aperte(
                        risposte_con_sent,
                        max_mostra=20,
                        lingua=st.session_state.lingua
                    )
            else:
                st.info(T['nessuna_risposta'])
        else:
            st.info(T['nessuna_risposta'])

    # ============================================================
    # TAB ANALISI TEMATICA (MULTI-LLM) — NUOVO TAB DEDICATO
    # ============================================================
    with tab_tematica:
        st.header(T['analisi_tematica'])

        if st.session_state.lingua == 'it':
            st.markdown("""
            Questa sezione esegue un'**analisi tematica assistita da due LLM indipendenti**
            (coder A e coder B) sulle risposte aperte. I due modelli codificano ogni
            risposta secondo un vocabolario chiuso di 12 temi, poi i risultati vengono
            confrontati e armonizzati.
            """)
        else:
            st.markdown("""
            This section runs **thematic analysis assisted by two independent LLMs**
            (coder A and coder B) on open responses. Both models code each response
            using a closed vocabulary of 12 themes, then results are compared and
            harmonized.
            """)

        st.markdown("---")

        # --- Configurazione provider ---
        st.subheader("⚙️ " + ("Configurazione" if st.session_state.lingua == 'it'
                              else "Configuration"))

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Coder A**")
            provider_A = st.selectbox(
                "Provider A",
                list(PROVIDER_DISPONIBILI.keys()),
                format_func=lambda k: PROVIDER_DISPONIBILI[k]['label'],
                key=f"provider_A_tab_{st.session_state.lingua}",   # <-- key dinamica
                label_visibility="collapsed",
            )
            model_A = st.selectbox(
                "Model A",
                PROVIDER_DISPONIBILI[provider_A]['models'],
                key=f"model_A_tab_{st.session_state.lingua}",
                label_visibility="collapsed",
            )

        with col_b:
            st.markdown("**Coder B**")
            provider_B = st.selectbox(
                "Provider B",
                list(PROVIDER_DISPONIBILI.keys()),
                index=1,
                format_func=lambda k: PROVIDER_DISPONIBILI[k]['label'],
                key=f"provider_B_tab_{st.session_state.lingua}",
                label_visibility="collapsed",
            )
            model_B = st.selectbox(
                "Model B",
                PROVIDER_DISPONIBILI[provider_B]['models'],
                key=f"model_B_tab_{st.session_state.lingua}",
                label_visibility="collapsed",
            )

        # --- Verifica chiavi ---
        env_vars = set()
        for p in [provider_A, provider_B]:
            env = PROVIDER_DISPONIBILI[p]['env_var']
            if env:
                env_vars.add(env)
        if env_vars:
            mancanti = [e for e in env_vars if not os.getenv(e)]
            if mancanti:
                st.error("⚠️ " + ("Chiavi mancanti: " if st.session_state.lingua == 'it'
                                  else "Missing keys: ") + ", ".join(mancanti))
            else:
                st.success("✅ " + ("Chiavi configurate" if st.session_state.lingua == 'it'
                                    else "Keys configured"))

        # --- Pulsante avvio ---
                # --- Opzione: modalità async ---
        modalita_async = st.checkbox(
            "⚡ " + ("Usa codifica parallela (async)" if st.session_state.lingua == 'it'
                     else "Use parallel coding (async)"),
            value=True,
            key=f"modalita_async_{st.session_state.lingua}",
            help=("Esegue Coder A e Coder B in parallelo. Richiede i client async. "
                  "Disattiva se riscontri errori.")
            if st.session_state.lingua == 'it' else
            ("Runs Coder A and Coder B in parallel. Requires async clients. "
             "Disable if you encounter errors."),
        )

        if st.button(
            "▶️ " + ("Avvia codifica" if st.session_state.lingua == 'it' else "Run coding"),
            key="btn_llm_codifica_tab",
            use_container_width=True,
        ):
            with st.spinner(
                "Codifica in corso con 2 LLM..." if st.session_state.lingua == 'it'
                else "Coding with 2 LLMs..."
            ):
                try:
                    analisi_tem = AnalisiTematicaLLM(
                        analisi_testo.df_testo,
                        config,
                        None,   # verranno assegnati sotto
                        None,
                        lingua=st.session_state.lingua,
                    )

                    if modalita_async:
                        # --- Codifica parallela (async) ---
                        chiama_A = crea_client_async(
                            provider=provider_A,
                            model=model_A,
                            log_file="llm_log.jsonl",
                        )
                        chiama_B = crea_client_async(
                            provider=provider_B,
                            model=model_B,
                            log_file="llm_log.jsonl",
                        )
                        analisi_tem.chiama_A = chiama_A
                        analisi_tem.chiama_B = chiama_B

                        # Workaround Streamlit per asyncio
                        esegui_coroutine(
                            analisi_tem.codifica_tutte_async(
                                batch_size=15, verbose=False
                            )
                        )
                    else:
                        # --- Codifica sequenziale (sincrona) ---
                        chiama_A = crea_client(
                            provider=provider_A,
                            model=model_A,
                            log_file="llm_log.jsonl",
                        )
                        chiama_B = crea_client(
                            provider=provider_B,
                            model=model_B,
                            log_file="llm_log.jsonl",
                            pausa=0.1,
                        )
                        analisi_tem.chiama_A = chiama_A
                        analisi_tem.chiama_B = chiama_B

                        analisi_tem.codifica_tutte(verbose=False, batch_size=15)

                    # --- Post-processing (uguale in entrambe le modalità) ---
                    analisi_tem.confronta()
                    analisi_tem.risolvi_disaccordi(strategia='unione')
                    st.session_state.analisi_tematica = analisi_tem
                    st.success(
                        "✅ Codifica completata!" if st.session_state.lingua == 'it'
                        else "✅ Coding completed!"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {type(e).__name__}: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # --- Risultati (se disponibili) ---
        analisi_tem = st.session_state.get('analisi_tematica')
        if analisi_tem is not None:
            st.markdown("---")

            # --- Accordo inter-coder ---
            st.subheader("📊 " + ("Accordo inter-coder" if st.session_state.lingua == 'it'
                                  else "Inter-coder agreement"))
            df_acc = pd.DataFrame([
                {
                    ('Domanda' if st.session_state.lingua == 'it' else 'Question'): 
                        config['domande_aperte'].get(k, k),
                    'N': v['n'],
                    ('Accordo %' if st.session_state.lingua == 'it' else 'Agreement %'): 
                        v['accordo_pct'],
                    "Cohen's κ": v['kappa'],
                }
                for k, v in analisi_tem.accordo.items()
            ])
            st.dataframe(df_acc, hide_index=True, use_container_width=True)

            # --- Temi rilevati ---
            st.subheader("🎯 " + ("Temi rilevati" if st.session_state.lingua == 'it'
                                  else "Detected themes"))
            riassunto = analisi_tem.riassunto_per_tema()
            sentiment_map = {
                'positivo': 'Positive' if st.session_state.lingua == 'en' else 'Positivo',
                'negativo': 'Negative' if st.session_state.lingua == 'en' else 'Negativo',
                'neutro':   'Neutral'  if st.session_state.lingua == 'en' else 'Neutro',
            }
            df_temi = pd.DataFrame([
                {
                    ('Tema' if st.session_state.lingua == 'it' else 'Theme'): 
                        nome_tema(k, st.session_state.lingua),
                    ('Frequenza' if st.session_state.lingua == 'it' else 'Frequency'): 
                        v['frequenza'],
                    'Sentiment': sentiment_map.get(
                        v['sentiment_prevalente'], v['sentiment_prevalente']
                    ),
                }
                for k, v in sorted(riassunto.items(), key=lambda x: -x[1]['frequenza'])
            ])
            st.dataframe(df_temi, hide_index=True, use_container_width=True)

                        # --- Risposte per tema, divise per sentiment ---
            st.subheader("🚨 " + ("Risposte per tema"
                                  if st.session_state.lingua == 'it'
                                  else "Responses by theme"))
            riepilogo_full = analisi_tem.riepilogo_per_tema_con_risposte(config=config)

            # Colori per sentiment
            stili_sent = {
                'positivo': {'bg': '#E8F5E9', 'border': '#43A047',
                             'label': '😊 Positivo' if st.session_state.lingua == 'it' else '😊 Positive'},
                'neutro':   {'bg': '#FFF3E0', 'border': '#FB8C00',
                             'label': '😐 Neutro' if st.session_state.lingua == 'it' else '😐 Neutral'},
                'negativo': {'bg': '#FFEBEE', 'border': '#E53935',
                             'label': '😞 Negativo' if st.session_state.lingua == 'it' else '😞 Negative'},
            }
            ordine_sent = ['negativo', 'neutro', 'positivo']

            if riepilogo_full:
                # Ordina temi per frequenza decrescente
                temi_ordinati = sorted(
                    riepilogo_full.items(),
                    key=lambda x: -x[1]['frequenza']
                )
                for tema, dati in temi_ordinati:
                    n_tot = dati['frequenza']
                    dist = dati['distribuzione_sentiment']
                    n_neg = dist.get('negativo', 0)
                    n_neu = dist.get('neutro', 0)
                    n_pos = dist.get('positivo', 0)

                    # Titolo expander con conteggio per sentiment
                    titolo_exp = (
                        f"{nome_tema(tema, st.session_state.lingua)} "
                        f"({n_tot}) — "
                        f"😞 {n_neg} · 😐 {n_neu} · 😊 {n_pos}"
                    )
                    with st.expander(titolo_exp):
                        for sent in ordine_sent:
                            risposte_sent = dati['risposte'].get(sent, [])
                            if not risposte_sent:
                                continue
                            stile = stili_sent[sent]
                            st.markdown(
                                f"<div style='margin:8px 0;'>"
                                f"<span style='background:{stile['border']}; color:white; "
                                f"padding:2px 10px; border-radius:12px; font-size:12px; "
                                f"font-weight:500;'>{stile['label']} ({len(risposte_sent)})</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            for it in risposte_sent:
                                st.markdown(
                                    f"<div style='background:{stile['bg']}; padding:8px; "
                                    f"border-left:4px solid {stile['border']}; margin:4px 0; "
                                    f"border-radius:4px;'>"
                                    f"<small><strong>[{it['domanda']}]</strong> "
                                    f"{it['testo']}</small></div>",
                                    unsafe_allow_html=True,
                                )
            else:
                st.info("Nessuna risposta codificata."
                        if st.session_state.lingua == 'it'
                        else "No coded responses.")
           
                        
                            # --- Tabella Priorità basata su analisi tematica ---
            st.markdown("---")
            st.subheader("🎯 " + ("Tabella Priorità (da analisi tematica)"
                                  if st.session_state.lingua == 'it'
                                  else "Priority Table (from thematic analysis)"))
            with st.expander("📖 " + ("Come si legge la tabella priorità tematica"
                                                if st.session_state.lingua == 'it'
                                                else "How to read the thematic priority table")):
                            if st.session_state.lingua == 'it':
                                st.markdown("""
                                **Punteggio = Severità + Frequenza + Bonus sentiment**
            
                                - **Severità (0–4)**: quanto è grave il tema secondo le euristiche di Nielsen.
                                Mappatura fissa tema → euristica (es. `stabilita` = 4, `documentazione` = 2).
                                - **Frequenza (1–4)**: in che % di risposte codificate appare il tema.
                                - ≥ 30% → 4
                                - 15–29% → 3
                                - 5–14% → 2
                                - < 5% → 1
                                - **Bonus sentiment (0–1)**: +1 se almeno il 60% delle risposte sul tema
                                ha sentiment negativo.
            
                                **Interpretazione del punteggio (range 1–9):**
                                - 🟢 1–3 → Priorità bassa
                                - 🟡 4-5 → Priorità media
                                - 🟠 6 → Priorità alta
                                - 🔴 7–9 → Priorità critica
            
                                ⚠️ Il tema `punto_forza` e il tema `non_codificabile` sono **esclusi**
                                dalla tabella (non rappresentano problemi).
                                """)
                            else:
                                st.markdown("""
                                **Score = Severity + Frequency + Sentiment bonus**
            
                                - **Severity (0–4)**: how serious the theme is according to Nielsen's heuristics.
                                Fixed mapping theme → heuristic (e.g. `stabilita` = 4, `documentazione` = 2).
                                - **Frequency (1–4)**: what % of coded responses mention the theme.
                                - ≥ 30% → 4
                                - 15–29% → 3
                                - 5–14% → 2
                                - < 5% → 1
                                - **Sentiment bonus (0–1)**: +1 if at least 60% of the theme's responses
                                have negative sentiment.
            
                                **Score interpretation (range 1–9):**
                                - 🟢 1–3 → Low priority
                                - 🟡 4-5 → Medium priority
                                - 🟠 6 → High priority
                                - 🔴 7–9 → Critical priority
            
                                ⚠️ Themes `punto_forza` and `non_codificabile` are **excluded**
                                from the table (they do not represent issues).
                                """)

            priorita_temi = analisi_tem.priorita_da_temi()

            if priorita_temi:
                df_prio = pd.DataFrame([
                    {
                        ('Tema' if st.session_state.lingua == 'it' else 'Theme'): p['nome_tema'],
                        ('Euristica' if st.session_state.lingua == 'it' else 'Heuristic'): 
                            p['euristica'],
                        ('Severità' if st.session_state.lingua == 'it' else 'Severity'): 
                            p['severita'],
                        ('Frequenza' if st.session_state.lingua == 'it' else 'Frequency'): 
                            f"{p['frequenza_valore']} ({p['frequenza_pct']}%)",
                        ('Sent. neg. %' if st.session_state.lingua == 'it' else 'Neg. sent. %'): 
                            f"{p['sentiment_negativo_pct']}%",
                        ('Bonus' if st.session_state.lingua == 'it' else 'Bonus'): 
                            p['bonus_sentiment'],
                        ('Punteggio' if st.session_state.lingua == 'it' else 'Score'): 
                            p['punteggio'],
                        ('Priorità' if st.session_state.lingua == 'it' else 'Priority'):  (
                            '🔴 Critica' if p['punteggio'] >= 7 else
                            '🟠 Alta'    if p['punteggio'] >= 6 else
                            '🟡 Media'   if p['punteggio'] >= 4 else
                            '🟢 Bassa'
                        ) if st.session_state.lingua == 'it' else (
                            '🔴 Critical' if p['punteggio'] >= 7 else
                            '🟠 High'     if p['punteggio'] >= 6 else
                            '🟡 Medium'   if p['punteggio'] >= 4 else
                            '🟢 Low'
                        )
                    }
                    for p in priorita_temi
                ])
                st.dataframe(df_prio, hide_index=True, use_container_width=True)


            # --- Download JSON ---
            st.markdown("---")
            st.download_button(
                "📥 " + ("Scarica risultati (JSON)" if st.session_state.lingua == 'it'
                         else "Download results (JSON)"),
                data=json.dumps({
                    'accordo': analisi_tem.accordo,
                    'riassunto': analisi_tem.riassunto_per_tema(),
                    'codifica_finale': analisi_tem.codifica_finale,
                }, ensure_ascii=False, indent=2, default=str),
                file_name="analisi_tematica.json",
                mime="application/json",
                key="download_tematica_tab",
            )
        else:
            st.info("ℹ️ " + ("Nessuna codifica eseguita. Clicca 'Avvia codifica' per iniziare."
                             if st.session_state.lingua == 'it'
                             else "No coding performed yet. Click 'Run coding' to start."))

    # ============================================================
    # TAB 4: DATI GREZZI
    # ============================================================
    with tab4:
        st.header(T['dati_grezzi'])
        st.info("ℹ️ " + ("Visualizza tutti i dati del questionario" if st.session_state.lingua == 'it' else "View all questionnaire data"))

        st.dataframe(df, use_container_width=True, height=500)

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 " + ("Scarica dati (CSV)" if st.session_state.lingua == 'it' else "Download data (CSV)"),
                data=csv,
                file_name=f"dati_questionario_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", key="download_data"
            )
        with col2:
            if st.session_state.lingua == 'it':
                info_html = f"""
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 4px solid #1E88E5;">
                    <strong>📊 Info Dataset</strong><br>
                    Righe: {len(df)}<br>
                    Colonne: {len(df.columns)}<br>
                    Dimensione: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
                </div>
                """
            else:
                info_html = f"""
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 4px solid #1E88E5;">
                    <strong>📊 Dataset Info</strong><br>
                    Rows: {len(df)}<br>
                    Columns: {len(df.columns)}<br>
                    Size: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
                </div>
                """
            st.markdown(info_html, unsafe_allow_html=True)

    # FOOTER
    st.markdown(f"""
        <div class="footer">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; max-width: 800px; margin: 0 auto;">
                <span>📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
                <span>📊 MANILA Analysis Tool v2.0</span>
                <span>🔧 Powered by Streamlit</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# REPORT NELLA SIDEBAR
# ============================================================================
# ============================================================================
# REPORT NELLA SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📄 Report")

    # Controlla se l'analisi tematica è stata eseguita
    tematica_pronta = (
        st.session_state.get('analisi_tematica') is not None
    )

    # Pulsante disabilitato finché non c'è la codifica
    if st.button(
        testo_bottone_genera,
        use_container_width=True,
        key="btn_report",
        disabled=not tematica_pronta,
    ):
        with st.spinner(testo_spinner):
            try:
                progress_bar = st.progress(0)
                progress_bar.progress(30)

                report_gen = ReportGenerator(
                    analisi_likert,
                    analisi_testo,
                    generatore_grafici,
                    config,
                    st.session_state.lingua,
                    analisi_profilo=analisi_profilo,
                    analisi_tematica=st.session_state.get('analisi_tematica'),
                )

                progress_bar.progress(60)
                html_content = report_gen.genera_html()
                progress_bar.progress(90)

                b64 = base64.b64encode(html_content.encode()).decode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                href = f'''
                    <a href="data:text/html;base64,{b64}"
                       download="report_questionario_{timestamp}.html"
                       style="display: inline-block;
                              background: linear-gradient(135deg, #43A047, #2E7D32);
                              color: white; padding: 12px 20px;
                              text-align: center; text-decoration: none;
                              border-radius: 8px; margin: 5px 0;
                              width: 100%; font-weight: 500;">
                        {testo_bottone_download}
                    </a>
                '''
                st.sidebar.markdown(href, unsafe_allow_html=True)

                progress_bar.progress(100)
                st.sidebar.success(testo_successo)

                import time
                time.sleep(2)
                progress_bar.empty()

            except Exception as e:
                st.sidebar.error(f"{testo_errore} {str(e)}")
                st.sidebar.info(testo_info)

    # --- Messaggio informativo sullo stato della codifica ---
    if not tematica_pronta:
        st.info(
            "⚠️ Esegui prima la codifica nel tab '🤖 Analisi Tematica'"
            if st.session_state.lingua == 'it'
            else "⚠️ First run the coding in the '🤖 Thematic Analysis' tab"
        )
    else:
        st.success(
            "✅ Pronto per generare il report"
            if st.session_state.lingua == 'it'
            else "✅ Ready to generate the report"
        )