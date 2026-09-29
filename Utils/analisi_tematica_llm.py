# utils/analisi_tematica_llm.py
"""
Analisi tematica assistita da 2 LLM indipendenti (coder A, coder B).

Vocabolario: 20 temi specifici per l'applicazione MANILA
- 13 temi-problema
- 6 temi-forza
- 1 tema neutro (non_codificabile)

Output:
- codifica per risposta (A, B, finale)
- accordo inter-coder (Cohen's κ multi-label)
- risoluzione conflitti (strategia 'unione')
- riassunti per tema + risposte per sentiment
- priorità con mappatura tema → euristica Nielsen
- problemi con soluzioni proposte
"""
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# VOCABOLARIO TEMI BILINGUE (20 temi)
# Ogni tema ha: nome, definizione, esempi, categoria
# I temi-problema hanno anche: soluzione, euristica, severita
# ============================================================
TEMI = {

    # --------------------------------------------------------
    # 🔴 TEMI-PROBLEMA (13)
    # --------------------------------------------------------
    'performance_lentezza': {
        'categoria': 'problema',
        'euristica': 'H1 · Visibilità stato',
        'severita': 3,
        'soluzione': {
            'it': 'Ottimizzare le performance (caching, parallelizzazione, uso GPU) '
                  'e mostrare una progress bar con stima dei tempi di completamento.',
            'en': 'Optimize performance (caching, parallelization, GPU usage) '
                  'and show a progress bar with estimated completion times.',
        },
        'it': {
            'nome': 'Lentezza / tempi di attesa',
            'definizione': 'Il sistema è lento nel caricamento dataset, training o benchmark',
            'esempi': ['very slow', 'takes too long', 'waiting time', 'lento', 'attesa'],
        },
        'en': {
            'nome': 'Slowness / waiting times',
            'definizione': 'The system is slow when loading datasets, training, or benchmarking',
            'esempi': ['very slow', 'takes too long', 'waiting time', 'lag', 'sluggish'],
        },
    },

    'interfaccia_confusa': {
        'categoria': 'problema',
        'euristica': 'H8 · Design minimalista + H2',
        'severita': 3,
        'soluzione': {
            'it': 'Riorganizzare la UI con navigazione più chiara, sostituire radio button '
                  'ambigui con checkbox multipli quando serve, aggiungere tooltip contestuali.',
            'en': 'Reorganize the UI with clearer navigation, replace ambiguous radio buttons '
                  'with multi-select checkboxes when needed, add contextual tooltips.',
        },
        'it': {
            'nome': 'Interfaccia poco chiara / ambigua',
            'definizione': 'Navigazione confusa, controlli ambigui, layout poco chiaro',
            'esempi': ['confusing', 'unclear', 'radio buttons', 'layout', 'non capisco dove'],
        },
        'en': {
            'nome': 'Unclear / ambiguous interface',
            'definizione': 'Confusing navigation, ambiguous controls, unclear layout',
            'esempi': ['confusing', 'unclear', 'radio buttons mutually exclusive',
                       'unclear layout', "don't understand where"],
        },
    },

    'documentazione_insufficiente': {
        'categoria': 'problema',
        'euristica': 'H10 · Aiuto e documentazione',
        'severita': 2,
        'soluzione': {
            'it': 'Ampliare la documentazione con: tutorial passo-passo, spiegazioni contestuali '
                  'in-platform (tooltip, help menu), descrizioni delle metriche e dei modelli.',
            'en': 'Expand documentation with: step-by-step tutorials, contextual in-platform '
                  'explanations (tooltips, help menu), metric and model descriptions.',
        },
        'it': {
            'nome': 'Documentazione / guidance insufficiente',
            'definizione': 'Mancano guide, tooltip, help menu, spiegazioni contestuali',
            'esempi': ['missing documentation', 'no help menu', 'no tooltips',
                       'short explanations', 'educational guidance'],
        },
        'en': {
            'nome': 'Insufficient documentation / guidance',
            'definizione': 'Missing guides, tooltips, help menu, contextual explanations',
            'esempi': ['missing documentation', 'no help menu', 'no tooltips',
                       'short explanations', 'educational guidance'],
        },
    },

    'messaggi_errore_poco_chiari': {
        'categoria': 'problema',
        'euristica': 'H9 · Recupero errori',
        'severita': 3,
        'soluzione': {
            'it': 'Riscrivere i messaggi di errore con causa, contesto e azione suggerita '
                  'per risolvere il problema.',
            'en': 'Rewrite error messages with cause, context, and suggested action '
                  'to solve the problem.',
        },
        'it': {
            'nome': 'Messaggi di errore poco chiari',
            'definizione': 'Errori senza causa, contesto o azione suggerita',
            'esempi': ['error message', "don't understand the error", 'no explanation',
                       'non capisco l\'errore'],
        },
        'en': {
            'nome': 'Unclear error messages',
            'definizione': 'Errors without cause, context, or suggested action',
            'esempi': ['error message', "don't understand the error",
                       'no explanation', 'unclear error'],
        },
    },

    'configurazione_complessa': {
        'categoria': 'problema',
        'euristica': "H7 · Flessibilità d'uso",
        'severita': 3,
        'soluzione': {
            'it': 'Introdurre preset di configurazione, valori di default intelligenti '
                  'e suggerimenti contestuali per ridurre le scelte richieste.',
            'en': 'Introduce configuration presets, smart defaults, and contextual hints '
                  'to reduce the number of required choices.',
        },
        'it': {
            'nome': 'Configurazione complessa',
            'definizione': 'Troppi parametri, difficile orientarsi nelle scelte',
            'esempi': ['too many parameters', 'complex configuration', 'complicated setup',
                       'difficile configurare'],
        },
        'en': {
            'nome': 'Complex configuration',
            'definizione': 'Too many parameters, difficult to navigate the choices',
            'esempi': ['too many parameters', 'complex configuration', 'complicated setup',
                       'hard to configure'],
        },
    },

    'mancanza_export_report': {
        'categoria': 'problema',
        'euristica': "H7 · Flessibilità d'uso",
        'severita': 3,
        'soluzione': {
            'it': 'Garantire che lo script Python generato sia completo ed eseguibile '
                  '(import, funzioni core implementate). Aggiungere download dello script '
                  'post-esperimento. Aggiungere etichette agli assi nei grafici.',
            'en': 'Ensure the generated Python script is complete and runnable '
                  '(imports, core functions implemented). Add post-experiment script download. '
                  'Add axis labels to charts.',
        },
        'it': {
            'nome': 'Export / script Python problematici',
            'definizione': 'Export mancante, script Python generato non funzionante, report incompleti',
            'esempi': ['script not working', 'missing import', 'cannot download',
                       'export', 'report'],
        },
        'en': {
            'nome': 'Problematic export / Python script',
            'definizione': 'Missing export, non-functional generated Python script, incomplete reports',
            'esempi': ['script not working', 'missing import', 'cannot download',
                       'export', 'report'],
        },
    },

    'mancanza_modelli_ml': {
        'categoria': 'problema',
        'euristica': "H7 · Flessibilità d'uso",
        'severita': 3,
        'soluzione': {
            'it': 'Integrare XGBoost, LightGBM e modelli deep learning per ampliare '
                  'il set di modelli disponibili.',
            'en': 'Integrate XGBoost, LightGBM, and deep learning models to expand '
                  'the set of available models.',
        },
        'it': {
            'nome': 'Modelli ML mancanti',
            'definizione': 'Richiesta di modelli ML aggiuntivi (boosting, deep learning, ecc.)',
            'esempi': ['XGBoost', 'LightGBM', 'deep learning', 'missing model'],
        },
        'en': {
            'nome': 'Missing ML models',
            'definizione': 'Request for additional ML models (boosting, deep learning, etc.)',
            'esempi': ['XGBoost', 'LightGBM', 'deep learning', 'missing model'],
        },
    },

    'mancanza_metodi_fairness': {
        'categoria': 'problema',
        'euristica': "H7 · Flessibilità d'uso",
        'severita': 3,
        'soluzione': {
            'it': 'Integrare Adversarial Debiasing, Exponentiated Gradient, KNN, '
                  'Gaussian Process e MetaFAIR Classifier.',
            'en': 'Integrate Adversarial Debiasing, Exponentiated Gradient, KNN, '
                  'Gaussian Process, and MetaFAIR Classifier.',
        },
        'it': {
            'nome': 'Metodi/metriche fairness mancanti',
            'definizione': 'Richiesta di metodi o metriche di fairness aggiuntivi',
            'esempi': ['Adversarial Debiasing', 'MetaFAIR', 'KNN', 'Gaussian Process',
                       'missing fairness method'],
        },
        'en': {
            'nome': 'Missing fairness methods/metrics',
            'definizione': 'Request for additional fairness methods or metrics',
            'esempi': ['Adversarial Debiasing', 'MetaFAIR', 'KNN', 'Gaussian Process',
                       'missing fairness method'],
        },
    },

    'incompatibilita_componenti': {
        'categoria': 'problema',
        'euristica': 'H5 · Prevenzione errori',
        'severita': 3,
        'soluzione': {
            'it': 'Rendere espliciti i vincoli di compatibilità modello↔metodo, '
                  'suggerendo automaticamente le combinazioni valide.',
            'en': 'Make model↔method compatibility constraints explicit, '
                  'automatically suggesting valid combinations.',
        },
        'it': {
            'nome': 'Vincoli di compatibilità',
            'definizione': 'Modello e metodo non compatibili, blocchi nella configurazione',
            'esempi': ['not compatible', 'incompatible', 'constraints', 'vincoli'],
        },
        'en': {
            'nome': 'Component compatibility constraints',
            'definizione': 'Model and method not compatible, configuration blocks',
            'esempi': ['not compatible', 'incompatible', 'constraints'],
        },
    },

    'risultati_inattesi': {
        'categoria': 'problema',
        'euristica': 'H2 · Corrispondenza sistema-mondo',
        'severita': 3,
        'soluzione': {
            'it': 'Rendere trasparente il calcolo del trade-off score: mostrare '
                  'normalizzazione e trasformazione delle metriche, con spiegazione '
                  'delle direzioni di ottimizzazione.',
            'en': 'Make trade-off score computation transparent: show metric normalization '
                  'and transformation, with explanations of optimization directions.',
        },
        'it': {
            'nome': 'Risultati inattesi / poca trasparenza',
            'definizione': 'Discrepanze tra attesi e prodotti, trade-off score non trasparente',
            'esempi': ['discrepancy', 'unexpected', 'not transparent', 'trade-off score',
                       'discrepanza', 'non trasparente'],
        },
        'en': {
            'nome': 'Unexpected results / lack of transparency',
            'definizione': 'Discrepancies between expected and produced, non-transparent trade-off score',
            'esempi': ['discrepancy', 'unexpected', 'not transparent', 'trade-off score'],
        },
    },

    'stabilita_crash': {
        'categoria': 'problema',
        'euristica': 'H5 · Prevenzione errori',
        'severita': 4,
        'soluzione': {
            'it': 'Aggiungere auto-save dello stato, riconnessione automatica '
                  'e logging diagnostico per i crash.',
            'en': 'Add state auto-save, automatic reconnection, '
                  'and diagnostic logging for crashes.',
        },
        'it': {
            'nome': 'Crash / instabilità',
            'definizione': 'Crash, perdita connessione, errori improvvisi, bug',
            'esempi': ['crash', 'lost connection', 'bug', 'freezes', 'si blocca'],
        },
        'en': {
            'nome': 'Crashes / instability',
            'definizione': 'Crashes, connection loss, sudden errors, bugs',
            'esempi': ['crash', 'lost connection', 'bug', 'freezes', 'hangs'],
        },
    },

    'curva_apprendimento': {
        'categoria': 'problema',
        'euristica': 'H6 · Riconoscimento vs ricordo',
        'severita': 2,
        'soluzione': {
            'it': 'Fornire tutorial interattivo iniziale, esempi di workflow completi '
                  'e suggerimenti contestuali.',
            'en': 'Provide an interactive initial tutorial, complete workflow examples, '
                  'and contextual suggestions.',
        },
        'it': {
            'nome': 'Curva di apprendimento',
            'definizione': 'Difficoltà per utenti non esperti, onboarding complesso',
            'esempi': ['difficult to learn', 'steep learning curve', 'complicated for beginners',
                       'difficile imparare'],
        },
        'en': {
            'nome': 'Learning curve',
            'definizione': 'Difficulty for non-expert users, complex onboarding',
            'esempi': ['difficult to learn', 'steep learning curve', 'complicated for beginners'],
        },
    },

    'limiti_dataset': {
        'categoria': 'problema',
        'euristica': "H7 · Flessibilità d'uso",
        'severita': 2,
        'soluzione': {
            'it': 'Supportare più formati, dataset più grandi (streaming) '
                  'e validazione automatica all\'upload.',
            'en': 'Support more formats, larger datasets (streaming), '
                  'and automatic validation on upload.',
        },
        'it': {
            'nome': 'Limiti caricamento dataset',
            'definizione': 'Problemi con formati, dimensioni o procedure di caricamento',
            'esempi': ['dataset too large', 'unsupported format', 'upload failed',
                       'formato non supportato'],
        },
        'en': {
            'nome': 'Dataset loading limits',
            'definizione': 'Issues with formats, sizes, or upload procedures',
            'esempi': ['dataset too large', 'unsupported format', 'upload failed'],
        },
    },

    # --------------------------------------------------------
    # 🟢 TEMI-FORZA (6)
    # --------------------------------------------------------
    'punto_forza_ui_intuitiva': {
        'categoria': 'forza',
        'it': {
            'nome': 'UI intuitiva',
            'definizione': 'Interfaccia facile, chiara, low-code, accessibile',
            'esempi': ['intuitive UI', 'easy to use', 'low-code', 'simple interface',
                       'user-friendly'],
        },
        'en': {
            'nome': 'Intuitive UI',
            'definizione': 'Easy, clear, low-code, accessible interface',
            'esempi': ['intuitive UI', 'easy to use', 'low-code', 'simple interface',
                       'user-friendly'],
        },
    },

    'punto_forza_flessibilita_config': {
        'categoria': 'forza',
        'it': {
            'nome': 'Flessibilità di configurazione',
            'definizione': 'Alta configurabilità, molte opzioni, adattabilità',
            'esempi': ['flexibility in configuration', 'high configurability',
                       'customization', 'flessibilità'],
        },
        'en': {
            'nome': 'Configuration flexibility',
            'definizione': 'High configurability, many options, adaptability',
            'esempi': ['flexibility in configuration', 'high configurability',
                       'customization', 'flexibility'],
        },
    },

    'punto_forza_workflow_guidato': {
        'categoria': 'forza',
        'it': {
            'nome': 'Workflow guidato step-by-step',
            'definizione': 'Struttura chiara, processo guidato, step-by-step',
            'esempi': ['step-by-step workflow', 'structured process', 'guided workflow',
                       'workflow guidato'],
        },
        'en': {
            'nome': 'Guided step-by-step workflow',
            'definizione': 'Clear structure, guided process, step-by-step',
            'esempi': ['step-by-step workflow', 'structured process', 'guided workflow'],
        },
    },

    'punto_forza_ampiezza_componenti': {
        'categoria': 'forza',
        'it': {
            'nome': 'Ampiezza componenti',
            'definizione': 'Buona scelta di modelli ML e metodi fairness disponibili',
            'esempi': ['wide range of models', 'many fairness methods',
                       'comprehensive', 'ampia scelta'],
        },
        'en': {
            'nome': 'Wide range of components',
            'definizione': 'Good selection of ML models and fairness methods available',
            'esempi': ['wide range of models', 'many fairness methods',
                       'comprehensive', 'wide selection'],
        },
    },

    'punto_forza_risultati_chiari': {
        'categoria': 'forza',
        'it': {
            'nome': 'Chiarezza dei risultati',
            'definizione': 'Output, grafici e metriche chiare e facili da interpretare',
            'esempi': ['clear results', 'useful metrics', 'easy to interpret',
                       'risultati chiari'],
        },
        'en': {
            'nome': 'Clear results',
            'definizione': 'Clear, easy-to-interpret outputs, charts, and metrics',
            'esempi': ['clear results', 'useful metrics', 'easy to interpret'],
        },
    },

    'punto_forza_generico': {
        'categoria': 'forza',
        'it': {
            'nome': 'Punto di forza generico',
            'definizione': 'Elogio senza specificare cosa funziona bene',
            'esempi': ['great application', 'very useful', 'excellent tool', 'ottimo'],
        },
        'en': {
            'nome': 'Generic strength',
            'definizione': 'Praise without specifying what works well',
            'esempi': ['great application', 'very useful', 'excellent tool'],
        },
    },

    # --------------------------------------------------------
    # ⚪ NEUTRO (1)
    # --------------------------------------------------------
    'non_codificabile': {
        'categoria': 'neutro',
        'it': {
            'nome': 'Non codificabile',
            'definizione': 'Risposta vuota, irrilevante o non informativa',
            'esempi': ['niente', 'non saprei', '', 'no'],
        },
        'en': {
            'nome': 'Non-codable',
            'definizione': 'Empty, irrelevant, or non-informative response',
            'esempi': ['nothing', "I don't know", '', 'no'],
        },
    },
}

CODICI_VALIDI = set(TEMI.keys())

# ============================================================
# EURISTICHE NIELSEN — versioni bilingui
# ============================================================
EURISTICHE = {
    'H1':  {'it': 'H1 · Visibilità stato',
            'en': 'H1 · Visibility of system status'},
    'H2':  {'it': 'H2 · Corrispondenza sistema-mondo',
            'en': 'H2 · Match system / real world'},
    'H3':  {'it': 'H3 · Controllo utente',
            'en': 'H3 · User control and freedom'},
    'H4':  {'it': 'H4 · Consistenza',
            'en': 'H4 · Consistency and standards'},
    'H5':  {'it': 'H5 · Prevenzione errori',
            'en': 'H5 · Error prevention'},
    'H6':  {'it': 'H6 · Riconoscimento vs ricordo',
            'en': 'H6 · Recognition vs recall'},
    'H7':  {'it': "H7 · Flessibilità d'uso",
            'en': 'H7 · Flexibility and efficiency'},
    'H8':  {'it': 'H8 · Design minimalista + H2',
            'en': 'H8 · Minimalist design + H2'},
    'H9':  {'it': 'H9 · Recupero errori',
            'en': 'H9 · Error recovery'},
    'H10': {'it': 'H10 · Aiuto e documentazione',
            'en': 'H10 · Help and documentation'},
}


def nome_euristica(stringa_euristica, lingua='it'):
    """
    Traduce una stringa euristica nella lingua richiesta.
    Accetta sia 'H7' che 'H7 · Flessibilità d'uso'.
    """
    import re
    match = re.match(r'(H\d+)', str(stringa_euristica))
    if match:
        codice = match.group(1)
        if codice in EURISTICHE:
            return EURISTICHE[codice][lingua]
    return stringa_euristica

# Categorie disponibili
CATEGORIE = {
    'problema': ['performance_lentezza', 'interfaccia_confusa', 'documentazione_insufficiente',
                 'messaggi_errore_poco_chiari', 'configurazione_complessa', 'mancanza_export_report',
                 'mancanza_modelli_ml', 'mancanza_metodi_fairness', 'incompatibilita_componenti',
                 'risultati_inattesi', 'stabilita_crash', 'curva_apprendimento', 'limiti_dataset'],
    'forza': ['punto_forza_ui_intuitiva', 'punto_forza_flessibilita_config',
              'punto_forza_workflow_guidato', 'punto_forza_ampiezza_componenti',
              'punto_forza_risultati_chiari', 'punto_forza_generico'],
    'neutro': ['non_codificabile'],
}


# ============================================================
# FEW-SHOT (14 esempi reali + sintetici)
# ============================================================
FEW_SHOT_EN = """
--- EXAMPLE 1: mancanza_modelli_ml ---
Response: "XGBoost or LightGBM which are much faster and more widely used in practice than the scikit-learn Gradient Boosting Classifier which is included"
Output:
{"temi": [
  {"codice": "mancanza_modelli_ml", "sentiment": "neutral", "citazione": "XGBoost or LightGBM which are much faster and more widely used"}
]}

--- EXAMPLE 2: mancanza_modelli_ml (deep learning) ---
Response: "The provided models covers the wide-adopted models. For future works, deep learning models can be integrated in the workflow to enable more in-depth comparison"
Output:
{"temi": [
  {"codice": "mancanza_modelli_ml", "sentiment": "neutral", "citazione": "deep learning models can be integrated"}
]}

--- EXAMPLE 3: mancanza_metodi_fairness ---
Response: "Adversarial Debaising which is a common processing technique that would give more options within that category alongside Exponentiated Gradient; MetaFAIR Classifier"
Output:
{"temi": [
  {"codice": "mancanza_metodi_fairness", "sentiment": "neutral", "citazione": "Adversarial Debaising which is a common processing technique"}
]}

--- EXAMPLE 4: documentazione_insufficiente ---
Response: "Adding short explanations or educational guidance directly within the platform could make it more accessible to less experienced users"
Output:
{"temi": [
  {"codice": "documentazione_insufficiente", "sentiment": "neutral", "citazione": "Adding short explanations or educational guidance directly within the platform"}
]}

--- EXAMPLE 5: documentazione_insufficiente + risultati_inattesi ---
Response: "On the results page, it would be helpful if a short explanation of the metrics. Documentation of the supported methods"
Output:
{"temi": [
  {"codice": "documentazione_insufficiente", "sentiment": "neutral", "citazione": "short explanation of the metrics"},
  {"codice": "risultati_inattesi", "sentiment": "neutral", "citazione": "Documentation of the supported methods"}
]}

--- EXAMPLE 6: risultati_inattesi ---
Response: "The calculation of the final trade-off score was not fully transparent, especially because the selected metrics have different optimization directions"
Output:
{"temi": [
  {"codice": "risultati_inattesi", "sentiment": "negative", "citazione": "calculation of the final trade-off score was not fully transparent"}
]}

--- EXAMPLE 7: stabilita_crash ---
Response: "During experiment execution I switched chrome tab and lost connection with server. I had to restart the experiment."
Output:
{"temi": [
  {"codice": "stabilita_crash", "sentiment": "negative", "citazione": "lost connection with server. I had to restart the experiment"}
]}

--- EXAMPLE 8: interfaccia_confusa ---
Response: "The Step 5 fairness-metric selection, where 'equally' and 'proportionally' are presented as radio buttons (mutually exclusive) even though the intended workflow suggests both should be selectable together"
Output:
{"temi": [
  {"codice": "interfaccia_confusa", "sentiment": "negative", "citazione": "radio buttons (mutually exclusive) even though both should be selectable"}
]}

--- EXAMPLE 9: mancanza_export_report ---
Response: "The generated Python script is missing the import for DEMV and core functions like build_model(), fit_with_fairness(), and compute_metrics() are not implemented"
Output:
{"temi": [
  {"codice": "mancanza_export_report", "sentiment": "negative", "citazione": "generated Python script is missing the import for DEMV and core functions"}
]}

--- EXAMPLE 10: punto_forza_ui_intuitiva + punto_forza_flessibilita_config ---
Response: "The intuitive UI and the flexibility in configuration settings"
Output:
{"temi": [
  {"codice": "punto_forza_ui_intuitiva", "sentiment": "positive", "citazione": "intuitive UI"},
  {"codice": "punto_forza_flessibilita_config", "sentiment": "positive", "citazione": "flexibility in configuration settings"}
]}

--- EXAMPLE 11: punto_forza_workflow_guidato ---
Response: "Its main strength is the structured, step-by-step workflow that makes it easy to compare multiple machine learning models and fairness methods"
Output:
{"temi": [
  {"codice": "punto_forza_workflow_guidato", "sentiment": "positive", "citazione": "structured, step-by-step workflow"}
]}

--- EXAMPLE 12: punto_forza_ampiezza_componenti ---
Response: "The low-code approach is very intuitive and significantly lowers the bar for researchers to perform fairness benchmarking; Usefulness of output metrics, ease of use and possibility to customize efficiently the models"
Output:
{"temi": [
  {"codice": "punto_forza_ui_intuitiva", "sentiment": "positive", "citazione": "low-code approach is very intuitive"},
  {"codice": "punto_forza_ampiezza_componenti", "sentiment": "positive", "citazione": "Usisfuness of output metrics, ease of use and possibility to customize efficiently the models"}
]}

--- EXAMPLE 13: punto_forza_risultati_chiari ---
Response: "No discrepancies because the results were consistent with what I expected given the configuration"
Output:
{"temi": [
  {"codice": "punto_forza_risultati_chiari", "sentiment": "positive", "citazione": "results were consistent with what I expected"}
]}

--- EXAMPLE 14: non_codificabile ---
Response: "nothing"
Output:
{"temi": [
  {"codice": "non_codificabile", "sentiment": "neutral", "citazione": "nothing"}
]}

--- EXAMPLE 15: performance_lentezza (sintetico) ---
Response: "The app takes too long to run a single benchmark, especially with larger datasets"
Output:
{"temi": [
  {"codice": "performance_lentezza", "sentiment": "negative", "citazione": "takes too long to run a single benchmark"}
]}

--- EXAMPLE 16: messaggi_errore_poco_chiari (sintetico) ---
Response: "When an error occurs, the message just says 'Something went wrong' without explaining what to do"
Output:
{"temi": [
  {"codice": "messaggi_errore_poco_chiari", "sentiment": "negative", "citazione": "the message just says 'Something went wrong' without explaining what to do"}
]}

--- EXAMPLE 17: configurazione_complessa (sintetico) ---
Response: "Too many parameters to set before running an experiment, it's overwhelming for new users"
Output:
{"temi": [
  {"codice": "configurazione_complessa", "sentiment": "negative", "citazione": "Too many parameters to set before running an experiment"}
]}

--- EXAMPLE 18: incompatibilita_componenti (sintetico) ---
Response: "Some fairness methods are not compatible with certain models but the app doesn't tell you until you try to run it"
Output:
{"temi": [
  {"codice": "incompatibilita_componenti", "sentiment": "negative", "citazione": "fairness methods are not compatible with certain models but the app doesn't tell you"}
]}

--- EXAMPLE 19: curva_apprendimento (sintetico) ---
Response: "It took me a while to understand how the workflow works, the learning curve is steep for beginners"
Output:
{"temi": [
  {"codice": "curva_apprendimento", "sentiment": "negative", "citazione": "learning curve is steep for beginners"}
]}

--- EXAMPLE 20: limiti_dataset (sintetico) ---
Response: "I couldn't upload my dataset because it was too large, there is no streaming option"
Output:
{"temi": [
  {"codice": "limiti_dataset", "sentiment": "negative", "citazione": "couldn't upload my dataset because it was too large"}
]}

--- EXAMPLE 21: punto_forza_generico (sintetico) ---
Response: "Great application, I really enjoyed using it"
Output:
{"temi": [
  {"codice": "punto_forza_generico", "sentiment": "positive", "citazione": "Great application"}
]}

--- EXAMPLE 22: risposta "va tutto bene" (no problemi) ---
Response: "I think all required models are available."
Output:
{"temi": [
  {"codice": "punto_forza_ampiezza_componenti", "sentiment": "positive", "citazione": "all required models are available"}
]}

--- EXAMPLE 23: negazione di discrepanze ---
Response: "No discrepancy was spotted in terms of expected results."
Output:
{"temi": [
  {"codice": "punto_forza_risultati_chiari", "sentiment": "positive", "citazione": "No discrepancy was spotted"}
]}
"""


class AnalisiTematicaLLM:
    def __init__(
        self,
        df_testo,
        config,
        chiama_llm_A,
        chiama_llm_B,
        lingua='it',
        max_temi_per_risposta=2,
    ):
        self.df_testo = df_testo
        self.config = config
        self.chiama_A = chiama_llm_A
        self.chiama_B = chiama_llm_B
        self.lingua = lingua
        self.max_temi = max_temi_per_risposta

        self.codifica_A = {}
        self.codifica_B = {}
        self.codifica_finale = {}
        self.accordo = {}

    # ========================================================
    # COSTRUZIONE PROMPT
    # ========================================================
    def _vocabolario_testo(self, lingua_vocab='en'):
        """Genera la sezione vocabolario per il prompt."""
        lines = []
        for codice, dati in TEMI.items():
            d = dati[lingua_vocab]
            esempi = ', '.join(d['esempi']) if d['esempi'] else '—'
            cat = dati['categoria']
            lines.append(f"- {codice} [{cat}]: {d['definizione']}. Examples: {esempi}")
        return '\n'.join(lines)

    def _system_prompt(self):
        vocab = self._vocabolario_testo('en')
        return f"""You are an expert qualitative thematic analysis researcher.
Your task is to code open-ended responses from a questionnaire
about the usability and fairness of the MANILA application.

RULES:
1. Code ONLY what is explicitly present in the response.
2. Do not invent themes not supported by the text.
3. If the response is empty, unintelligible or irrelevant,
   use the theme "non_codificabile".
4. Use ONLY the codes from the vocabulary below.
5. Assign AT MOST {self.max_temi} themes per response.
6. For each theme, indicate sentiment as one of: "positive", "neutral", "negative".
7. For each theme, extract a textual quote (max 25 words).
8. Output: valid JSON, no text outside JSON.
9. IMPORTANT: If a response says "no problem", "everything works", "I'm satisfied", 
   or similar, use the appropriate punto_forza_* theme — NEVER use non_codificabile 
   for positive feedback. non_codificabile is ONLY for empty or irrelevant responses.
10. IMPORTANT: If a response NEGATES an issue (e.g. "no discrepancy", "no missing method"), 
   do NOT code the corresponding problem theme. Code only the positive theme if applicable.

THEME VOCABULARY:
{vocab}
"""

    def _user_prompt(self, domanda_testo, risposte):
        risposte_numerate = '\n'.join(f'{i+1}. "{r}"' for i, r in enumerate(risposte))
        return f"""{FEW_SHOT_EN}

--- NOW CODE THESE RESPONSES ---
Question: "{domanda_testo}"
Responses:
{risposte_numerate}

Return a JSON with this EXACT structure:
{{
  "codifiche": [
    {{"id": 1, "temi": [{{"codice": "...", "sentiment": "...", "citazione": "..."}}]}},
    {{"id": 2, "temi": [...]}}
  ]
}}
No comments outside the JSON.
"""

    # ========================================================
    # PARSING E VALIDAZIONE
    # ========================================================
    def _parse_json(self, testo):
        if not testo:
            raise ValueError("Risposta vuota")
        try:
            return json.loads(testo)
        except Exception:
            pass
        m = re.search(r'```(?:json)?\s*(.*?)```', testo, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        start = testo.find('{')
        if start == -1:
            raise ValueError("Nessun JSON trovato")
        depth = 0
        for i in range(start, len(testo)):
            if testo[i] == '{':
                depth += 1
            elif testo[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(testo[start:i+1])
                    except Exception as e:
                        raise ValueError(f"JSON non valido: {e}")
        raise ValueError("JSON non bilanciato")

    def _normalizza_sentiment(self, s):
        s = str(s).strip().lower()
        mappa = {
            'positive': 'positivo', 'positivo': 'positivo',
            'negative': 'negativo', 'negativo': 'negativo',
            'neutral': 'neutro', 'neutro': 'neutro',
        }
        return mappa.get(s, 'neutro')

    def _valida_codifica(self, codifica):
        temi_validi = []
        for t in codifica.get('temi', []):
            codice = str(t.get('codice', '')).strip().lower()
            if codice not in CODICI_VALIDI:
                continue
            temi_validi.append({
                'codice': codice,
                'sentiment': self._normalizza_sentiment(t.get('sentiment', 'neutro')),
                'citazione': str(t.get('citazione', ''))[:200],
            })
        return temi_validi[:self.max_temi]

    # ========================================================
    # CODIFICA SINCRONA
    # ========================================================
    def codifica_tutte(self, batch_size=20, verbose=True):
        for col, risposte in self.df_testo.items():
            if not risposte:
                continue
            domanda_testo = self.config['domande_aperte'].get(col, col)
            if verbose:
                print(f"🔍 Codifica {col} ({len(risposte)} risposte)...")
            self.codifica_A[col] = self._codifica_con(
                self.chiama_A, domanda_testo, risposte, batch_size
            )
            self.codifica_B[col] = self._codifica_con(
                self.chiama_B, domanda_testo, risposte, batch_size
            )
        return self.codifica_A, self.codifica_B

    def _codifica_con(self, funzione_llm, domanda_testo, risposte, batch_size):
        risultati = []
        for i in range(0, len(risposte), batch_size):
            batch = risposte[i:i+batch_size]
            prompt = self._system_prompt() + "\n" + self._user_prompt(domanda_testo, batch)
            parsed = None
            for tentativo in range(4):
                try:
                    raw = funzione_llm(prompt)
                    parsed = self._parse_json(raw)
                    break
                except ValueError as e:
                    # JSON malformato: ritenta con backoff
                    msg = str(e).lower()
                    if ("json" in msg or "delimiter" in msg or "bilanciato" in msg) and tentativo < 3:
                        attesa = 2 ** tentativo
                        print(f"  🔁 JSON malformato (tentativo {tentativo+1}/4), ritento tra {attesa}s...")
                        time.sleep(attesa)
                        continue
                    else:
                        print(f"  ⚠️ JSON malformato, rinuncio: {e}")
                        break
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate" in msg or "503" in msg or "unavailable" in msg or "overload" in msg:
                        attesa = 2 ** tentativo
                        print(f"  ⏳ Rate limit, attendo {attesa}s...")
                        time.sleep(attesa)
                    else:
                        print(f"  ⚠️ Errore codifica batch: {type(e).__name__}: {e}")
                        break
            if parsed is None:
                codifiche = []
            else:
                codifiche = parsed.get('codifiche', [])
            by_id = {c.get('id'): c for c in codifiche if isinstance(c, dict)}
            for j, risposta in enumerate(batch):
                c = by_id.get(j + 1, {'temi': []})
                temi = self._valida_codifica(c)
                if not temi:
                    temi = [{
                        'codice': 'non_codificabile',
                        'sentiment': 'neutro',
                        'citazione': str(risposta)[:100],
                    }]
                risultati.append({'testo': risposta, 'temi': temi})
        return risultati

    # ========================================================
    # CODIFICA ASINCRONA
    # ========================================================
    async def codifica_tutte_async(self, batch_size=20, verbose=True):
        import asyncio
        for col, risposte in self.df_testo.items():
            if not risposte:
                continue
            domanda_testo = self.config['domande_aperte'].get(col, col)
            if verbose:
                print(f"🔍 Codifica parallela {col} ({len(risposte)} risposte)...")
            task_A = self._codifica_con_async(self.chiama_A, domanda_testo, risposte, batch_size)
            task_B = self._codifica_con_async(self.chiama_B, domanda_testo, risposte, batch_size)
            self.codifica_A[col], self.codifica_B[col] = await asyncio.gather(task_A, task_B)
        return self.codifica_A, self.codifica_B

    async def _codifica_con_async(self, funzione_llm, domanda_testo, risposte, batch_size):
        import asyncio
        risultati = []
        for i in range(0, len(risposte), batch_size):
            batch = risposte[i:i+batch_size]
            prompt = self._system_prompt() + "\n" + self._user_prompt(domanda_testo, batch)
            parsed = None
            for tentativo in range(4):
                try:
                    raw = await funzione_llm(prompt)
                    parsed = self._parse_json(raw)
                    break
                except ValueError as e:
                    # JSON malformato: ritenta con backoff
                    msg = str(e).lower()
                    if ("json" in msg or "delimiter" in msg or "bilanciato" in msg) and tentativo < 3:
                        attesa = 2 ** tentativo
                        print(f"  🔁 JSON malformato (tentativo {tentativo+1}/4), ritento tra {attesa}s...")
                        await asyncio.sleep(attesa)
                        continue
                    else:
                        print(f"  ⚠️ JSON malformato, rinuncio: {e}")
                        break
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate" in msg or "503" in msg or "unavailable" in msg or "overload" in msg:
                        attesa = 2 ** tentativo
                        print(f"  ⏳ Rate limit async, attendo {attesa}s...")
                        await asyncio.sleep(attesa)
                    else:
                        print(f"  ⚠️ Errore codifica async: {type(e).__name__}: {e}")
                        break
            if parsed is None:
                codifiche = []
            else:
                codifiche = parsed.get('codifiche', [])
            by_id = {c.get('id'): c for c in codifiche if isinstance(c, dict)}
            for j, risposta in enumerate(batch):
                c = by_id.get(j + 1, {'temi': []})
                temi = self._valida_codifica(c)
                if not temi:
                    temi = [{
                        'codice': 'non_codificabile',
                        'sentiment': 'neutro',
                        'citazione': str(risposta)[:100],
                    }]
                risultati.append({'testo': risposta, 'temi': temi})
        return risultati

    # ========================================================
    # CONFRONTO / ACCORDO
    # ========================================================
    def confronta(self):
        self.accordo = {}
        for col in self.codifica_A.keys():
            cod_A = self.codifica_A[col]
            cod_B = self.codifica_B.get(col, [])
            n = min(len(cod_A), len(cod_B))
            if n == 0:
                continue
            set_A = [frozenset(t['codice'] for t in cod_A[i]['temi']) for i in range(n)]
            set_B = [frozenset(t['codice'] for t in cod_B[i]['temi']) for i in range(n)]
            esatti = sum(1 for a, b in zip(set_A, set_B) if a == b)
            kappa = self._cohen_kappa_multilabel(set_A, set_B)
            self.accordo[col] = {
                'n': n,
                'accordo_esatto': esatti,
                'accordo_pct': round(esatti / n * 100, 1) if n else 0,
                'kappa': kappa,
            }
        return self.accordo

    def _cohen_kappa_multilabel(self, set_A, set_B):
        temi = sorted(CODICI_VALIDI)
        n = len(set_A)
        if n == 0:
            return 0.0
        tp = tn = fp = fn = 0
        for a, b in zip(set_A, set_B):
            for t in temi:
                in_a = t in a
                in_b = t in b
                if in_a and in_b:
                    tp += 1
                elif in_a and not in_b:
                    fn += 1
                elif not in_a and in_b:
                    fp += 1
                else:
                    tn += 1
        tot = tp + tn + fp + fn
        if tot == 0:
            return 0.0
        po = (tp + tn) / tot
        p_a_pos = (tp + fn) / tot
        p_b_pos = (tp + fp) / tot
        p_a_neg = (tn + fp) / tot
        p_b_neg = (tn + fn) / tot
        pe = p_a_pos * p_b_pos + p_a_neg * p_b_neg
        if pe >= 1.0:
            return 1.0
        return round((po - pe) / (1 - pe), 3)

    # ========================================================
    # RISOLUZIONE CONFLITTI
    # ========================================================
    def risolvi_disaccordi(self, strategia='unione'):
        self.codifica_finale = {}
        for col in self.codifica_A.keys():
            cod_A = self.codifica_A[col]
            cod_B = self.codifica_B.get(col, [])
            n = min(len(cod_A), len(cod_B))
            finali = []
            for i in range(n):
                temi_A = cod_A[i]['temi']
                temi_B = cod_B[i]['temi']
                if strategia == 'intersezione':
                    codici_A = {t['codice'] for t in temi_A}
                    codici_B = {t['codice'] for t in temi_B}
                    comuni = codici_A & codici_B
                    temi = [t for t in temi_A if t['codice'] in comuni]
                elif strategia == 'A_prevalente':
                    temi = temi_A if temi_A else temi_B
                else:  # unione
                    visti = set()
                    temi = []
                    for t in temi_A + temi_B:
                        if t['codice'] not in visti:
                            temi.append(t)
                            visti.add(t['codice'])
                    temi = temi[:self.max_temi]
                if not temi:
                    temi = [{
                        'codice': 'non_codificabile',
                        'sentiment': 'neutro',
                        'citazione': cod_A[i]['testo'][:100],
                    }]
                finali.append({'testo': cod_A[i]['testo'], 'temi': temi})
            self.codifica_finale[col] = finali
        return self.codifica_finale

    # ========================================================
    # RIASSUNTI
    # ========================================================
    def riassunto_per_tema(self):
        sorgente = self.codifica_finale or self.codifica_A
        riassunto = defaultdict(lambda: {
            'frequenza': 0, 'sentiment': Counter(), 'citazioni': [],
        })
        for col, codifiche in sorgente.items():
            for c in codifiche:
                for t in c['temi']:
                    codice = t['codice']
                    riassunto[codice]['frequenza'] += 1
                    riassunto[codice]['sentiment'][t['sentiment']] += 1
                    if t['citazione'] and len(riassunto[codice]['citazioni']) < 5:
                        riassunto[codice]['citazioni'].append(t['citazione'])
        out = {}
        for codice, dati in riassunto.items():
            sent_prevalente = 'neutro'
            if dati['sentiment']:
                sent_prevalente = dati['sentiment'].most_common(1)[0][0]
            out[codice] = {
                'frequenza': dati['frequenza'],
                'sentiment_prevalente': sent_prevalente,
                'distribuzione_sentiment': dict(dati['sentiment']),
                'citazioni': dati['citazioni'],
            }
        return out

    def risposte_per_tema(self, sentiment=None, config=None):
        config = config or self.config
        sorgente = self.codifica_finale or self.codifica_A
        out = defaultdict(list)
        for col, codifiche in sorgente.items():
            domanda = config['domande_aperte'].get(col, col)
            for c in codifiche:
                for t in c['temi']:
                    if sentiment is not None and t['sentiment'] != sentiment:
                        continue
                    out[t['codice']].append({
                        'domanda': domanda,
                        'testo': c['testo'],
                        'citazione': t['citazione'],
                        'sentiment': t['sentiment'],
                    })
        return dict(out)

    def riepilogo_per_tema_con_risposte(self, config=None):
        """
        Restituisce {codice_tema: {frequenza, distribuzione_sentiment,
                                    risposte: {positivo: [...], neutro: [...], negativo: [...]}}}
        Il parametro config permette di tradurre le domande nella lingua corrente.
        """
        config = config or self.config
        sorgente = self.codifica_finale or self.codifica_A
        out = defaultdict(lambda: {
            'frequenza': 0,
            'distribuzione_sentiment': Counter(),
            'risposte': {'positivo': [], 'neutro': [], 'negativo': []},
        })
        for col, codifiche in sorgente.items():
            domanda = config['domande_aperte'].get(col, col)
            for c in codifiche:
                for t in c['temi']:
                    codice = t['codice']
                    sent = t['sentiment']
                    out[codice]['frequenza'] += 1
                    out[codice]['distribuzione_sentiment'][sent] += 1
                    out[codice]['risposte'][sent].append({
                        'domanda': domanda,
                        'testo': c['testo'],
                        'citazione': t['citazione'],
                    })
        for codice in out:
            out[codice]['distribuzione_sentiment'] = dict(out[codice]['distribuzione_sentiment'])
        return dict(out)

    # ========================================================
    # PRIORITÀ
    # ========================================================
    def priorita_da_temi(self, min_frequenza=1):
        sorgente = self.codifica_finale or self.codifica_A
        totale_risposte = sum(len(v) for v in sorgente.values())
        if totale_risposte == 0:
            return []
        riepilogo = self.riepilogo_per_tema_con_risposte(config=self.config)
        risultati = []
        for codice, dati in riepilogo.items():
            tema = TEMI.get(codice, {})
            if tema.get('categoria') != 'problema':
                continue
            if dati['frequenza'] < min_frequenza:
                continue
            euristica_raw = tema.get('euristica', '—')
            euristica = nome_euristica(euristica_raw, self.lingua)
            severita = tema.get('severita', 2)
            freq_pct = dati['frequenza'] / totale_risposte * 100
            if freq_pct >= 30:
                freq_valore = 4
            elif freq_pct >= 15:
                freq_valore = 3
            elif freq_pct >= 5:
                freq_valore = 2
            else:
                freq_valore = 1
            dist = dati['distribuzione_sentiment']
            n_neg = dist.get('negativo', 0)
            sent_neg_pct = (n_neg / dati['frequenza'] * 100) if dati['frequenza'] > 0 else 0
            bonus = 1 if sent_neg_pct >= 60 else 0
            punteggio = severita + freq_valore + bonus
            citazioni = []
            for r in dati['risposte'].get('negativo', [])[:3]:
                citazioni.append(r['testo'][:200])
            for r in dati['risposte'].get('neutro', [])[:2]:
                citazioni.append(r['testo'][:200])
            risultati.append({
                'codice_tema': codice,
                'nome_tema': nome_tema(codice, self.lingua),
                'euristica': euristica,
                'severita': severita,
                'frequenza_valore': freq_valore,
                'frequenza_pct': round(freq_pct, 1),
                'n_risposte': dati['frequenza'],
                'sentiment_negativo_pct': round(sent_neg_pct, 1),
                'bonus_sentiment': bonus,
                'punteggio': punteggio,
                'citazioni': citazioni,
            })
        return sorted(risultati, key=lambda x: -x['punteggio'])

    # ========================================================
    # PROBLEMI CON SOLUZIONI
    # ========================================================
    def problemi_con_soluzioni(self, min_frequenza=1):
        """
        Restituisce una lista di problemi con la soluzione proposta.
        Usato per la nuova sezione del report "Cosa non funziona e come risolvere".
        """
        sorgente = self.codifica_finale or self.codifica_A
        totale_risposte = sum(len(v) for v in sorgente.values())
        if totale_risposte == 0:
            return []
        riepilogo = self.riepilogo_per_tema_con_risposte(config=self.config)
        risultati = []
        for codice, dati in riepilogo.items():
            tema = TEMI.get(codice, {})
            if tema.get('categoria') != 'problema':
                continue
            if dati['frequenza'] < min_frequenza:
                continue
            freq_pct = dati['frequenza'] / totale_risposte * 100
            dist = dati['distribuzione_sentiment']
            n_neg = dist.get('negativo', 0)
            sent_neg_pct = (n_neg / dati['frequenza'] * 100) if dati['frequenza'] > 0 else 0
            soluzione_dict = tema.get('soluzione', {})
            soluzione = soluzione_dict.get(self.lingua, soluzione_dict.get('it', '—'))
            risultati.append({
                'codice_tema': codice,
                'nome_tema': nome_tema(codice, self.lingua),
                'descrizione': tema.get(self.lingua, {}).get('definizione', '—'),
                'soluzione': soluzione,
                'n_risposte': dati['frequenza'],
                'frequenza_pct': round(freq_pct, 1),
                'sentiment_negativo_pct': round(sent_neg_pct, 1),
                'esempi': [r['testo'][:200] for r in dati['risposte'].get('negativo', [])[:3]],
            })
        return sorted(risultati, key=lambda x: -x['n_risposte'])

    # ========================================================
    # UTILITÀ
    # ========================================================
    def temi_per_categoria(self, categoria):
        """Restituisce i codici dei temi appartenenti a una categoria."""
        return CATEGORIE.get(categoria, [])

    def salva_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'lingua': self.lingua,
                'vocabolario': TEMI,
                'codifica_A': self.codifica_A,
                'codifica_B': self.codifica_B,
                'codifica_finale': self.codifica_finale,
                'accordo': self.accordo,
            }, f, ensure_ascii=False, indent=2)


# ============================================================
# UTILITÀ PER IL REPORT
# ============================================================
def nome_tema(codice, lingua='it'):
    return TEMI.get(codice, {}).get(lingua, {}).get('nome', codice)


def categoria_tema(codice):
    return TEMI.get(codice, {}).get('categoria', 'neutro')


def soluzione_tema(codice, lingua='it'):
    s = TEMI.get(codice, {}).get('soluzione', {})
    return s.get(lingua, s.get('it', '—'))

def nome_euristica_export(stringa_euristica, lingua='it'):
    return nome_euristica(stringa_euristica, lingua)