# utils/report_generator.py - Versione con analisi tematica multi-LLM e priorità

import pandas as pd
import numpy as np
from datetime import datetime
import re


class ReportGenerator:
    def __init__(self, analisi_likert, analisi_testo, generatore_grafici, config,
                 lingua='it', analisi_profilo=None, analisi_tematica=None):
        self.analisi_likert = analisi_likert
        self.analisi_testo = analisi_testo
        self.generatore_grafici = generatore_grafici
        self.config = config
        self.lingua = lingua
        self.analisi_profilo = analisi_profilo
        self.analisi_tematica = analisi_tematica

        if lingua == 'it':
            self.T = {
                'titolo': 'Report Analisi Questionario MANILA',
                'sottotitolo': "Valutazione dell'applicazione MANILA",
                'data': 'Data generazione',
                'panoramica': 'Panoramica Generale',
                'profilo': 'Profilo dei Rispondenti',
                'totale_risposte': 'Totale risposte',
                'domande_likert': 'Domande Likert',
                'domande_aperte': 'Domande Aperte',
                'affidabilita': 'Affidabilità (Alpha di Cronbach)',
                'statistiche_descrittive': 'Statistiche Descrittive per Scala',
                'analisi_testo': 'Analisi Risposte Aperte',
                'sentiment': 'Analisi del Sentiment',
                'risposte_esempio': 'Esempi di Risposte',
                'conclusioni': 'Conclusioni e Raccomandazioni',
                'punti_forza': 'Punti di Forza',
                'aree_miglioramento': 'Aree da Migliorare',
                'priorita': 'Priorità di Intervento',
                'media': 'Media',
                'mediana': 'Mediana',
                'std': 'Dev. Std.',
                'n': 'N. Risposte',
                'domanda': 'Domanda',
                'interpretazione': 'Interpretazione',
                'eccellente': 'Eccellente', 'buona': 'Buona', 'discreta': 'Discreta',
                'sufficiente': 'Sufficiente', 'scarsa': 'Scarsa', 'molto_scarsa': 'Molto Scarsa',
                'neutro': 'Neutro', 'positivo': 'Positivo', 'negativo': 'Negativo',
                'non_calcolabile': 'Non calcolabile',
                'risposta': 'Risposta', 'frequenza': 'Frequenza', 'percentuale': 'Percentuale',
                'consenso': 'Consenso', 'posizione': 'Posizione attuale',
                'esperienza': 'Esperienza ML', 'conoscenza': 'Conoscenza Fairness',
                'analisi_tematica': 'Analisi Tematica (Multi-LLM)',
                'accordo_intercoder': 'Accordo inter-coder',
                'temi_rilevati': 'Temi rilevati',
                'risposte_per_tema': 'Risposte per tema',
                'kappa': "Cohen's κ",
                'priorita_tematica': 'Tabella Priorità (da analisi tematica)',
                'problemi_soluzioni': 'Cosa non funziona e come risolvere',
                'panoramica_problemi': 'Panoramica dei problemi',
                'dettaglio_problemi': 'Dettaglio e soluzioni proposte',
                'punti_forza_tematica': 'Punti di forza',
                'soluzione_proposta': 'Soluzione proposta',
                'descrizione': 'Descrizione',
                'esempi': 'Esempi',
                'procedura': 'Procedura',
                'procedura_testo': (
                    'Questa sezione raccoglie i problemi ricorrenti emersi dalle risposte aperte, '
                    'classificati secondo un vocabolario di temi specifici. Per ciascun problema '
                    'sono indicati la frequenza, la percentuale di sentiment negativo e una soluzione '
                    'proposta di intervento.'
                ),
                'come_leggere_punti_forza': (
                    'Questa sezione raccoglie gli aspetti positivi emersi dalle risposte aperte. '
                    'Per ciascun punto di forza sono indicati la frequenza e alcuni esempi testuali '
                    'delle risposte originali.'
                ),
            }
        else:
            self.T = {
                'titolo': 'MANILA Questionnaire Analysis Report',
                'sottotitolo': 'Evaluation of the MANILA application',
                'data': 'Generation date',
                'panoramica': 'Overview',
                'profilo': 'Respondent Profile',
                'totale_risposte': 'Total responses',
                'domande_likert': 'Likert questions',
                'domande_aperte': 'Open questions',
                'affidabilita': "Reliability (Cronbach's Alpha)",
                'statistiche_descrittive': 'Descriptive Statistics by Scale',
                'analisi_testo': 'Open Responses Analysis',
                'sentiment': 'Sentiment Analysis',
                'risposte_esempio': 'Sample Responses',
                'conclusioni': 'Conclusions and Recommendations',
                'punti_forza': 'Strengths',
                'aree_miglioramento': 'Areas for Improvement',
                'priorita': 'Intervention Priorities',
                'media': 'Mean', 'mediana': 'Median', 'std': 'Std Dev.', 'n': 'N Responses',
                'domanda': 'Question', 'interpretazione': 'Interpretation',
                'eccellente': 'Excellent', 'buona': 'Good', 'discreta': 'Fair',
                'sufficiente': 'Sufficient', 'scarsa': 'Poor', 'molto_scarsa': 'Very Poor',
                'neutro': 'Neutral', 'positivo': 'Positive', 'negativo': 'Negative',
                'non_calcolabile': 'Not calculable',
                'risposta': 'Response', 'frequenza': 'Frequency', 'percentuale': 'Percentage',
                'consenso': 'Consent', 'posizione': 'Position',
                'esperienza': 'ML Experience', 'conoscenza': 'Fairness Knowledge',
                'analisi_tematica': 'Thematic Analysis (Multi-LLM)',
                'accordo_intercoder': 'Inter-coder agreement',
                'temi_rilevati': 'Detected themes',
                'risposte_per_tema': 'Responses by theme',
                'kappa': "Cohen's κ",
                'priorita_tematica': 'Priority Table (from thematic analysis)',
                'problemi_soluzioni': 'What does not work and how to fix it',
                'panoramica_problemi': 'Overview of issues',
                'dettaglio_problemi': 'Details and proposed solutions',
                'punti_forza_tematica': 'Strengths',
                'soluzione_proposta': 'Proposed solution',
                'descrizione': 'Description',
                'esempi': 'Examples',
                'procedura': 'Procedure',
                'procedura_testo': (
                    'This section collects recurring issues from open responses, '
                    'classified according to a vocabulary of specific themes. For each issue, '
                    'frequency, negative sentiment percentage, and a proposed solution are shown.'
                ),
                'come_leggere_punti_forza': (
                    'This section collects positive aspects from open responses. '
                    'For each strength, frequency and textual examples are shown.'
                ),
            }

    # ============================================================
    # HELPER
    # ============================================================
    def _get_interpretazione(self, valore):
        if valore >= 4.5:
            return f'🟢 {self.T["eccellente"]}'
        elif valore >= 4.0:
            return f'🟢 {self.T["buona"]}'
        elif valore >= 3.5:
            return f'🟡 {self.T["discreta"]}'
        elif valore >= 3.0:
            return f'🟠 {self.T["sufficiente"]}'
        elif valore >= 2.5:
            return f'🔴 {self.T["scarsa"]}'
        else:
            return f'🔴 {self.T["molto_scarsa"]}'

    def _get_alpha_interpretazione(self, alpha):
        if alpha is None:
            return self.T['non_calcolabile']
        elif alpha >= 0.9:
            return f'✅ {self.T["eccellente"]}'
        elif alpha >= 0.8:
            return f'✅ {self.T["buona"]}'
        elif alpha >= 0.7:
            return f'ℹ️ {self.T["discreta"]}'
        elif alpha >= 0.6:
            return f'⚠️ {self.T["sufficiente"]}'
        else:
            return f'⚠️ {self.T["scarsa"]}'

    def _get_colore_media(self, media):
        if media >= 4.0: return 'good'
        elif media >= 3.5: return 'fair'
        elif media >= 3.0: return 'sufficient'
        else: return 'poor'

    def _badge(self, testo, bg, fg):
        return (f'<span style="display:inline-block; background:{bg}; '
                f'color:{fg}; padding:2px 10px; border-radius:4px; '
                f'font-weight:bold; font-size:0.9em;">{testo}</span>')

    def _label_priorita_punteggio(self, p):
        if p >= 7:
            return ('🔴 ' + ('Critica' if self.lingua == 'it' else 'Critical'), '#e74c3c')
        elif p >= 6:
            return ('🟠 ' + ('Alta' if self.lingua == 'it' else 'High'), '#f39c12')
        elif p >= 4:
            return ('🟡 ' + ('Media' if self.lingua == 'it' else 'Medium'), '#f1c40f')
        else:
            return ('🟢 ' + ('Bassa' if self.lingua == 'it' else 'Low'), '#2ecc71')

    def _genera_sezione_tematica(self):
        """Genera la sezione HTML dell'analisi tematica multi-LLM."""
        if self.analisi_tematica is None:
            return ""

        from Utils.analisi_tematica_llm import nome_tema, TEMI

        at = self.analisi_tematica
        html = [f'<h2>🤖 {self.T["analisi_tematica"]}</h2>']

        # --------------------------------------------------------
        # 1. ACCORDO INTER-CODER
        # --------------------------------------------------------
        if at.accordo:
            html.append(f'<h3>📊 {self.T["accordo_intercoder"]}</h3>')
            html.append('<table><thead><tr>'
                        '<th>Domanda</th><th>N</th><th>Accordo %</th>'
                        f'<th>{self.T["kappa"]}</th>'
                        '</tr></thead><tbody>')
            for col, v in at.accordo.items():
                domanda = self.config['domande_aperte'].get(col, col)
                kappa = v['kappa']
                if kappa >= 0.8:
                    colore = '#2ecc71'
                elif kappa >= 0.6:
                    colore = '#f1c40f'
                else:
                    colore = '#e74c3c'
                html.append(
                    f'<tr><td>{domanda}</td><td>{v["n"]}</td>'
                    f'<td>{v["accordo_pct"]}%</td>'
                    f'<td style="color:{colore}; font-weight:bold;">{kappa}</td></tr>'
                )
            html.append('</tbody></table>')

        # --------------------------------------------------------
        # 2. TEMI RILEVATI
        # --------------------------------------------------------
        riassunto = at.riassunto_per_tema()
        if riassunto:
            html.append(f'<h3>🎯 {self.T["temi_rilevati"]}</h3>')
            html.append('<table><thead><tr>'
                        '<th>Tema</th><th>Categoria</th><th>Frequenza</th><th>Sentiment</th>'
                        '</tr></thead><tbody>')
            for codice, v in sorted(riassunto.items(), key=lambda x: -x[1]['frequenza']):
                nome = nome_tema(codice, self.lingua)
                categoria = TEMI.get(codice, {}).get('categoria', '—')
                cat_label = {
                    'problema': '🔴 Problema',
                    'forza': '🟢 Forza',
                    'neutro': '⚪ Neutro',
                }.get(categoria, categoria)
                html.append(
                    f'<tr><td><strong>{nome}</strong><br>'
                    f'<small style="color:#999;">({codice})</small></td>'
                    f'<td>{cat_label}</td>'
                    f'<td>{v["frequenza"]}</td>'
                    f'<td>{v["sentiment_prevalente"]}</td></tr>'
                )
            html.append('</tbody></table>')

        # --------------------------------------------------------
        # 3. RISPOSTE PER TEMA (a tendina, divise per sentiment)
        # --------------------------------------------------------
        riepilogo_full = at.riepilogo_per_tema_con_risposte()
        if riepilogo_full:
            html.append(f'<h3>📋 {self.T["risposte_per_tema"]}</h3>')

            stili = {
                'positivo': ('#E8F5E9', '#43A047',
                            '😊 Positivo' if self.lingua == 'it' else '😊 Positive'),
                'neutro':   ('#FFF3E0', '#FB8C00',
                            '😐 Neutro' if self.lingua == 'it' else '😐 Neutral'),
                'negativo': ('#FFEBEE', '#E53935',
                            '😞 Negativo' if self.lingua == 'it' else '😞 Negative'),
            }
            ordine_sent = ['negativo', 'neutro', 'positivo']

            for codice, dati in sorted(riepilogo_full.items(),
                                    key=lambda x: -x[1]['frequenza']):
                nome = nome_tema(codice, self.lingua)
                dist = dati['distribuzione_sentiment']
                n_neg = dist.get('negativo', 0)
                n_neu = dist.get('neutro', 0)
                n_pos = dist.get('positivo', 0)

                html.append(
                    f'<details style="margin:10px 0; border:1px solid #ddd; '
                    f'border-radius:6px; padding:8px 12px; background:#fafbfc;">'
                    f'<summary style="cursor:pointer; font-weight:bold; color:#2c3e50;">'
                    f'{nome} ({dati["frequenza"]}) — '
                    f'😞 {n_neg} · 😐 {n_neu} · 😊 {n_pos}'
                    f'</summary>'
                )

                for sent in ordine_sent:
                    risposte_sent = dati['risposte'].get(sent, [])
                    if not risposte_sent:
                        continue
                    bg, border, label = stili[sent]
                    html.append(
                        f'<div style="margin:8px 0 4px 0;">'
                        f'<span style="background:{border}; color:white; '
                        f'padding:2px 10px; border-radius:12px; font-size:0.85em; '
                        f'font-weight:500;">{label} ({len(risposte_sent)})</span>'
                        f'</div>'
                    )
                    html.append('<ul style="margin:4px 0 12px 20px;">')
                    for it in risposte_sent:
                        html.append(
                            f'<li style="background:{bg}; padding:6px 10px; '
                            f'border-left:3px solid {border}; margin:4px 0; '
                            f'border-radius:3px; list-style:none;">'
                            f'<small><strong>[{it["domanda"]}]</strong> {it["testo"]}</small></li>'
                        )
                    html.append('</ul>')

                html.append('</details>')

        # --------------------------------------------------------
        # 4. COSA NON FUNZIONA E COME RISOLVERE
        #    (senza colonna Priorità — Opzione C)
        # --------------------------------------------------------
        problemi = at.problemi_con_soluzioni()
        if problemi:
            html.append(f'<h3>🔧 {self.T["problemi_soluzioni"]}</h3>')
            html.append(f'<p style="font-size:0.9em; color:#555;">{self.T["procedura_testo"]}</p>')

            # --- 4a. Tabella panoramica (senza Priorità) ---
            html.append(f'<h4>📊 {self.T["panoramica_problemi"]}</h4>')
            html.append(
                '<table><thead><tr>'
                '<th>Problema</th>'
                '<th>Frequenza</th>'
                '<th>Sent. neg. %</th>'
                '</tr></thead><tbody>'
            )
            for p in problemi:
                html.append(
                    f'<tr>'
                    f'<td><strong>{p["nome_tema"]}</strong></td>'
                    f'<td>{p["n_risposte"]} ({p["frequenza_pct"]}%)</td>'
                    f'<td>{p["sentiment_negativo_pct"]}%</td>'
                    f'</tr>'
                )
            html.append('</tbody></table>')

            # --- 4b. Dettaglio blocchi (a tendina) ---
            html.append(f'<h4>📋 {self.T["dettaglio_problemi"]}</h4>')

            for i, p in enumerate(problemi, 1):
                # Header della tendina
                html.append(
                    f'<details style="margin:12px 0; border:1px solid #ddd; '
                    f'border-left:5px solid #e74c3c; border-radius:6px; '
                    f'padding:8px 14px; background:#fafbfc;">'
                    f'<summary style="cursor:pointer; font-weight:bold; color:#2c3e50;">'
                    f'{i}. {p["nome_tema"]} '
                    f'<span style="font-weight:normal; color:#777; font-size:0.9em;">'
                    f'— {p["n_risposte"]} risposte'
                    f'</span>'
                    f'</summary>'
                )

                # Contenuto della tendina
                html.append('<div style="margin-top:12px;">')

                # Descrizione
                html.append(
                    f'<div style="margin:8px 0;">'
                    f'<strong style="color:#34495e;">📌 {self.T["descrizione"]}</strong><br>'
                    f'<span style="color:#555;">{p["descrizione"]}</span>'
                    f'</div>'
                )

                # Soluzione
                html.append(
                    f'<div style="margin:12px 0; background:#e8f5e9; '
                    f'padding:10px 14px; border-radius:4px; border-left:3px solid #43A047;">'
                    f'<strong style="color:#2e7d32;">💡 {self.T["soluzione_proposta"]}</strong><br>'
                    f'<span style="color:#333;">{p["soluzione"]}</span>'
                    f'</div>'
                )

                # Esempi
                if p['esempi']:
                    html.append(f'<div style="margin:8px 0;">'
                                f'<strong style="color:#34495e;">📝 {self.T["esempi"]}</strong>'
                                f'<ul style="margin:6px 0 0 20px;">')
                    for ex in p['esempi']:
                        testo = ex[:250] + ('...' if len(ex) > 250 else '')
                        html.append(f'<li style="margin:4px 0; color:#555;">'
                                    f'<small>"{testo}"</small></li>')
                    html.append('</ul></div>')

                html.append('</div>')  # chiude contenuto
                html.append('</details>')

        # --------------------------------------------------------
        # 5. TABELLA PRIORITÀ TEMATICA (unica fonte di priorità)
        # --------------------------------------------------------
        priorita_temi = at.priorita_da_temi()
        if priorita_temi:
            html.append(f'<h3>🎯 {self.T.get("priorita_tematica", "Tabella Priorità (da analisi tematica)")}</h3>')

            nota = (
                'Il punteggio è calcolato come <strong>Severità + Frequenza + Bonus sentiment</strong>. '
                'La severità deriva dalla mappatura tema → euristica di Nielsen; '
                'la frequenza dalla percentuale di risposte codificate; '
                'il bonus sentiment premia i temi con ≥60% di risposte negative.'
                if self.lingua == 'it' else
                'The score is computed as <strong>Severity + Frequency + Sentiment bonus</strong>. '
                'Severity comes from the theme → Nielsen heuristic mapping; '
                'frequency from the percentage of coded responses; '
                'the sentiment bonus rewards themes with ≥60% negative responses.'
            )
            html.append(f'<p style="font-size:0.9em; color:#555;">{nota}</p>')

            html.append(
                '<table><thead><tr>'
                '<th>Tema</th><th>Euristica</th><th>Severità</th>'
                '<th>Frequenza</th><th>Sent. neg. %</th><th>Bonus</th>'
                '<th>Punteggio</th><th>Priorità</th>'
                '</tr></thead><tbody>'
            )

            for p in priorita_temi:
                label, colore = self._label_priorita_punteggio(p['punteggio'])
                html.append(
                    f'<tr>'
                    f'<td>{p["nome_tema"]}</td>'
                    f'<td>{p["euristica"]}</td>'
                    f'<td>{p["severita"]}</td>'
                    f'<td>{p["frequenza_valore"]} ({p["frequenza_pct"]}%)</td>'
                    f'<td>{p["sentiment_negativo_pct"]}%</td>'
                    f'<td>{p["bonus_sentiment"]}</td>'
                    f'<td>{p["punteggio"]}</td>'
                    f'<td style="color:{colore}; font-weight:bold; white-space:nowrap;">{label}</td>'
                    f'</tr>'
                )
            html.append('</tbody></table>')

        return '\n'.join(html)

    # ============================================================
    # SEZIONE PRIORITÀ (Nielsen — legacy, usata solo nel tab interattivo)
    # ============================================================
    def _estrai_problemi_da_risposte(self, domande_target=None):
        """Versione keyword matching (usata solo in app.py per tab analisi testo)."""
        if not hasattr(self.analisi_testo, 'df_testo') or not self.analisi_testo.df_testo:
            return []

        euristiche_nielsen = {
            1: {'short_it': 'H1 · Visibilità stato', 'short_en': 'H1 · Visibility'},
            2: {'short_it': 'H2 · Linguaggio', 'short_en': 'H2 · Real world match'},
            3: {'short_it': 'H3 · Controllo utente', 'short_en': 'H3 · User control'},
            4: {'short_it': 'H4 · Consistenza', 'short_en': 'H4 · Consistency'},
            5: {'short_it': 'H5 · Prevenzione errori', 'short_en': 'H5 · Error prevention'},
            6: {'short_it': 'H6 · Riconoscimento', 'short_en': 'H6 · Recognition'},
            7: {'short_it': 'H7 · Flessibilità', 'short_en': 'H7 · Flexibility'},
            8: {'short_it': 'H8 · Design minimalista', 'short_en': 'H8 · Minimalist design'},
            9: {'short_it': 'H9 · Recupero errori', 'short_en': 'H9 · Error recovery'},
            10: {'short_it': 'H10 · Documentazione', 'short_en': 'H10 · Documentation'},
        }

        problemi_definiti = [
            {'nome_it': 'Lentezza / performance scadenti', 'nome_en': 'Slowness / poor performance',
             'keywords': ['lento', 'lenta', 'lentezza', 'performance', 'velocità', 'rallenta',
                          'tempo', 'attesa', 'slow', 'slowly'], 'severita': 3, 'euristiche': [1]},
            {'nome_it': 'Documentazione insufficiente', 'nome_en': 'Insufficient documentation',
             'keywords': ['documentazione', 'guida', 'tutorial', 'help', 'aiuto', 'spiegazione',
                          'chiarimento', 'documentation', 'guide'], 'severita': 2, 'euristiche': [10]},
            {'nome_it': 'Interfaccia poco chiara / confusa', 'nome_en': 'Unclear / confusing interface',
             'keywords': ['confuso', 'confusa', 'chiaro', 'chiara', 'intuitivo', 'interfaccia',
                          'layout', 'navigazione', 'interface', 'confusing', 'unclear'],
             'severita': 3, 'euristiche': [8, 2]},
            {'nome_it': 'Mancanza di funzionalità (export, report, ecc.)',
             'nome_en': 'Missing features (export, report, etc.)',
             'keywords': ['export', 'esportare', 'import', 'report', 'salvare', 'configurazione',
                          'manca', 'mancante', 'missing', 'save', 'configuration'],
             'severita': 3, 'euristiche': [7]},
            {'nome_it': 'Errori / instabilità', 'nome_en': 'Errors / instability',
             'keywords': ['errore', 'crash', 'bug', 'instabile', 'blocca', 'fallisce',
                          'problema tecnico', 'error', 'unstable', 'fails', 'broken',
                          'discrepanz', 'discrepanc'], 'severita': 4, 'euristiche': [5]},
            {'nome_it': 'Messaggi di errore poco chiari', 'nome_en': 'Unclear error messages',
             'keywords': ['messaggio', 'indicazione', 'capire', 'risolvere', 'spiegazione errore',
                          'message', 'understand'], 'severita': 2, 'euristiche': [9]},
            {'nome_it': 'Mancanza di modelli ML o metodi fairness',
             'nome_en': 'Missing ML models or fairness methods',
             'keywords': ['modello', 'modelli', 'metodo', 'metodi', 'fairness', 'model',
                          'method', 'missing model', 'missing method'], 'severita': 3, 'euristiche': [7]},
            {'nome_it': 'Difficoltà di apprendimento / onboarding',
             'nome_en': 'Learning curve / onboarding difficulty',
             'keywords': ['imparare', 'apprendere', 'difficile', 'complicato', 'learning',
                          'difficult', 'complicated', 'steep'], 'severita': 2, 'euristiche': [6]},
        ]

        pesi_domande = {
            'Q27': 1.0, 'Q28': 1.2, 'Q36': 1.2, 'Q19': 1.0,
            'Q14': 1.0, 'Q33': 0.8, 'Q9': 0.8, 'Q10': 0.8, 'Q35': 0.5,
        }

        if domande_target is None:
            domande_da_usare = [c for c in pesi_domande.keys() if c in self.analisi_testo.df_testo]
        else:
            domande_da_usare = [c for c in domande_target if c in self.analisi_testo.df_testo]

        if not domande_da_usare:
            return []

        risposte_pesate = []
        for col in domande_da_usare:
            peso = pesi_domande.get(col, 1.0)
            for risposta in self.analisi_testo.df_testo[col]:
                risposte_pesate.append((str(risposta), peso))

        totale_risposte = len(risposte_pesate)
        if totale_risposte == 0:
            return []

        risultati = []
        for problema in problemi_definiti:
            conteggio_pesato = 0.0
            conteggio_grezzo = 0
            for risposta, peso in risposte_pesate:
                risposta_lower = risposta.lower()
                if any(kw in risposta_lower for kw in problema['keywords']):
                    conteggio_pesato += peso
                    conteggio_grezzo += 1

            if conteggio_grezzo > 0:
                freq_normalizzata = conteggio_pesato / totale_risposte
                if freq_normalizzata >= 0.90: freq_valore = 4
                elif freq_normalizzata >= 0.51: freq_valore = 3
                elif freq_normalizzata >= 0.11: freq_valore = 2
                else: freq_valore = 1

                punteggio = problema['severita'] + freq_valore
                nome = problema['nome_it'] if self.lingua == 'it' else problema['nome_en']
                eur_labels = []
                for eid in problema['euristiche']:
                    h = euristiche_nielsen[eid]
                    eur_labels.append(h['short_it'] if self.lingua == 'it' else h['short_en'])

                risultati.append({
                    'problema': nome,
                    'severita': problema['severita'],
                    'frequenza_valore': freq_valore,
                    'frequenza_pct': round(freq_normalizzata * 100, 1),
                    'conteggio': conteggio_grezzo,
                    'punteggio': punteggio,
                    'euristiche_ids': problema['euristiche'],
                    'euristiche_labels': eur_labels,
                })

        return sorted(risultati, key=lambda x: x['punteggio'], reverse=True)

    # ============================================================
    # SEZIONE PROFILO
    # ============================================================
    def _genera_sezione_profilo(self):
        if self.analisi_profilo is None:
            return ""

        html = [f'<h2>{self.T["profilo"]}</h2>']

        if 'Q1' in self.analisi_profilo.df_profilo.columns:
            freq = self.analisi_profilo.frequenze('Q1')
            if freq is not None:
                html.append(f'<h3>Q1 — {self.T["consenso"]}</h3>')
                html.append('<table><thead><tr>'
                            f'<th>{self.T["risposta"]}</th>'
                            f'<th>{self.T["frequenza"]}</th>'
                            f'<th>{self.T["percentuale"]}</th>'
                            '</tr></thead><tbody>')
                for _, row in freq.iterrows():
                    html.append(f'<tr><td>{row["Risposta"]}</td>'
                                f'<td>{row["Frequenza"]}</td>'
                                f'<td>{row["Percentuale"]}%</td></tr>')
                html.append('</tbody></table>')

        if 'Q2' in self.analisi_profilo.df_profilo.columns:
            freq = self.analisi_profilo.frequenze('Q2')
            if freq is not None:
                html.append(f'<h3>Q2 — {self.T["posizione"]}</h3>')
                html.append('<table><thead><tr>'
                            f'<th>{self.T["risposta"]}</th>'
                            f'<th>{self.T["frequenza"]}</th>'
                            f'<th>{self.T["percentuale"]}</th>'
                            '</tr></thead><tbody>')
                for _, row in freq.iterrows():
                    html.append(f'<tr><td>{row["Risposta"]}</td>'
                                f'<td>{row["Frequenza"]}</td>'
                                f'<td>{row["Percentuale"]}%</td></tr>')
                html.append('</tbody></table>')

        if 'Q3' in self.analisi_profilo.df_profilo.columns:
            stats = self.analisi_profilo.statistiche_ordinali('Q3')
            if stats:
                html.append(f'<h3>Q3 — {self.T["esperienza"]}</h3>')
                html.append(f'<p><strong>{self.T["media"]}:</strong> {stats["media"]:.2f} '
                            f'(n={stats["n"]})</p>')
                labels_q3 = self.config.get('labels_profilo', {}).get('Q3', {})
                html.append('<table><thead><tr>'
                            f'<th>{self.T["risposta"]}</th>'
                            f'<th>{self.T["frequenza"]}</th>'
                            '</tr></thead><tbody>')
                for valore, count in stats['distribuzione'].items():
                    etichetta = labels_q3.get(int(valore), str(valore))
                    html.append(f'<tr><td>{etichetta}</td><td>{count}</td></tr>')
                html.append('</tbody></table>')

        if 'Q4' in self.analisi_profilo.df_profilo.columns:
            stats = self.analisi_profilo.statistiche_ordinali('Q4')
            if stats:
                html.append(f'<h3>Q4 — {self.T["conoscenza"]}</h3>')
                html.append(f'<p><strong>{self.T["media"]}:</strong> {stats["media"]:.2f} '
                            f'(n={stats["n"]})</p>')
                labels_q4 = self.config.get('labels_profilo', {}).get('Q4', {})
                html.append('<table><thead><tr>'
                            f'<th>{self.T["risposta"]}</th>'
                            f'<th>{self.T["frequenza"]}</th>'
                            '</tr></thead><tbody>')
                for valore, count in stats['distribuzione'].items():
                    etichetta = labels_q4.get(int(valore), str(valore))
                    html.append(f'<tr><td>{etichetta}</td><td>{count}</td></tr>')
                html.append('</tbody></table>')

        return '\n'.join(html)

    # ============================================================
    # GENERAZIONE HTML
    # ============================================================
    def genera_html(self, output_file=None):
        try:
            stats = self.analisi_likert.statistiche_descrittive_con_codici()
        except AttributeError:
            stats_fallback = self.analisi_likert.statistiche_descrittive()
            stats = pd.DataFrame()
            for codice, testo in self.config['domande_likert'].items():
                if testo in stats_fallback.index:
                    stats.loc[codice] = stats_fallback.loc[testo]

        domande_per_scala = self.analisi_likert.get_domande_per_scala()
        alpha_per_scala = self.analisi_likert.calcola_alpha_cronbach()

        html = []
        html.append(f"""<!DOCTYPE html>
<html lang="{self.lingua}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.T['titolo']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; line-height: 1.5; }}
        h1 {{ color: #1E88E5; text-align: center; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 40px; }}
        h3 {{ color: #34495e; margin-top: 25px; }}
        h4 {{ color: #34495e; margin-top: 20px; font-size: 1.05em; }}
        h5 {{ color: #2c3e50; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header .subtitle {{ color: #7f8c8d; }}
        .header .date {{ color: #95a5a6; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background-color: #3498db; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-box {{ display: inline-block; background-color: #f0f2f6; padding: 15px; margin: 5px; border-radius: 8px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1E88E5; }}
        .metric-label {{ font-size: 14px; color: #555; }}
        .good {{ color: #2ecc71; }}
        .fair {{ color: #f1c40f; }}
        .sufficient {{ color: #e67e22; }}
        .poor {{ color: #e74c3c; }}
        .section {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .footer {{ text-align: center; margin-top: 40px; color: #95a5a6; font-size: 0.8em; border-top: 1px solid #ddd; padding-top: 20px; }}
        .strength {{ background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }}
        .weakness {{ background-color: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; margin: 10px 0; }}
        .priority-high {{ background-color: #f8d7da; padding: 10px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
        .priority-medium {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #f39c12; margin: 10px 0; }}
        .priority-low {{ background-color: #d1ecf1; padding: 10px; border-left: 4px solid #17a2b8; margin: 10px 0; }}
        .no-data {{ color: #999; font-style: italic; }}
        .scala-label {{ background-color: #e8f4fd; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        details summary::-webkit-details-marker {{ display: none; }}
        details summary::before {{ content: "▶ "; color: #1E88E5; }}
        details[open] summary::before {{ content: "▼ "; }}
        @media print {{
            body {{ margin: 20px; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.T['titolo']}</h1>
        <div class="subtitle">{self.T['sottotitolo']}</div>
        <div class="date">{self.T['data']}: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
        <div class="no-print" style="margin-top: 15px;">
            <button onclick="window.print()" style="padding: 10px 20px; background-color: #1E88E5; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
                🖨️ {'Stampa / Print' if self.lingua == 'it' else 'Print / Stampa'}
            </button>
        </div>
    </div>
""")

        # PANORAMICA
        html.append(f"""
    <h2>{self.T['panoramica']}</h2>
    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
        <div class="metric-box">
            <div class="metric-value">{len(self.analisi_likert.df_likert)}</div>
            <div class="metric-label">{self.T['totale_risposte']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(self.config['domande_likert'])}</div>
            <div class="metric-label">{self.T['domande_likert']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(self.config['domande_aperte'])}</div>
            <div class="metric-label">{self.T['domande_aperte']}</div>
        </div>
""")

        for scala, alpha in alpha_per_scala.items():
            if alpha is not None and not np.isnan(alpha) and not np.isinf(alpha):
                nome_scala = self.config['scale'][scala]['title']
                html.append(f"""
        <div class="metric-box">
            <div class="metric-value">{alpha:.3f}</div>
            <div class="metric-label">{self.T['affidabilita']} - {nome_scala}</div>
            <div style="font-size: 12px;">{self._get_alpha_interpretazione(alpha)}</div>
        </div>
""")

        html.append("</div>")

        # PROFILO
        html.append(self._genera_sezione_profilo())

        # STATISTICHE DESCRITTIVE
        html.append(f'<h2>{self.T["statistiche_descrittive"]}</h2>')

        if stats.empty:
            html.append('<div class="section"><p class="no-data">⚠️ Nessun dato statistico disponibile.</p></div>')
        else:
            for scala, domande_codici in domande_per_scala.items():
                domande_con_dati = [codice for codice in domande_codici if codice in stats.index]
                if not domande_con_dati:
                    continue

                nome_scala = self.config['scale'][scala]['title']
                html.append(f"""
    <h3><span class="scala-label">{nome_scala}</span></h3>
    <table>
        <thead>
            <tr>
                <th>{self.T['domanda']}</th>
                <th>{self.T['media']}</th>
                <th>{self.T['mediana']}</th>
                <th>{self.T['std']}</th>
                <th>{self.T['n']}</th>
                <th>{self.T['interpretazione']}</th>
            </tr>
        </thead>
        <tbody>
""")
                for codice in domande_con_dati:
                    row = stats.loc[codice]
                    media = row.get('media', 0)
                    if pd.isna(media) or np.isinf(media):
                        continue
                    testo_domanda = self.config['domande_likert'].get(codice, codice)
                    interpretazione = self._get_interpretazione(media)
                    classe = self._get_colore_media(media)
                    html.append(f"""
            <tr>
                <td>{testo_domanda}</td>
                <td class="{classe}">{media:.2f}</td>
                <td>{row.get('mediana', 0):.0f}</td>
                <td>{row.get('std', 0):.2f}</td>
                <td>{row.get('n', 0):.0f}</td>
                <td>{interpretazione}</td>
            </tr>
""")
                html.append("</tbody></table>")

        # ANALISI TESTO (senza parole più frequenti)
        if hasattr(self.analisi_testo, 'df_testo') and self.analisi_testo.df_testo:
            html.append(f'<h2>{self.T["analisi_testo"]}</h2>')
            for col, testi in self.analisi_testo.df_testo.items():
                if not testi:
                    continue
                nome_domanda = self.config['domande_aperte'].get(col, col)
                html.append(f'<h3>{nome_domanda}</h3>')

                # Solo sentiment (rimosse parole frequenti)
                try:
                    sentimenti = self.analisi_testo.analisi_sentiment(col)
                    if sentimenti:
                        total = sum(sentimenti.values())
                        pos_pct = (sentimenti.get('positivo', 0) / total * 100) if total > 0 else 0
                        neu_pct = (sentimenti.get('neutro', 0) / total * 100) if total > 0 else 0
                        neg_pct = (sentimenti.get('negativo', 0) / total * 100) if total > 0 else 0
                        html.append(f"""
    <div style="background:#f8f9fa; padding:12px 18px; border-radius:6px; margin:10px 0;">
        <h4 style="margin-top:0;">🎭 {self.T['sentiment']}</h4>
        <ul style="margin:0;">
            <li>😊 {self.T['positivo']}: {sentimenti.get('positivo', 0)} ({pos_pct:.0f}%)</li>
            <li>😐 {self.T['neutro']}: {sentimenti.get('neutro', 0)} ({neu_pct:.0f}%)</li>
            <li>😞 {self.T['negativo']}: {sentimenti.get('negativo', 0)} ({neg_pct:.0f}%)</li>
        </ul>
    </div>
""")
                except Exception:
                    pass

                # Esempi di risposte negative
                risposte_negative = self.analisi_testo.risposte_per_sentiment(col, 'negativo')
                risposte_esempio = risposte_negative[:3]
                titolo_esempi = ('📝 ' + ('Esempi di Risposte Negative'
                                          if self.lingua == 'it'
                                          else 'Sample Negative Responses'))

                if risposte_esempio:
                    html.append(f'<div class="section"><h4>{titolo_esempi}</h4><ul>')
                    for i, risposta in enumerate(risposte_esempio, 1):
                        risposta_breve = risposta[:200] + ('...' if len(risposta) > 200 else '')
                        html.append(f'<li><strong>{i}.</strong> {risposta_breve}</li>')
                    html.append('</ul></div>')
                else:
                    html.append(f'<div class="section"><h4>{titolo_esempi}</h4>'
                                f'<p class="no-data">Nessuna risposta negativa rilevata.</p></div>')

        # CONCLUSIONI (basate su Likert)
        html.append(f'<h2>{self.T["conclusioni"]}</h2>')
        punti_forza_likert = []
        aree_miglioramento = []

        if not stats.empty:
            for codice, row in stats.iterrows():
                try:
                    media = row.get('media', 0)
                    if pd.isna(media) or np.isinf(media):
                        continue
                    testo = self.config['domande_likert'].get(codice, codice)
                    if media >= 4.0:
                        punti_forza_likert.append((testo, media))
                    elif media <= 3.95:
                        aree_miglioramento.append((testo, media))
                except Exception:
                    continue

        if punti_forza_likert:
            html.append(f'<h3>✅ {self.T["punti_forza"]} (Likert)</h3>')
            for domanda, media in sorted(punti_forza_likert, key=lambda x: -x[1])[:5]:
                html.append(f'<div class="strength"><strong>{domanda}</strong>: '
                            f'{media:.2f} - {self._get_interpretazione(media)}</div>')

        if aree_miglioramento:
            html.append(f'<h3>⚠️ {self.T["aree_miglioramento"]} (Likert)</h3>')
            for domanda, media in sorted(aree_miglioramento, key=lambda x: x[1])[:5]:
                html.append(f'<div class="weakness"><strong>{domanda}</strong>: '
                            f'{media:.2f} - {self._get_interpretazione(media)}</div>')

        # ANALISI TEMATICA (include problemi+soluzioni e punti di forza)
        html.append(self._genera_sezione_tematica())

        # FOOTER
        html.append(f"""
    <div class="footer">
        Report generato automaticamente da MANILA Analysis Tool<br>
        {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </div>
</body>
</html>
""")

        html_content = '\n'.join(html)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return None
        else:
            return html_content

    def genera_pdf(self, output_file):
        try:
            from weasyprint import HTML
            HTML(string=self.genera_html()).write_pdf(output_file)
            return True
        except ImportError:
            print("⚠️ Installa WeasyPrint: pip install weasyprint")
            return False