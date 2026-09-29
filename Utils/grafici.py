# utils/grafici.py - Versione con numeri domande e palette Verde/Arancione/Rosso

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import textwrap
import re
from matplotlib.colors import LinearSegmentedColormap

class GeneratoreGrafici:
    def __init__(self, output_dir='output/grafici'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_style("whitegrid")
        
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14

    # ============================================================
    # ESTRAZIONE PAROLE CHIAVE + NUMERO DOMANDA
    # ============================================================
    def _estrai_parole_chiave(self, testo, max_parole=5):
        """Estrae parole chiave rimuovendo stop-words italiane."""
        if not testo:
            return ""
        
        stop_words = {
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
            'e', 'ed', 'o', 'ma', 'però', 'anche', 'come', 'dove', 'quando',
            'che', 'chi', 'cui', 'non', 'più', 'meno', 'molto', 'poco',
            'essere', 'avere', 'fare', 'si', 'mi', 'ti', 'ci', 'vi',
            'del', 'dello', 'della', 'dei', 'degli', 'delle', 'al', 'allo',
            'alla', 'ai', 'agli', 'alle', 'dal', 'dallo', 'dalla', 'dai',
            'dagli', 'dalle', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle',
            'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle', 'è', 'sono', 'ha',
            'questa', 'questo', 'questi', 'queste', 'quel', 'quella', 'quei'
        }
        
        testo_pulito = re.sub(r'[^\w\s]', '', testo.lower())
        parole = testo_pulito.split()
        parole_significative = [p for p in parole if p not in stop_words and len(p) > 3]
        
        if not parole_significative:
            return self._truncate_text(testo, max_words=max_parole)
        
        if len(parole_significative) > max_parole:
            risultato = []
            for p in parole:
                if p in parole_significative:
                    risultato.append(p)
                if len(risultato) >= max_parole:
                    break
            return ' '.join(risultato).capitalize() + '...'
        
        return testo

    def _estrai_numero_domanda(self, chiave_colonna, testo_domanda):
        """
        Estrae il numero della domanda.
        Prova prima dalla chiave della colonna (es. 'q5', 'domanda_5'),
        poi dal testo (es. '5. L'applicazione...').
        """
        # Prova dalla chiave colonna
        match = re.search(r'(\d+)', str(chiave_colonna))
        if match:
            return match.group(1)
        
        # Prova dal testo
        match = re.match(r'^\s*(\d+)[\.\)\s]', testo_domanda)
        if match:
            return match.group(1)
        
        return None

    def _etichetta_con_numero(self, chiave_colonna, testo_domanda, lingua='it', max_parole=5):
        """
        Crea un'etichetta nel formato '5. Parole chiave...'
        mantenendo il numero della domanda.
        """
        numero = self._estrai_numero_domanda(chiave_colonna, testo_domanda)
        
        if lingua == 'en':
            contenuto = testo_domanda
        else:
            contenuto = self._estrai_parole_chiave(testo_domanda, max_parole=max_parole)
        
        if numero:
            return f"{numero}. {contenuto}"
        return contenuto

    def _truncate_text(self, text, max_words=5, max_chars=30):
        if not text: return ""
        if len(text) <= max_chars: return text
        parole = text.split()
        if len(parole) > max_words:
            return ' '.join(parole[:max_words]) + '...'
        return text[:max_chars] + '...'

    # ============================================================
    # MATRICE DI CORRELAZIONE - VERDE/ARANCIONE/ROSSO
    # ============================================================
    def matrice_correlazione(self, df_likert, domande_likert, titolo="Matrice di Correlazione", salva=True, lingua='it'):
        try:
            colonne = [col for col in domande_likert.keys() 
                      if col in df_likert.columns and not col.endswith('_inverse')]
            
            if len(colonne) < 2:
                print("⚠️ Servono almeno 2 domande per la matrice di correlazione")
                return None
            
            df_corr = df_likert[colonne].dropna()
            if len(df_corr) < 3:
                print("⚠️ Dati insufficienti per la correlazione")
                return None
            
            corr = df_corr.corr()
            
            # 🔥 Etichette con numero domanda + parole chiave
            etichette_brevi = {}
            for col in corr.columns:
                etichette_brevi[col] = self._etichetta_con_numero(
                    col, domande_likert[col], lingua=lingua, max_parole=5
                )
            
            corr.index = [etichette_brevi[col] for col in corr.index]
            corr.columns = [etichette_brevi[col] for col in corr.columns]
            
            n_items = len(colonne)
            if lingua == 'en':
                figsize = (14, 12) if n_items <= 8 else (16, 14)
                fontsize = 9 if n_items <= 8 else 8
            else:
                figsize = (12, 10) if n_items <= 8 else (14, 12)
                fontsize = 9 if n_items <= 8 else 8
            
            fig, ax = plt.subplots(figsize=figsize)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # 🔥 Palette Verde (alto) - Arancione (0) - Rosso (basso)
            # Verde = correlazione positiva alta (favorevole)
            # Arancione = nessuna correlazione (neutro)
            # Rosso = correlazione negativa (sfavorevole)
            cmap = LinearSegmentedColormap.from_list(
                'verde_arancio_rosso',
                ['#c0392b',  # Rosso scuro (-1)
                 '#e74c3c',  # Rosso (-0.5)
                 '#f39c12',  # Arancione (0)
                 '#f1c40f',  # Giallo-arancio (0.5)
                 '#2ecc71']  # Verde (1)
            )
            
            label_cbar = 'Correlation' if lingua == 'en' else 'Correlazione'
            sns.heatmap(corr, 
                       mask=mask,
                       annot=True, 
                       cmap=cmap,
                       center=0,
                       square=True,
                       linewidths=0.5,
                       fmt='.2f',
                       vmin=-1, vmax=1,
                       cbar_kws={"shrink": 0.8, "label": label_cbar},
                       annot_kws={'size': fontsize},
                       ax=ax)
            
            ax.set_title(titolo, fontsize=14)
            
            if lingua == 'en':
                rotation = 45 if n_items <= 6 else 90
            else:
                rotation = 90 if n_items > 5 else 45
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center', fontsize=fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
            
            plt.tight_layout()
            
            if salva:
                suffix = 'en' if lingua == 'en' else 'it'
                plt.savefig(f'{self.output_dir}/correlazione_{suffix}_{datetime.now().strftime("%Y%m%d_%H%M")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                return None
            else:
                return fig
        except Exception as e:
            print(f"⚠️ Errore nella matrice di correlazione: {e}")
            
            return None

    # ============================================================
    # GRAFICO LIKERT - VERDE/ARANCIONE/ROSSO
    # ============================================================
    def grafico_likert(self, df_likert, domande_likert, scale_info, titolo=None, salva=True, lingua='it'):
        try:
            colonne = [col for col in domande_likert.keys() if col in df_likert.columns and not col.endswith('_inverse')]
            
            if len(colonne) < 1:
                print("⚠️ Nessuna domanda valida per il grafico Likert")
                return None
            
            df_plot = df_likert[colonne].copy()
            
            # 🔥 Etichette con numero domanda + parole chiave
            etichette_brevi = {}
            for col in df_plot.columns:
                etichette_brevi[col] = self._etichetta_con_numero(
                    col, domande_likert[col], lingua=lingua, max_parole=6
                )
            
            df_plot.columns = [etichette_brevi[col] for col in df_plot.columns]
            
            medie = df_plot.mean().sort_values()
            df_plot = df_plot[medie.index]
            
            n_items = len(colonne)
            if lingua == 'en':
                height = max(6, n_items * 0.6 + 2)
                width = max(12, n_items * 0.4 + 8)
            else:
                height = max(6, n_items * 0.6 + 2)
                width = max(10, n_items * 0.4 + 8)
            
            fig, ax = plt.subplots(figsize=(width, height))
            
            nome_x = 'Question' if lingua == 'en' else 'Domanda'
            nome_y = 'Rating' if lingua == 'en' else 'Valutazione'
            df_melted = df_plot.melt(var_name=nome_x, value_name=nome_y)
            fontsize = 10 if n_items <= 5 else 8
            
            # 🔥 Palette a 3 colori: Verde / Arancione / Rosso
            palette = []
            for col in df_plot.columns:
                media_val = df_plot[col].mean()
                if media_val >= 4.0:
                    palette.append('#2ecc71')  # Verde (Favorevole)
                elif media_val >= 3.0:
                    palette.append('#f39c12')  # Arancione (Neutro)
                else:
                    palette.append('#e74c3c')  # Rosso (Sfavorevole)
            
            sns.boxplot(data=df_melted, x=nome_x, y=nome_y, palette=palette, ax=ax)
            
            ax.set_ylabel(nome_y, fontsize=12)
            ax.set_ylim(0.5, 5.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)
            
            if titolo:
                ax.set_title(titolo, fontsize=14)
            
            plt.tight_layout()
            
            if salva:
                suffix = 'en' if lingua == 'en' else 'it'
                plt.savefig(f'{self.output_dir}/likert_{suffix}_{datetime.now().strftime("%Y%m%d_%H%M")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                return None
            else:
                return fig
        except Exception as e:
            print(f"⚠️ Errore nel grafico Likert: {e}")
            
            return None

    # ============================================================
    # GRAFICO MEDIE PER SEZIONE - VERDE/ARANCIONE/ROSSO
    # ============================================================
    def grafico_medie_per_sezione(self, df_likert, sezioni, domande_likert, salva=True, lingua='it'):
        try:
            medie_per_sezione = {}
            
            if lingua == 'en':
                abbreviazioni = {
                    'Completeness of components': 'Completeness',
                    'Accuracy of results': 'Accuracy',
                    'Configuration flow': 'Configuration',
                    'Web interface usability': 'Usability',
                    'Additional features': 'Features',
                    'Overall rating': 'Overall',
                    'Completeness and accuracy of components': 'Completeness',
                    'Accuracy and clarity of results': 'Accuracy',
                    'Completeness of the configuration flow': 'Configuration',
                    'Usability of the web interface': 'Usability',
                    'Usefulness of additional features': 'Features',
                }
                titolo = 'Mean ratings by section'
                label_y = 'Mean rating'
                label_neutro = 'Neutral'
                label_accordo = 'Agree'
            else:
                abbreviazioni = {
                    'Completezza dei componenti': 'Completezza',
                    'Accuratezza dei risultati': 'Accuratezza',
                    'Flusso di configurazione': 'Configurazione',
                    'Usabilità interfaccia web': 'Usabilità',
                    'Funzionalità aggiuntive': 'Funzionalità',
                    'Valutazione complessiva': 'Valutazione',
                    'Completezza e accuratezza dei componenti': 'Completezza',
                    'Accuratezza e chiarezza dei risultati': 'Accuratezza',
                    'Completezza del flusso di configurazione': 'Configurazione',
                    'Usabilità dell\'interfaccia web': 'Usabilità',
                    'Utilità delle funzionalità aggiuntive': 'Funzionalità',
                }
                titolo = 'Media delle valutazioni per sezione'
                label_y = 'Media delle valutazioni'
                label_neutro = 'Neutro'
                label_accordo = "D'accordo"
            
            for sezione, domande in sezioni.items():
                domande_presenti = [d for d in domande if d in df_likert.columns and not d.endswith('_inverse')]
                if domande_presenti:
                    medie = df_likert[domande_presenti].mean()
                    nome_mostrato = abbreviazioni.get(sezione, sezione)
                    medie_per_sezione[nome_mostrato] = medie.mean()
            
            if not medie_per_sezione:
                return None
            
            n_sezioni = len(medie_per_sezione)
            width = max(8, n_sezioni * 0.8 + 2)
            height = max(5, n_sezioni * 0.4 + 3)
            
            fig, ax = plt.subplots(figsize=(width, height))
            
            sorted_items = sorted(medie_per_sezione.items(), key=lambda x: x[1])
            nomi = [item[0] for item in sorted_items]
            valori = [item[1] for item in sorted_items]
            
            # 🔥 Palette a 3 colori: Verde / Arancione / Rosso
            colors = []
            for v in valori:
                if v >= 4.0:
                    colors.append('#2ecc71')  # Verde (Favorevole)
                elif v >= 3.0:
                    colors.append('#f39c12')  # Arancione (Neutro)
                else:
                    colors.append('#e74c3c')  # Rosso (Sfavorevole)
            
            bars = ax.barh(nomi, valori, color=colors, edgecolor='black', alpha=0.8)
            
            fontsize = 11 if n_sezioni <= 5 else 9
            for bar, val in zip(bars, valori):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                       va='center', fontsize=fontsize)
            
            ax.axvline(3, color='gray', linestyle='--', alpha=0.7, label=label_neutro)
            ax.axvline(4, color='green', linestyle='--', alpha=0.5, label=label_accordo)
            
            ax.set_xlabel(label_y, fontsize=12)
            ax.set_title(titolo, fontsize=14)
            ax.set_xlim(1, 5.5)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')
            
            if lingua == 'en':
                label_fontsize = 10
            else:
                max_label_len = max([len(str(label)) for label in nomi]) if nomi else 0
                label_fontsize = 10 if max_label_len <= 20 else 8
            
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=label_fontsize)
            
            plt.tight_layout()
            
            if salva:
                suffix = 'en' if lingua == 'en' else 'it'
                plt.savefig(f'{self.output_dir}/medie_per_sezione_{suffix}.png', dpi=300, bbox_inches='tight')
                plt.close()
                return None
            else:
                return fig
        except Exception as e:
            print(f"⚠️ Errore nel grafico delle medie: {e}")
            return None