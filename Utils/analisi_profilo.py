# utils/analisi_profilo.py
# Analisi delle domande di profilo (Q1-Q4): categoriche e ordinali

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AnalisiProfilo:
    def __init__(self, df, domande_profilo, opzioni_multipla=None,
                 labels_profilo=None, mapping_valori_profilo=None):
        """
        df: DataFrame completo
        domande_profilo: dict {codice: testo_domanda}
        opzioni_multipla: dict {codice: [opzioni]} per Q1, Q2
        labels_profilo: dict {codice: {1: 'etichetta', ...}} per Q3, Q4
        mapping_valori_profilo: dict {codice: {'etichetta': 1, ...}} per conversione
        """
        self.df = df
        self.domande_profilo = domande_profilo
        self.opzioni_multipla = opzioni_multipla or {}
        self.labels_profilo = labels_profilo or {}
        self.mapping_valori_profilo = mapping_valori_profilo or {}
        self._prepara_dati()

    def _prepara_dati(self):
        """Estrae le colonne di profilo se presenti"""
        self.df_profilo = pd.DataFrame()
        for col in self.domande_profilo.keys():
            if col in self.df.columns:
                self.df_profilo[col] = self.df[col]

    def _T(self, lingua):
        """Restituisce le etichette tradotte."""
        if lingua == 'en':
            return {
                'risposta': 'Response',
                'frequenza': 'Frequency',
                'percentuale': 'Percentage',
                'numero_risposte': 'Number of responses',
                'media': 'Mean',
            }
        return {
            'risposta': 'Risposta',
            'frequenza': 'Frequenza',
            'percentuale': 'Percentuale',
            'numero_risposte': 'Numero risposte',
            'media': 'Media',
        }

    # ============================================================
    # CATEGORICHE (Q1, Q2)
    # ============================================================
    def frequenze(self, colonna, lingua='it'):
        """Restituisce le frequenze (value_counts) per una domanda categorica"""
        if colonna not in self.df_profilo.columns:
            return None
        serie = self.df_profilo[colonna].dropna()
        if serie.empty:
            return None
        T = self._T(lingua)
        freq = serie.value_counts()
        perc = (freq / freq.sum() * 100).round(1)
        return pd.DataFrame({
            T['risposta']: freq.index,
            T['frequenza']: freq.values,
            T['percentuale']: perc.values
        })

    # ============================================================
    # ORDINALI (Q3, Q4)
    # ============================================================
    def _valori_a_numerico(self, colonna):
        """Converte i valori di una colonna ordinale in numeri."""
        if colonna not in self.df_profilo.columns:
            return None

        serie = self.df_profilo[colonna].dropna()
        if serie.empty:
            return None

        serie_num = pd.to_numeric(serie, errors='coerce')

        if serie_num.isna().all():
            mapping = self.mapping_valori_profilo.get(colonna, {})
            if mapping:
                inverso = {str(k).strip().lower(): int(v) for k, v in mapping.items()}
                serie_num = serie.apply(
                    lambda x: inverso.get(str(x).strip().lower(), np.nan)
                )

            if serie_num.isna().all():
                labels = self.labels_profilo.get(colonna, {})
                if labels:
                    inverso = {str(v).strip().lower(): int(k) for k, v in labels.items()}
                    serie_num = serie.apply(
                        lambda x: inverso.get(str(x).strip().lower(), np.nan)
                    )

        return serie_num.dropna()

    def statistiche_ordinali(self, colonna):
        """Media, mediana, distribuzione per una domanda ordinale."""
        serie = self._valori_a_numerico(colonna)
        if serie is None or serie.empty:
            return None

        return {
            'media': serie.mean(),
            'mediana': serie.median(),
            'std': serie.std(),
            'n': len(serie),
            'distribuzione': serie.value_counts().sort_index()
        }

    # ============================================================
    # GRAFICI
    # ============================================================
    def grafico_categorico(self, colonna, titolo, salva=False, lingua='it'):
        """Grafico a barre orizzontali per Q1/Q2"""
        freq = self.frequenze(colonna, lingua=lingua)
        if freq is None or freq.empty:
            return None

        T = self._T(lingua)
        col_risposta = T['risposta']
        col_frequenza = T['frequenza']
        col_percentuale = T['percentuale']

        fig, ax = plt.subplots(figsize=(10, max(3, len(freq) * 0.5)))
        y_pos = range(len(freq))
        ax.barh(y_pos, freq[col_frequenza], color='#1E88E5')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(freq[col_risposta])
        ax.invert_yaxis()
        ax.set_xlabel(T['numero_risposte'])
        ax.set_title(titolo, fontweight='bold')

        for i, (v, p) in enumerate(zip(freq[col_frequenza], freq[col_percentuale])):
            ax.text(v + 0.1, i, f'{v} ({p}%)', va='center')

        plt.tight_layout()
        return fig

    def grafico_ordinale(self, colonna, titolo, labels=None, salva=False, lingua='it'):
        """Grafico a barre ordinate per Q3/Q4 (scala 1-5)"""
        stats = self.statistiche_ordinali(colonna)
        if stats is None:
            return None

        dist = stats['distribuzione']
        if dist.empty:
            return None

        if labels is None:
            labels = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}

        T = self._T(lingua)

        fig, ax = plt.subplots(figsize=(9, 4))
        x = list(dist.index)
        ax.bar(x, dist.values, color='#43A047', width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels.get(int(v), str(v)) for v in x],
            rotation=15, ha='right'
        )
        ax.set_ylabel(T['numero_risposte'])
        ax.set_title(
            f"{titolo} — {T['media']}: {stats['media']:.2f} (n={stats['n']})",
            fontweight='bold'
        )

        for xi, yi in zip(x, dist.values):
            ax.text(xi, yi + 0.1, str(yi), ha='center')

        plt.tight_layout()
        return fig