# utils/tabelle.py
# Generazione di tabelle di riepilogo per il questionario

import pandas as pd
import numpy as np
from pathlib import Path


class GeneratoreTabelle:
    def __init__(self, output_dir='output/tabelle'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _T(self, lingua):
        """Restituisce le etichette tradotte."""
        if lingua == 'en':
            return {
                'scala': 'Scale',
                'alpha': "Cronbach's Alpha",
                'interpretazione': 'Interpretation',
                'totale': 'Total',
                'eccellente': 'Excellent',
                'buona': 'Good',
                'accettabile': 'Acceptable',
                'dubbia': 'Questionable',
                'inaccettabile': 'Unacceptable',
                'risposta': 'Response',
                'frequenza': 'Frequency',
                'percentuale': 'Percentage',
            }
        else:
            return {
                'scala': 'Scala',
                'alpha': 'Alpha di Cronbach',
                'interpretazione': 'Interpretazione',
                'totale': 'Totale',
                'eccellente': 'Eccellente',
                'buona': 'Buona',
                'accettabile': 'Accettabile',
                'dubbia': 'Dubbia',
                'inaccettabile': 'Inaccettabile',
                'risposta': 'Risposta',
                'frequenza': 'Frequenza',
                'percentuale': 'Percentuale',
            }

    def tabella_statistiche(self, analisi_likert, domande_likert, lingua='it'):
        stats = analisi_likert.statistiche_descrittive()
        if stats.empty:
            return pd.DataFrame()

        T = self._T(lingua)

        stats[T['scala']] = stats.index.map(
            lambda x: analisi_likert.df_scale_info.get(
                [k for k, v in domande_likert.items() if v == x][0] if x in domande_likert.values() else None,
                'N/A'
            ) if x in domande_likert.values() else 'N/A'
        )

        colonne = [T['scala'], 'media', 'mediana', 'moda', 'std', 'min', 'max', 'n', 'mancanti']
        stats = stats[[c for c in colonne if c in stats.columns]]

        for col in ['media', 'mediana', 'std']:
            if col in stats.columns:
                stats[col] = stats[col].round(2)

        return stats

    def tabella_distribuzione(self, df_likert, domande_likert, lingua='it'):
        T = self._T(lingua)
        distribuzioni = {}
        for col in domande_likert.keys():
            if col in df_likert.columns and not col.endswith('_inverse'):
                counts = df_likert[col].value_counts().sort_index()
                distribuzioni[domande_likert[col]] = {
                    1: counts.get(1, 0),
                    2: counts.get(2, 0),
                    3: counts.get(3, 0),
                    4: counts.get(4, 0),
                    5: counts.get(5, 0),
                    T['totale']: counts.sum()
                }

        df_dist = pd.DataFrame(distribuzioni).T
        df_dist.columns = ['1', '2', '3', '4', '5', T['totale']]
        return df_dist

    def tabella_alpha_cronbach(self, alpha_per_scala, scale, lingua='it'):
        if not alpha_per_scala:
            return pd.DataFrame()

        T = self._T(lingua)
        data = []
        for scala, alpha in alpha_per_scala.items():
            if alpha is not None:
                if alpha >= 0.9:
                    interpretazione = T['eccellente']
                elif alpha >= 0.8:
                    interpretazione = T['buona']
                elif alpha >= 0.7:
                    interpretazione = T['accettabile']
                elif alpha >= 0.6:
                    interpretazione = T['dubbia']
                else:
                    interpretazione = T['inaccettabile']

                data.append({
                    T['scala']: scale[scala]['title'],
                    T['alpha']: round(alpha, 3),
                    T['interpretazione']: interpretazione
                })

        return pd.DataFrame(data)

    def tabella_correlazione(self, analisi_likert, domande_likert, lingua='it'):
        colonne = [col for col in domande_likert.keys()
                   if col in analisi_likert.df_likert.columns
                   and not col.endswith('_inverse')]

        if len(colonne) < 2:
            return pd.DataFrame()

        corr = analisi_likert.df_likert[colonne].corr()
        corr.index = [domande_likert.get(col, col) for col in corr.index]
        corr.columns = [domande_likert.get(col, col) for col in corr.columns]

        return corr.round(3)

    def tabella_risposte_aperte_riepilogo(self, analisi_testo, lingua='it'):
        return analisi_testo.riepilogo_domande_aperte()

    def salva_csv(self, df, nome_file):
        if df is None or df.empty:
            return None
        percorso = f'{self.output_dir}/{nome_file}.csv'
        df.to_csv(percorso, index=True)
        return percorso

    def esporta_tutte_tabelle(self, analisi_likert, analisi_testo, domande_likert, scale, lingua='it'):
        risultati = {}

        stats = self.tabella_statistiche(analisi_likert, domande_likert, lingua=lingua)
        percorso = self.salva_csv(stats, 'statistiche_descrittive')
        if percorso:
            risultati['statistiche'] = percorso

        dist = self.tabella_distribuzione(analisi_likert.df_likert, domande_likert, lingua=lingua)
        percorso = self.salva_csv(dist, 'distribuzione_risposte')
        if percorso:
            risultati['distribuzione'] = percorso

        alpha = analisi_likert.calcola_alpha_cronbach()
        df_alpha = self.tabella_alpha_cronbach(alpha, scale, lingua=lingua)
        percorso = self.salva_csv(df_alpha, 'alpha_cronbach')
        if percorso:
            risultati['alpha'] = percorso

        corr = self.tabella_correlazione(analisi_likert, domande_likert, lingua=lingua)
        percorso = self.salva_csv(corr, 'matrice_correlazione')
        if percorso:
            risultati['correlazione'] = percorso

        riepilogo = self.tabella_risposte_aperte_riepilogo(analisi_testo, lingua=lingua)
        percorso = self.salva_csv(riepilogo, 'riepilogo_domande_aperte')
        if percorso:
            risultati['aperte'] = percorso

        return risultati