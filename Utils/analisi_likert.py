# utils/analisi_likert.py
# Analisi delle domande a scala Likert (con supporto scale multiple)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class AnalisiLikert:
    def __init__(self, df, domande_likert, mappatura_scale, scale, domande_inverse=None):
        """
        Inizializza l'analisi Likert con supporto per scale multiple
        
        Args:
            df: DataFrame con i dati
            domande_likert: Dizionario {codice: testo_domanda}
            mappatura_scale: Dizionario {codice: 'accordo'|'utilita'}
            scale: Dizionario con le configurazioni delle scale
            domande_inverse: Lista di domande con codifica inversa
        """
        self.df = df
        self.domande_likert = domande_likert
        self.mappatura_scale = mappatura_scale
        self.scale = scale
        self.domande_inverse = domande_inverse or []
        self._prepara_dati_likert()
    
    def _prepara_dati_likert(self):
        """Prepara i dati delle domande Likert"""
        print("📊 Preparazione dati Likert...")
        self.df_likert = pd.DataFrame()
        self.df_scale_info = {}  # Per tenere traccia della scala di ogni domanda
        
        for col in self.domande_likert.keys():
            if col in self.df.columns:
                self.df_likert[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Salva la scala associata
                self.df_scale_info[col] = self.mappatura_scale.get(col, 'accordo')
                print(f"  ✓ {col}: {self.df_likert[col].count()} risposte valide")
            else:
                print(f"  ⚠️ {col}: NON TROVATA nel DataFrame!")
        
        # Applica codifica inversa
        for domanda in self.domande_inverse:
            if domanda in self.df_likert.columns:
                max_val = self.df_likert[domanda].max()
                self.df_likert[f'{domanda}_inverse'] = max_val + 1 - self.df_likert[domanda]
                print(f"  ✓ Codifica inversa applicata a {domanda}")
        
        print(f"  ✓ Dati Likert preparati: {len(self.df_likert)} righe, {len(self.df_likert.columns)} colonne")
    
    def statistiche_descrittive(self):
        """Calcola statistiche descrittive per ogni domanda (con testo come indice)"""
        stats = {}
        for col in self.domande_likert.keys():
            if col in self.df_likert.columns:
                dati = self.df_likert[col].dropna()
                if len(dati) > 0:
                    scala = self.df_scale_info.get(col, 'accordo')
                    stats[col] = {
                        'scala': self.scale[scala]['title'],
                        'media': dati.mean(),
                        'mediana': dati.median(),
                        'moda': dati.mode()[0] if not dati.mode().empty else None,
                        'std': dati.std(),
                        'min': dati.min(),
                        'max': dati.max(),
                        'n': len(dati),
                        'mancanti': self.df_likert[col].isna().sum()
                    }
        
        df_stats = pd.DataFrame(stats).T
        if not df_stats.empty:
            df_stats.index = [self.domande_likert.get(idx, idx) for idx in df_stats.index]
        return df_stats
    
    def statistiche_descrittive_con_codici(self):
        """
        Calcola statistiche descrittive per ogni domanda usando i CODICI come indice
        Restituisce un DataFrame con indice = codice domanda (Q5, Q6, ecc.)
        Questo metodo è usato dal ReportGenerator.
        """
        stats = {}
        for col in self.domande_likert.keys():
            if col in self.df_likert.columns:
                dati = self.df_likert[col].dropna()
                if len(dati) > 0:
                    scala = self.df_scale_info.get(col, 'accordo')
                    stats[col] = {
                        'codice': col,
                        'scala': self.scale[scala]['title'],
                        'media': dati.mean(),
                        'mediana': dati.median(),
                        'moda': dati.mode()[0] if not dati.mode().empty else None,
                        'std': dati.std(),
                        'min': dati.min(),
                        'max': dati.max(),
                        'n': len(dati),
                        'mancanti': self.df_likert[col].isna().sum()
                    }
        
        df_stats = pd.DataFrame(stats).T
        
        return df_stats
    
    def calcola_alpha_cronbach(self, scala=None):
        """
        Calcola l'alpha di Cronbach per una scala specifica o per tutte
        
        Args:
            scala: 'accordo', 'utilita', o None per tutte
            
        Returns:
            Dizionario o valore singolo
        """
        if scala is None:
            # Calcola per ogni scala
            risultati = {}
            for s in set(self.df_scale_info.values()):
                risultati[s] = self._calcola_alpha_per_scala(s)
            return risultati
        else:
            return self._calcola_alpha_per_scala(scala)
    
    def _calcola_alpha_per_scala(self, scala):
        """Calcola alpha di Cronbach per una specifica scala"""
        # Seleziona le domande di questa scala (escludendo quelle inverse)
        domande = [col for col, s in self.df_scale_info.items() 
                  if s == scala and col in self.df_likert.columns 
                  and not col.endswith('_inverse')]
        
        if len(domande) > 1:
            df_alpha = self.df_likert[domande].dropna()
            if len(df_alpha) > 1:
                n_items = df_alpha.shape[1]
                varianze = df_alpha.var()
                varianza_totale = df_alpha.sum(axis=1).var()
                alpha = (n_items / (n_items - 1)) * (1 - varianze.sum() / varianza_totale)
                return alpha
        return None
    
    def get_domande_per_scala(self):
        """Restituisce le domande raggruppate per scala"""
        domande_per_scala = {}
        for col, scala in self.df_scale_info.items():
            if col in self.domande_likert and not col.endswith('_inverse'):
                if scala not in domande_per_scala:
                    domande_per_scala[scala] = []
                domande_per_scala[scala].append(col)
        return domande_per_scala
    
    def get_risposte_domanda(self, colonna):
        """Restituisce le risposte per una specifica domanda"""
        if colonna in self.df_likert.columns:
            return self.df_likert[colonna].dropna()
        return pd.Series()