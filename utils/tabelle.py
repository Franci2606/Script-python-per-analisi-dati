# utils/tabelle.py
# Generazione di tabelle di riepilogo per il questionario

import pandas as pd
import numpy as np
from pathlib import Path

class GeneratoreTabelle:
    def __init__(self, output_dir='output/tabelle'):
        """
        Inizializza il generatore di tabelle
        
        Args:
            output_dir: Directory dove salvare le tabelle
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def tabella_statistiche(self, analisi_likert, domande_likert):
        """
        Genera una tabella con le statistiche descrittive
        
        Args:
            analisi_likert: Istanza di AnalisiLikert
            domande_likert: Dizionario {codice: testo}
            
        Returns:
            DataFrame con le statistiche
        """
        stats = analisi_likert.statistiche_descrittive()
        
        if stats.empty:
            return pd.DataFrame()
        
        # Aggiungi colonna per la scala
        stats['Scala'] = stats.index.map(
            lambda x: analisi_likert.df_scale_info.get(
                [k for k, v in domande_likert.items() if v == x][0] if x in domande_likert.values() else None,
                'N/A'
            ) if x in domande_likert.values() else 'N/A'
        )
        
        # Riordina le colonne
        colonne = ['Scala', 'media', 'mediana', 'moda', 'std', 'min', 'max', 'n', 'mancanti']
        stats = stats[[c for c in colonne if c in stats.columns]]
        
        # Arrotonda i valori numerici
        for col in ['media', 'mediana', 'std']:
            if col in stats.columns:
                stats[col] = stats[col].round(2)
        
        return stats
    
    def tabella_distribuzione(self, df_likert, domande_likert):
        """
        Genera una tabella con la distribuzione delle risposte
        
        Args:
            df_likert: DataFrame con i dati Likert
            domande_likert: Dizionario {codice: testo}
            
        Returns:
            DataFrame con le distribuzioni
        """
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
                    'Totale': counts.sum()
                }
        
        df_dist = pd.DataFrame(distribuzioni).T
        df_dist.columns = ['1', '2', '3', '4', '5', 'Totale']
        return df_dist
    
    def tabella_alpha_cronbach(self, alpha_per_scala, scale):
        """
        Genera una tabella con i valori di alpha di Cronbach
        
        Args:
            alpha_per_scala: Dizionario {scala: alpha}
            scale: Configurazione delle scale
            
        Returns:
            DataFrame con i risultati
        """
        if not alpha_per_scala:
            return pd.DataFrame()
        
        data = []
        for scala, alpha in alpha_per_scala.items():
            if alpha is not None:
                if alpha >= 0.9:
                    interpretazione = 'Eccellente'
                elif alpha >= 0.8:
                    interpretazione = 'Buona'
                elif alpha >= 0.7:
                    interpretazione = 'Accettabile'
                elif alpha >= 0.6:
                    interpretazione = 'Dubbia'
                else:
                    interpretazione = 'Inaccettabile'
                
                data.append({
                    'Scala': scale[scala]['title'],
                    'Alpha di Cronbach': round(alpha, 3),
                    'Interpretazione': interpretazione
                })
        
        return pd.DataFrame(data)
    
    def tabella_correlazione(self, analisi_likert, domande_likert):
        """
        Genera una tabella con la matrice di correlazione
        
        Args:
            analisi_likert: Istanza di AnalisiLikert
            domande_likert: Dizionario {codice: testo}
            
        Returns:
            DataFrame con la matrice di correlazione
        """
        colonne = [col for col in domande_likert.keys() 
                  if col in analisi_likert.df_likert.columns 
                  and not col.endswith('_inverse')]
        
        if len(colonne) < 2:
            return pd.DataFrame()
        
        corr = analisi_likert.df_likert[colonne].corr()
        corr.index = [domande_likert.get(col, col) for col in corr.index]
        corr.columns = [domande_likert.get(col, col) for col in corr.columns]
        
        return corr.round(3)
    
    def tabella_risposte_aperte_riepilogo(self, analisi_testo):
        """
        Genera un riepilogo delle domande aperte
        
        Args:
            analisi_testo: Istanza di AnalisiTesto
            
        Returns:
            DataFrame con il riepilogo
        """
        return analisi_testo.riepilogo_domande_aperte()
    
    def salva_csv(self, df, nome_file):
        """
        Salva un DataFrame come CSV nella directory di output
        
        Args:
            df: DataFrame da salvare
            nome_file: Nome del file (senza estensione)
            
        Returns:
            Percorso del file salvato
        """
        if df is None or df.empty:
            return None
        
        percorso = f'{self.output_dir}/{nome_file}.csv'
        df.to_csv(percorso, index=True)
        return percorso
    
    def esporta_tutte_tabelle(self, analisi_likert, analisi_testo, domande_likert, scale):
        """
        Esporta tutte le tabelle di riepilogo
        
        Args:
            analisi_likert: Istanza di AnalisiLikert
            analisi_testo: Istanza di AnalisiTesto
            domande_likert: Dizionario {codice: testo}
            scale: Configurazione delle scale
            
        Returns:
            Dizionario con i percorsi dei file salvati
        """
        risultati = {}
        
        # Statistiche descrittive
        stats = self.tabella_statistiche(analisi_likert, domande_likert)
        percorso = self.salva_csv(stats, 'statistiche_descrittive')
        if percorso:
            risultati['statistiche'] = percorso
        
        # Distribuzione
        dist = self.tabella_distribuzione(analisi_likert.df_likert, domande_likert)
        percorso = self.salva_csv(dist, 'distribuzione_risposte')
        if percorso:
            risultati['distribuzione'] = percorso
        
        # Alpha di Cronbach
        alpha = analisi_likert.calcola_alpha_cronbach()
        df_alpha = self.tabella_alpha_cronbach(alpha, scale)
        percorso = self.salva_csv(df_alpha, 'alpha_cronbach')
        if percorso:
            risultati['alpha'] = percorso
        
        # Matrice di correlazione
        corr = self.tabella_correlazione(analisi_likert, domande_likert)
        percorso = self.salva_csv(corr, 'matrice_correlazione')
        if percorso:
            risultati['correlazione'] = percorso
        
        # Riepilogo domande aperte
        riepilogo = self.tabella_risposte_aperte_riepilogo(analisi_testo)
        percorso = self.salva_csv(riepilogo, 'riepilogo_domande_aperte')
        if percorso:
            risultati['aperte'] = percorso
        
        return risultati