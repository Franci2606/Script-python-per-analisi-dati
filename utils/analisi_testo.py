# utils/analisi_testo.py
# Analisi delle risposte aperte del questionario

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from config import DOMANDE_APERTE_IT, DOMANDE_APERTE_EN

# Hugging Face Transformers per l'analisi del sentiment
from transformers import pipeline
import torch

# Download delle risorse NLTK (solo la prima volta)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')


class AnalisiTesto:
    def __init__(self, df, domande_aperte, lingua='it', sentiment_pipeline=None):
        self.df = df
        self.domande_aperte = domande_aperte
        self.lingua = lingua
        self.lang = 'italian' if lingua == 'it' else 'english'
        self.stopwords_set = set(stopwords.words(self.lang))
        self.stemmer = SnowballStemmer(self.lang)
    
        # Se la pipeline è già in cache (Streamlit), usala; altrimenti caricala
        if sentiment_pipeline is not None:
            self.sentiment_pipeline = sentiment_pipeline
            print("🧠 Sentiment pipeline ricevuta dalla cache")
        else:
            self._inizializza_sentiment_analyzer()
    
        self._prepara_dati_testo()

    def _inizializza_sentiment_analyzer(self):
        """
        Inizializza la pipeline di sentiment usando
        tabularisai/multilingual-sentiment-analysis (23 lingue incl. IT/EN).
        """
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model="tabularisai/multilingual-sentiment-analysis",
                device=device,
                truncation=True,
                max_length=512
            )
            print(f"🧠 Sentiment analyzer (multilingual-sentiment-analysis) inizializzato "
                  f"(device={'GPU' if device == 0 else 'CPU'})")
        except Exception as e:
            print(f"⚠️ Impossibile inizializzare il modello di sentiment: {e}")
            self.sentiment_pipeline = None

    def _prepara_dati_testo(self):
        """Prepara i dati delle domande aperte"""
        print("📝 Preparazione dati testo...")
        self.df_testo = {}
        for col in self.domande_aperte.keys():
            if col in self.df.columns:
                # Converti in stringa e rimuovi NaN
                testi = self.df[col].dropna().astype(str)
                # Filtra risposte vuote o troppo corte
                testi = testi[testi.str.len() > 2]
                self.df_testo[col] = testi.tolist()
                print(f"  ✓ Domanda {col}: {len(testi)} risposte valide")

    def pulisci_testo(self, testo):
   
        if not isinstance(testo, str):
            return []

        # Lowercase
        testo = testo.lower()
        # Rimuovi punteggiatura e numeri
        testo = re.sub(r'[^\w\s]', '', testo)
        testo = re.sub(r'\d+', '', testo)
        # Tokenizzazione
        try:
            tokens = word_tokenize(testo, language=self.lang)
        except:
            tokens = testo.split()

        # Stopwords estese per lingua
        if self.lingua == 'it':
            stopwords_extra = {
                # Articoli
                'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
                # Preposizioni
                'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
                'del', 'dello', 'della', 'dei', 'degli', 'delle',
                'al', 'allo', 'alla', 'ai', 'agli', 'alle',
                'dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle',
                'nel', 'nello', 'nella', 'nei', 'negli', 'nelle',
                'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle',
                # Congiunzioni
                'e', 'ed', 'o', 'od', 'ma', 'però', 'anche', 'oppure', 'né',
                'perché', 'poiché', 'siccome', 'quindi', 'dunque', 'allora',
                # Verbi ausiliari e comuni
                'essere', 'sono', 'sei', 'è', 'siamo', 'siete',
                'avere', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
                'fare', 'fatto', 'fa', 'fanno', 'facendo',
                'potere', 'posso', 'puoi', 'può', 'possiamo', 'potete', 'possono',
                'dovere', 'devo', 'devi', 'deve', 'dobbiamo', 'dovete', 'devono',
                'volere', 'voglio', 'vuoi', 'vuole', 'vogliamo', 'volete', 'vogliono',
                'sarebbe', 'sarebbero', 'sarei', 'saresti',
                'stato', 'stata', 'stati', 'state',
                # Avverbi comuni
                'molto', 'poco', 'tanto', 'troppo', 'più', 'meno', 'già', 'ancora',
                'sempre', 'mai', 'qui', 'qua', 'lì', 'là', 'così', 'come',
                # Pronomi
                'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro',
                'mi', 'ti', 'si', 'ci', 'vi', 'lo', 'la', 'li', 'le',
                'che', 'chi', 'cui', 'quale', 'quali',
            }
        else:
            stopwords_extra = {
                # Articles
                'a', 'an', 'the',
                # Prepositions
                'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
                'into', 'onto', 'upon', 'about', 'between', 'through',
                # Conjunctions
                'and', 'or', 'but', 'nor', 'so', 'yet',
                'because', 'since', 'although', 'though', 'while',
                # Auxiliary verbs
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having',
                'do', 'does', 'did', 'doing', 'done',
                'will', 'would', 'shall', 'should', 'can', 'could',
                'may', 'might', 'must',
                # Pronouns
                'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'its', 'our', 'their',
                'this', 'that', 'these', 'those',
                'who', 'whom', 'which', 'what', 'whose',
                # Adverbs
                'very', 'too', 'also', 'just', 'only', 'even', 'still',
                'already', 'yet', 'always', 'never', 'here', 'there',
            }

        # Combina stopwords NLTK + extra
        stopwords_totali = self.stopwords_set | stopwords_extra

        # Rimuovi stopwords e parole corte
        tokens = [t for t in tokens if t not in stopwords_totali and len(t) > 2]

        # NIENTE STEMMING: le parole restano leggibili
        return tokens

    def parole_piu_frequenti(self, colonna, n=20):
        """
        Restituisce le n parole più frequenti per una domanda aperta

        Args:
            colonna: Codice della domanda (es. 'Q9')
            n: Numero di parole da restituire

        Returns:
            Lista di tuple (parola, frequenza)
        """
        if colonna not in self.df_testo or not self.df_testo[colonna]:
            return []

        all_tokens = []
        for testo in self.df_testo[colonna]:
            tokens = self.pulisci_testo(testo)
            all_tokens.extend(tokens)

        counter = Counter(all_tokens)
        return counter.most_common(n)

    def crea_wordcloud(self, colonna, salva=True, output_dir='output/grafici'):
        """
        Crea una nuvola di parole per una domanda aperta

        Args:
            colonna: Codice della domanda (es. 'Q9')
            salva: Se salvare il file
            output_dir: Directory di output

        Returns:
            Figura matplotlib o None
        """
        if colonna not in self.df_testo or not self.df_testo[colonna]:
            print(f"  ⚠️ Nessuna risposta valida per {colonna}")
            return None

        # Unisci tutti i testi
        testi_concatenati = ' '.join(self.df_testo[colonna])

        # Pulisci il testo
        tokens = self.pulisci_testo(testi_concatenati)
        testo_pulito = ' '.join(tokens)

        if not testo_pulito:
            return None

        # Crea wordcloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(testo_pulito)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        titolo = self.domande_aperte.get(colonna, colonna)
        ax.set_title(f'Nuvola di parole - {titolo}', fontsize=14)
        plt.tight_layout()

        if salva:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{output_dir}/wordcloud_{colonna}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Wordcloud creata per {colonna}")
            return None
        else:
            return fig

    def _classifica_sentiment(self, testo):
    
        if self.sentiment_pipeline is None:
            return 'neutro'

        # --- GUARDRAIL: Rileva "nessun problema/limite" come neutro ---
        testo_lower = testo.lower()
        
        # Pattern comuni che indicano assenza di problemi (sentiment neutro/positivo)
        pattern_negazione_positiva = [
            r"\bno\s+(?:major\s+)?(?:limitation|issue|problem|concern|complaint|drawback)",
            r"\bnothing\s+(?:to\s+)?(?:improve|fix|complain\s+about)",
            r"\bnot\s+(?:many\s+)?(?:limitations|issues|problems)",
            r"\bno\s+need\s+for\s+improvement",
            r"\bnon\s+(?:ho|riscontro)\s+(?:problemi|limiti|criticità)",  # Italiano
            r"\bnessun\s+(?:problema|limite|aspetto\s+da\s+migliorare)", # Italiano
        ]
    
        import re
        for pattern in pattern_negazione_positiva:
            if re.search(pattern, testo_lower):
                # Se troviamo uno di questi pattern, forziamo il risultato
                # Puoi decidere se forzare a 'neutro' o 'positivo'. 
                # 'Neutro' è più sicuro e conservativo.
                return 'neutro'
        
        # --- FINE GUARDRAIL ---

        try:
            # Tronca a 512 caratteri per sicurezza (il modello gestisce 512 token)
            risultato = self.sentiment_pipeline(testo[:512])[0]
            label = str(risultato['label']).strip().lower()

            if 'very positive' in label or label == 'positive':
                return 'positivo'
            elif 'very negative' in label or label == 'negative':
                return 'negativo'
            else:
                return 'neutro'
        except Exception as e:
            print(f"  ⚠️ Errore sentiment: {type(e).__name__}: {e}")
            return 'neutro'

    def analisi_sentiment(self, colonna):
        """
        Analisi del sentiment per una domanda aperta.

        Args:
            colonna: Codice della domanda

        Returns:
            Counter con conteggi di positivo/negativo/neutro
        """
        if colonna not in self.df_testo:
            return None
        if self.sentiment_pipeline is None:
            print("  ⚠️ Sentiment analyzer non disponibile")
            return None

        sentimenti = [self._classifica_sentiment(t) for t in self.df_testo[colonna]]
        return Counter(sentimenti)
    
    def risposte_per_sentiment(self, colonna, sentiment='negativo'):
    
        if not hasattr(self, 'df_testo') or colonna not in self.df_testo:
            return []

        risposte = self.df_testo[colonna]
        negative = []

        for r in risposte:
            try:
                s = self._classifica_sentiment(str(r))  # metodo interno esistente
            except Exception:
                s = 'neutro'
            if s == sentiment:
                negative.append(r)

        return negative

    def risposte_con_sentiment(self, colonna):
        """
        Restituisce una lista di tuple (testo, sentiment) per una domanda aperta.
        Usato per evidenziare le risposte in base al sentiment.

        Args:
            colonna: Codice della domanda

        Returns:
            Lista di tuple (testo: str, sentiment: str)
            sentiment ∈ {'positivo', 'negativo', 'neutro'}
        """
        if colonna not in self.df_testo:
            return []
        if self.sentiment_pipeline is None:
            # Senza analyzer, restituisce tutto come neutro
            return [(t, 'neutro') for t in self.df_testo[colonna]]

        return [(t, self._classifica_sentiment(t)) for t in self.df_testo[colonna]]

    def get_risposte(self, colonna, max_mostra=10):
        """
        Restituisce le risposte per una domanda aperta

        Args:
            colonna: Codice della domanda
            max_mostra: Numero massimo di risposte da mostrare

        Returns:
            Lista di risposte
        """
        if colonna not in self.df_testo:
            return []
        return self.df_testo[colonna][:max_mostra]

    def riepilogo_domande_aperte(self):
        """
        Restituisce un riepilogo di tutte le domande aperte

        Returns:
            DataFrame con riepilogo
        """
        riepilogo = []
        for col, testi in self.df_testo.items():
            if testi:
                # Calcola lunghezza media
                lunghezze = [len(t.split()) for t in testi]
                riepilogo.append({
                    'Domanda': self.domande_aperte.get(col, col),
                    'Codice': col,
                    'N. risposte': len(testi),
                    'Lunghezza media': round(sum(lunghezze) / len(lunghezze), 1),
                    'Lunghezza max': max(lunghezze),
                })
        return pd.DataFrame(riepilogo)

    def cambia_lingua(self, lingua):
        """
        Cambia la lingua dell'analisi del testo.
        Il modello multilingue non richiede re-inizializzazione,
        ma aggiorniamo stopwords e stemmer per wordcloud/frequenze.

        Args:
            lingua: 'it' o 'en'
        """
        self.lingua = lingua
        self.lang = 'italian' if lingua == 'it' else 'english'
        self.stopwords_set = set(stopwords.words(self.lang))
        self.stemmer = SnowballStemmer(self.lang)
        print(f"🔄 Lingua cambiata a: {self.lang} (il modello sentiment è multilingue)")