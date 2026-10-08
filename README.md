# Analisi dati del questionario per l'applicazione MANILA
 
Dashboard interattiva (Streamlit) e script Python per analizzare le risposte al questionario di valutazione di
**MANILA**, l'applicazione low-code per il benchmarking della fairness nel machine learning.
 
Il progetto legge le risposte esportate da Google Moduli e produce:
 
- statistiche e grafici sulle domande a scala Likert (accordo e utilità);
- analisi delle domande aperte (parole frequenti, nuvola di parole, sentiment);
- un'**analisi tematica assistita da due LLM indipendenti**, con accordo tra i due codificatori e tabella di priorità
  dei problemi;
- un **report HTML** pensato per lo sviluppatore, con punti di forza, criticità e soluzioni proposte.
Tutta l'interfaccia è disponibile in italiano e in inglese.
 
---
 
## Struttura del progetto
 
```
.
├── app.py                  # Dashboard Streamlit: punto di ingresso del programma
├── config.py               # Testi delle domande, scale, sezioni e traduzioni (IT/EN)
├── mapping_config.py       # Mappatura delle colonne di Google Moduli sui codici Q1…Q36
├── requirements.txt        # Dipendenze 
├── .gitignore              
├── .vscode/
│   └── settings.json       # Impostazioni dell'editor
└── Utils/                  # Moduli di analisi usati da app.py
    ├── _init_.py
    ├── analisi_profilo.py
    ├── analisi_likert.py
    ├── analisi_testo.py
    ├── analisi_tematica_llm.py
    ├── llm_clients.py
    ├── llm_async.py
    ├── async_helpers.py
    ├── grafici.py
    ├── tabelle.py
    └── report_generator.py
```
 
### File nella cartella principale
 
| File | A cosa serve |
|---|---|
| `app.py` | Dashboard interattiva: carica il file, esegue le analisi, mostra i risultati in schede e genera il report. Dettagli nella sezione successiva. |
| `config.py` | Contiene i testi delle domande Likert, di utilità, di profilo e aperte (in inglese e in italiano), le etichette delle scale a 5 punti, la divisione in sezioni e le scale (`accordo` e `utilita`) a cui appartiene ogni domanda. La funzione `get_config(lingua)` restituisce la configurazione nella lingua scelta. |
| `mapping_config.py` | Dizionario `MAPPING_GOOGLE_FORMS` che associa il testo di ogni colonna del file di Google Moduli al codice della domanda (`Q1`, `Q2`, …, `Q36`). Contiene anche le scale usate per le domande di profilo Q3 e Q4. |
| `requirements.txt` | Elenco delle librerie da installare. |
 
### Cartella `Utils/`
 
| Modulo | Contenuto |
|---|---|
| `analisi_profilo.py` | Classe `AnalisiProfilo`: frequenze e statistiche delle domande di profilo (Q1 consenso, Q2 posizione, Q3 esperienza con strumenti di benchmarking, Q4 conoscenza di fairness) e relativi grafici. |
| `analisi_likert.py` | Classe `AnalisiLikert`: statistiche descrittive (media, mediana, deviazione standard, ecc.) per ogni domanda e **alfa di Cronbach** per ciascuna scala. |
| `analisi_testo.py` | Classe `AnalisiTesto`: pulizia del testo, parole più frequenti, nuvola di parole e **analisi del sentiment** delle domande aperte con un modello multilingue (`tabularisai/multilingual-sentiment-analysis`, eseguito con la libreria Transformers). |
| `analisi_tematica_llm.py` | Classe `AnalisiTematicaLLM`: analisi tematica con due LLM (coder A e coder B). Contiene il **vocabolario di 20 temi** (13 problemi, 6 punti di forza, 1 tema neutro) con definizioni, euristica di usabilità collegata, gravità e soluzione proposta. Calcola l'accordo tra i due codificatori (percentuale e κ di Cohen), risolve i disaccordi unendo i temi assegnati e calcola il punteggio di priorità dei problemi. |
| `llm_clients.py` | Client sincroni per i modelli linguistici. Provider supportati: Google Gemini, Mistral, OpenAI, Anthropic e Ollama (locale). Contiene anche `PROVIDER_DISPONIBILI`, l'elenco di provider e modelli selezionabili dalla dashboard. |
| `llm_async.py` | Versione asincrona dei client, usata per far lavorare i due modelli in parallelo. |
| `async_helpers.py` | Funzione `esegui_coroutine`, che permette di eseguire codice asincrono dentro Streamlit, dove esiste già un ciclo di eventi attivo. |
| `grafici.py` | Classe `GeneratoreGrafici`: grafici delle risposte Likert, medie per sezione e matrice di correlazione. |
| `tabelle.py` | Classe `GeneratoreTabelle`: tabelle di statistiche, distribuzioni, alfa di Cronbach, correlazioni e riepilogo delle risposte aperte, con esportazione in CSV. |
| `report_generator.py` | Classe `ReportGenerator`: costruisce il report HTML autonomo (stili incorporati, bilingue). Include anche `genera_pdf`, che richiede la libreria WeasyPrint. |
| `_init_.py` | Elenca le classi principali del pacchetto . |
 
---
 
## Cosa compare in `app.py`
 
`app.py` è il programma principale: si avvia con `streamlit run app.py`. Il file si può leggere dall'alto verso il basso
in queste parti.
 
### 1. Importazioni e impostazioni iniziali
- Importa Streamlit, pandas, matplotlib, le classi di `Utils/`, `get_config` da `config.py` e la mappatura da `mapping_config.py`.
- Carica le variabili d'ambiente da un file `.env` (con `python-dotenv`), da cui leggere le chiavi API dei modelli linguistici.
### 2. Funzioni di supporto
| Funzione | Cosa fa |
|---|---|
| `carica_sentiment_pipeline()` | Carica una sola volta (con la cache di Streamlit) il modello di sentiment, usando la GPU se disponibile. |
| `load_css()` | Applica lo stile grafico della dashboard. |
| `crea_metrica_card(...)` | Disegna le schede con i numeri riassuntivi in alto. |
| `crea_container_grafico(...)` / `chiudi_container_grafico()` | Racchiudono ogni grafico in un riquadro. |
| `mostra_risposte_aperte(...)` | Mostra le risposte aperte con il sentiment evidenziato. |
| `_normalizza_header(...)` / `applica_mappatura_flessibile(...)` | Rinominano le colonne del file caricato nei codici Q1…Q36 in modo flessibile, tollerando piccole differenze nel testo delle intestazioni. |
| `ricrea_analisi(...)` | Costruisce gli oggetti `AnalisiLikert`, `AnalisiTesto` e `AnalisiProfilo` a partire dai dati e dalla lingua scelta. |
 
### 3. Pagina e stato della sessione
`st.set_page_config` imposta la pagina, mentre `st.session_state` conserva tra un'interazione e l'altra la lingua,
i dati caricati, l'analisi tematica e le tabelle di priorità.
 
### 4. Barra laterale (sidebar)
- **Lingua:** scelta tra italiano e inglese.
- **Caricamento dati:** pulsante per caricare un file CSV o Excel (`.csv`, `.xlsx`, `.xls`).
- **Report:** pulsante "Genera Report HTML", disattivato finché non è stata eseguita l'analisi tematica, e collegamento per scaricare il report generato.
### 5. Corpo della pagina
Dopo il caricamento del file compaiono quattro schede con i numeri riassuntivi (totale risposte, numero di domande Likert,
numero di domande aperte, percentuale di completamento) e sei schede (tab):
 
| Scheda | Contenuto |
|---|---|
| **Profilo** | Frequenze e grafici delle domande Q1–Q4: consenso, posizione, esperienza con strumenti di benchmarking e conoscenza di fairness. |
| **Panoramica** | Grafico delle medie delle valutazioni per sezione del questionario. |
| **Analisi Likert** | Selezione della scala (accordo o utilità), statistiche descrittive scaricabili in CSV, distribuzione delle risposte, alfa di Cronbach e matrice di correlazione. |
| **Analisi testo** | Selezione di una domanda aperta, nuvola di parole, parole più frequenti, sentiment, risposte evidenziate e una tabella di priorità basata su parole chiave (scaricabile in CSV). |
| **Analisi tematica** | Scelta di provider e modello per i due codificatori, avvio della codifica (a gruppi di 15 risposte), accordo tra i codificatori, temi rilevati, risposte raggruppate per tema, tabella di priorità e scaricamento dei risultati in JSON. |
| **Dati grezzi** | Tabella completa dei dati caricati, con scaricamento in CSV. |
 
### 6. Generazione del report
Il pulsante nella sidebar crea un oggetto `ReportGenerator` con tutte le analisi, produce il report HTML e lo rende scaricabile.
 
