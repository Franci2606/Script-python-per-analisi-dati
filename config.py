# config.py - Configurazione con scale multiple

# ============================================================================
# 1. DOMANDE A SCALA LIKERT (in inglese - originali)
# ============================================================================
DOMANDE_LIKERT_EN = {
    'Q5': "5. The application offers a sufficiently broad range of ML models for my benchmarking needs",
    'Q6': "6. The included fairness methods cover a wide range of bias mitigation techniques",
    'Q7': "7. The available fairness metrics are adequate to assess fairness in my use cases",
    'Q8': "8. The effectiveness metrics (accuracy, precision, recall, etc.) cover all the performance measures I need",
    'Q11': "11. The benchmarking results are plausible and consistent with the expected values",
    'Q12': "12. The identification of the best configuration is done clearly and correctly",
    'Q13': "13. The final results are presented through graphical representations which facilitate their interpretation",
    'Q15': "15. The interface guides you step by step through all the configuration phases in a clear way",
    'Q16': "16. It is possible to upload custom datasets and the procedure is comprehensive and well documented",
    'Q17': "17. The system correctly enforces compatibility constraints between components",
    'Q18': "18. The Python code generated includes all necessary dependencies and reproduces the experiment",
    'Q20': "20. The information needed to perform benchmarking tasks are always visible and easy to find",
    'Q21': "21. The information and controls in the app are easy to understand and use",
    'Q22': "22. The interface allows you to immediately select the actions needed to achieve the goal",
    'Q23': "23. The application clearly communicates the status of operations",
    'Q24': "24. Using MANILA is quick and easy to learn",
    'Q25': "25. Help features are easy to find and related to the actions I'm performing",
    'Q26': "26. The application prevents errors and clearly reports them with instructions",
    'Q34': "34. Would you recommend MANILA to other Machine Learning researchers or professionals?",
}

DOMANDE_UTILITA_EN = {
    'Q29': "29. Fairness metrics recommendation system",
    'Q30': "30. Ability to export and import experiment configurations",
    'Q31': "31. A repository of known fairness experiment results",
    'Q32': "32. Automatic generation of a report describing experiment results via LLM",
}

DOMANDE_LIKERT_IT = {
    'Q5': "5. L'applicazione offre una gamma di modelli di machine learning sufficientemente ampia per le mie esigenze di benchmarking",
    'Q6': "6. I metodi di fairness inclusi coprono un'ampia gamma di tecniche di mitigazione del bias",
    'Q7': "7. Le metriche di fairness disponibili sono adeguate per valutare l'equità nei miei scenari d'uso",
    'Q8': "8. Le metriche di efficacia (accuratezza, precisione, richiamo, ecc.) coprono tutte le misure di performance di cui ho bisogno",
    'Q11': "11. I risultati del benchmarking sono verosimili e corrispondono ai valori attesi",
    'Q12': "12. L'applicazione è in grando di mostare la configurazione migliore in modo chiaro",
    'Q13': "13. I risultati finali vengono presentati tramite rappresentazioni grafiche (tabelle comparative, grafici) che ne facilitano l'interpretazione",
    'Q15': "15. L'interfaccia ti guida passo dopo passo attraverso tutte le fasi di configurazione  (dataset, modelli, metodi di fairness, metriche, trade-off) in modo chiaro e completo",
    'Q16': "16. È possibile caricare dataset personalizzati e la procedura è completa e ben documentata",
    'Q17': "17. Il sistema disabilita correttamente metodi di fairness non compatibili con i modelli di machine learning selezionati",
    'Q18': "18. Il codice Python generato al termine del flusso di lavoro include tutte le dipendenze necessarie e riproduce esattamente l'esperimento configurato",
    'Q20': "20. Le informazioni e i comandi necessari per eseguire il benchmarking sono sempre visibili e facili da trovare nell'interfaccia",
    'Q21': "21. Le informazioni e i comandi presenti nell'app sono facili da comprendere e da utilizzare",
    'Q22': "22. L'interfaccia consente di selezionare immediatamente le azioni necessarie per raggiungere l'obiettivo (es. caricare un dataset, scegliere un modello, avviare un benchmark)",
    'Q23': "23. L'applicazione mi comunica chiaramente lo stato delle operazioni (es.esperimento in corso,completato,fallito), gli effetti delle mie azioni e le informazioni necessarie per valutare le modifiche effettuate",
    'Q24': "24. L'utilizzo di MANILA è facile e rapido da apprendere, anche per chi non ha familiarità con strumenti di benchmarking della fairness",
    'Q25': "25. Le funzionalità di aiuto (guide in linea, documentazione) sono facili da reperire e sono collegate alle azioni che sto svolgendo",
    'Q26': "26. L'applicazione previene gli errori (es. disabilitando opzioni incompatibili). Quando si verifica un errore, viene segnalato chiaramente con indicazioni su come risolverlo",
    'Q34': "34. Consiglieresti MANILA ad altri ricercatori o professionisti del campo del Machine Learning?",
}

DOMANDE_UTILITA_IT = {
    'Q29': "29. Sistema di raccomandazione delle metriche di fairness",
    'Q30': "30. Possibilità di esportare ed importare le configurazioni degli esperimenti ",
    'Q31': "31. Repository di risultati di esperimenti di fairness noti, al fine di migliorare la sostenibilità degli esperimenti",
    'Q32': "32. Generazione automatica tramite LLM di un report che descriva i risultati di un esperimento",
}

# ============================================================================
# 2. DOMANDE DI PROFILO (Q1-Q4) - NON Likert
# ============================================================================
DOMANDE_PROFILO_EN = {
    'Q1': "1. I consent to participate in the questionnaire and to the processing of my data for research purposes",
    'Q2': "2. What is your current position?",
    'Q3': "3. What is your experience with Machine Learning benchmarking tools?",
    'Q4': "4. What is your level of knowledge regarding fairness and bias in Machine Learning?",
}

DOMANDE_PROFILO_IT = {
    'Q1': "1. Acconsento a partecipare al questionario e al trattamento dei miei dati per scopi di ricerca",
    'Q2': "2. Qual è la tua posizione attuale?",
    'Q3': "3. Qual è la tua esperienza con gli strumenti di benchmarking per Machine Learning?",
    'Q4': "4. Qual è il tuo livello di conoscenza riguardo fairness e bias nel Machine Learning?",
}

# ============================================================================
# 3. CONFIGURAZIONE DELLE SCALE LIKERT
# ============================================================================
SCALA_ACCORDO_EN = {1: 'Strongly Disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly Agree'}
SCALA_ACCORDO_IT = {1: 'Fortemente in disaccordo', 2: 'In disaccordo', 3: 'Né d\'accordo né in disaccordo', 4: 'D\'accordo', 5: 'Fortemente d\'accordo'}
SCALA_ACCORDO_ORDER_EN = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
SCALA_ACCORDO_ORDER_IT = ['Fortemente in disaccordo', 'In disaccordo', 'Né d\'accordo né in disaccordo', 'D\'accordo', 'Fortemente d\'accordo']

SCALA_UTILITA_EN = {1: 'Not useful at all', 2: 'Slightly useful', 3: 'Moderately useful', 4: 'Very useful', 5: 'Extremely useful'}
SCALA_UTILITA_IT = {1: 'Per niente utile', 2: 'Poco utile', 3: 'Moderatamente utile', 4: 'Molto utile', 5: 'Estremamente utile'}
SCALA_UTILITA_ORDER_EN = ['Not useful at all', 'Slightly useful', 'Moderately useful', 'Very useful', 'Extremely useful']
SCALA_UTILITA_ORDER_IT = ['Per niente utile', 'Poco utile', 'Moderatamente utile', 'Molto utile', 'Estremamente utile']

# ============================================================================
# 4. MAPPATURA DOMANDE -> SCALA (solo accordo e utilità)
# ============================================================================
MAPPATURA_SCALE = {
    'Q5': 'accordo', 'Q6': 'accordo', 'Q7': 'accordo', 'Q8': 'accordo',
    'Q11': 'accordo', 'Q12': 'accordo', 'Q13': 'accordo',
    'Q15': 'accordo', 'Q16': 'accordo', 'Q17': 'accordo', 'Q18': 'accordo',
    'Q20': 'accordo', 'Q21': 'accordo', 'Q22': 'accordo', 'Q23': 'accordo',
    'Q24': 'accordo', 'Q25': 'accordo', 'Q26': 'accordo', 'Q34': 'accordo',
    'Q29': 'utilita', 'Q30': 'utilita', 'Q31': 'utilita', 'Q32': 'utilita',
}

# ============================================================================
# 5. DOMANDE APERTE
# ============================================================================
DOMANDE_APERTE_EN = {
    
    'Q9': "9. Which machine learning model do you find missing but useful for your work?",
    'Q10': "10. Which fairness method do you find missing but useful for your work?",
    'Q14': "14. Have you encountered any discrepancies between the expected results and those produced by the application? If so, please describe them.",
    'Q19': "19. Which component of the platform would you like to see expanded or improved in terms of completeness?",
    'Q27': "27. Which aspect of MANILA's usability do you find most worthy of improvement?",
    'Q28': "28. Which aspect of MANILA's usability do you find most critical?",
    'Q33': "33. Do you have any other suggestions or comments?",
    'Q35': "35. What is MANILA's main strength?",
    'Q36': "36. What is the main limitation or aspect to improve?",
}

DOMANDE_APERTE_IT = {
    
    'Q9': "9. Quale modello di machine learning trovi assente ma utile per il tuo lavoro?",
    'Q10': "10. Quale metodo di fairness trovi assente ma utile per il tuo lavoro?",
    'Q14': "14. Hai riscontrato discrepanze tra i risultati attesi e quelli prodotti dall'applicazione? Se sì, descrivi",
    'Q19': "19. Quale componente della piattaforma vorresti vedere esteso o migliorato in termini di completezza?",
    'Q27': "27. Quale aspetto dell'usabilità di MANILA ritieni meritevole di miglioramento?",
    'Q28': "28. Quale aspetto dell'usabilità di MANILA ritieni più critico?",
    'Q33': "33. Hai altri suggerimenti o osservazioni?",
    'Q35': "35. Qual è il principale punto di forza?",
    'Q36': "36. Qual è il principale limite o aspetto da migliorare?",
}

# ============================================================================
# 6. OPZIONI SCELTA MULTIPLA (per Q1-Q4)
# ============================================================================
OPZIONI_MULTIPLA = {
    'Q1': ['I agree', 'I do not agree'],
    'Q2': ['B.Sc. Student', 'M.Sc. Student', 'PhD Student', 'Postdoctoral Researcher',
           'Assistant Professor', 'Associate/Full Professor', 'Company employee', 'Altro:'],
    'Q3': [ 'Basic', 'Intermediate level', 'Advanced'],
    'Q4': [ 'Basic', 'Intermediate level', 'Advanced'],
}

# ============================================================================
# 7. ETICHETTE PER Q3 E Q4 (scale ordinali di profilo)
# ============================================================================
LABELS_PROFILO_EN = {
    'Q3': { 2: 'Basic', 3: 'Intermediate level', 4: 'Advanced'},
    'Q4': { 2: 'Basic', 3: 'Intermediate level', 4: 'Advanced' },
}
LABELS_PROFILO_IT = {
    'Q3': { 2: 'Base', 3: 'Livello Intermedio', 4: 'Avanzato'},
    'Q4': {2: 'Base', 3: 'Livello Intermedio', 4: 'Avanzato'},
}

# ============================================================================
# 8. SEZIONI DEL QUESTIONARIO (Likert)
# ============================================================================
SEZIONI_EN = {
    'Completeness of components': ['Q5', 'Q6', 'Q7', 'Q8'],
    'Accuracy of results': ['Q11', 'Q12', 'Q13'],
    'Configuration flow': ['Q15', 'Q16', 'Q17', 'Q18'],
    'Web interface usability': ['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26'],
    'Additional features': ['Q29', 'Q30', 'Q31', 'Q32'],
    'Overall rating': ['Q34'],
}

SEZIONI_IT = {
    'Completezza dei componenti': ['Q5', 'Q6', 'Q7', 'Q8'],
    'Accuratezza dei risultati': ['Q11', 'Q12', 'Q13'],
    'Flusso di configurazione': ['Q15', 'Q16', 'Q17', 'Q18'],
    'Usabilità interfaccia web': ['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26'],
    'Funzionalità aggiuntive': ['Q29', 'Q30', 'Q31', 'Q32'],
    'Valutazione complessiva': ['Q34'],
}

# ============================================================================
# 9. METADATI
# ============================================================================
METADATI = {
    'nome': 'Evaluation of MANILA',
    'nome_it': 'Valutazione di MANILA',
    'istituzione': "University of L'Aquila",
    'istituzione_it': 'Università dell\'Aquila',
    'scala_accordo': '5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree)',
    'scala_accordo_it': 'Scala Likert a 5 punti (1 = Fortemente in disaccordo, 5 = Fortemente d\'accordo)',
    'scala_utilita': '5-point usefulness scale (1 = Not useful at all, 5 = Extremely useful)',
    'scala_utilita_it': 'Scala di utilità a 5 punti (1 = Per niente utile, 5 = Estremamente utile)',
    'totale_domande': 36,
    'domande_likert_accordo': 19,
    'domande_likert_utilita': 4,
    'domande_aperte': 9,
    'domande_profilo': 4,
}

MAPPING_VALORI_PROFILO = {
    'Q3': {
        # Etichette inglesi
        'Basic': 2,
                'Intermediate level': 3,
                'Advanced': 4,
                # Etichette italiane
                
                'Base': 2,
                'Livello Intermedio': 3,
                'Avanzato': 4,
    },
    'Q4': {
        # Etichette inglesi
        
        'Basic': 2,
        'Intermediate level': 3,
        'Advanced': 4,
        # Etichette italiane
        
        'Base': 2,
        'Livello Intermedio': 3,
        'Avanzato': 4,
    },
}

# ============================================================================
# 10. FUNZIONE PER OTTENERE LA CONFIGURAZIONE
# ============================================================================
def get_config(lingua='it'):
    """
    Restituisce la configurazione per la lingua specificata
    lingua: 'en' per inglese, 'it' per italiano
    """
    if lingua == 'en':
        domande_likert = {}
        domande_likert.update(DOMANDE_LIKERT_EN)
        domande_likert.update(DOMANDE_UTILITA_EN)

        meta_en = {k: v for k, v in METADATI.items() if not k.endswith('_it')}

        return {
            'domande_likert': domande_likert,
            'domande_profilo': DOMANDE_PROFILO_EN,
            'domande_aperte': DOMANDE_APERTE_EN,
            'opzioni_multipla': OPZIONI_MULTIPLA,
            'labels_profilo': LABELS_PROFILO_EN,
            'mapping_valori_profilo': MAPPING_VALORI_PROFILO,
            'mappatura_scale': MAPPATURA_SCALE,
            'scale': {
                'accordo': {
                    'labels': SCALA_ACCORDO_EN,
                    'order': SCALA_ACCORDO_ORDER_EN,
                    'title': 'Agreement Scale'
                },
                'utilita': {
                    'labels': SCALA_UTILITA_EN,
                    'order': SCALA_UTILITA_ORDER_EN,
                    'title': 'Usefulness Scale'
                }
            },
            'sezioni': SEZIONI_EN,
            'meta': meta_en,
            'lingua': 'en'
        }
    else:
        domande_likert = {}
        domande_likert.update(DOMANDE_LIKERT_IT)
        domande_likert.update(DOMANDE_UTILITA_IT)

        meta_it = {}
        for k, v in METADATI.items():
            if k.endswith('_it'):
                meta_it[k[:-3]] = v
            else:
                meta_it[k] = v

        return {
            'domande_likert': domande_likert,
            'domande_profilo': DOMANDE_PROFILO_IT,
            'domande_aperte': DOMANDE_APERTE_IT,
            'opzioni_multipla': OPZIONI_MULTIPLA,
            'labels_profilo': LABELS_PROFILO_IT,
            'mapping_valori_profilo': MAPPING_VALORI_PROFILO,
            'mappatura_scale': MAPPATURA_SCALE,
            'scale': {
                'accordo': {
                    'labels': SCALA_ACCORDO_IT,
                    'order': SCALA_ACCORDO_ORDER_IT,
                    'title': 'Scala di Accordo'
                },
                'utilita': {
                    'labels': SCALA_UTILITA_IT,
                    'order': SCALA_UTILITA_ORDER_IT,
                    'title': 'Scala di Utilità'
                }
            },
            'sezioni': SEZIONI_IT,
            'meta': meta_it,
            'lingua': 'it'
        }


# ============================================================================
# 11. FUNZIONE PER STAMPARE RIEPILOGO
# ============================================================================
def stampa_riepilogo(lingua='it'):
    config = get_config(lingua)
    meta = config['meta']
    print("=" * 70)
    print(f"📋 {meta.get('nome', 'Questionario')}")
    print(f"🏛️  {meta.get('istituzione', '')}")
    print("=" * 70)
    print("\n📊 SCALE LIKERT:")
    for nome_scala, dati_scala in config['scale'].items():
        print(f"   {dati_scala['title']}:")
        for k, v in dati_scala['labels'].items():
            print(f"      {k} = {v}")
    print("\n" + "=" * 70)
    print(f"📝 Domande Likert: {len(config['domande_likert'])}")
    print(f"📝 Domande profilo (Q1-Q4): {len(config['domande_profilo'])}")
    print(f"📝 Domande aperte: {len(config['domande_aperte'])}")
    print("=" * 70)


if __name__ == "__main__":
    print("\n🇬🇧 CONFIGURAZIONE INGLESE")
    stampa_riepilogo('en')
    print("\n" + "=" * 70)
    print("\n🇮🇹 CONFIGURAZIONE ITALIANA")
    stampa_riepilogo('it')