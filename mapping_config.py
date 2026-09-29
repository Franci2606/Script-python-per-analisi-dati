# mapping_config.py
# Mappatura dei nomi delle colonne di Google Forms ai codici Q1, Q2, ...

MAPPING_GOOGLE_FORMS = {
    # Colonne da ignorare (non sono domande del questionario)
    'Informazioni cronologiche': 'Timestamp',
    
    # Q1: Consenso
    'I consent to participate in the questionnaire and to the processing of my data for the research purposes described above.': 'Q1',
    
    # Q2: Posizione attuale (con campo Altro)
    'What is your current position?': 'Q2',
    
    # Q3: Esperienza con ML benchmarking tools (NON è una domanda Likert, è una scala separata)
    'What is your experience with Machine Learning benchmarking tools?': 'Q3',
    
    # Q4: Conoscenza fairness e bias (NON è una domanda Likert)
    'What is your level of knowledge regarding fairness and bias in Machine Learning ?': 'Q4',
    
    # Q5-Q8: Completezza componenti
    'The application offers a sufficiently broad range of machine learning models for my benchmarking needs.': 'Q5',
    'The included fairness methods cover a wide range of bias mitigation techniques': 'Q6',
    'The available fairness metrics are adequate to assess fairness in my use cases.': 'Q7',
    'The effectiveness metrics (accuracy, precision, recall, etc.) cover all the performance measures I need': 'Q8',
    
    # Q9: Modelli ML mancanti
    'Which machine learning model do you find missing but useful for your work?': 'Q9',
    
    # Q10: Metodi fairness mancanti
    'Which fairness method do you find missing but useful for your work?': 'Q10',
    
    # Q11-Q13: Accuratezza risultati
    'The benchmarking results are plausible and consistent with the expected values': 'Q11',
    'The identification of the best configuration is done clearly and correctly': 'Q12',
    'The final results are presented through graphical representations (comparative tables, graphs) which facilitate their interpretation.': 'Q13',
    
    # Q14: Discrepanze
    'Have you encountered any discrepancies between the expected results and those produced by the application? If so, please describe them.': 'Q14',
    
    # Q15-Q18: Completezza configurazione
    'The interface guides you step by step through all the configuration phases (datasets, models, fairness methods, metrics, trade-offs) in a clear and comprehensive way.': 'Q15',
    'It is possible to upload custom datasets and the procedure is comprehensive and well documented.': 'Q16',
    'The system correctly enforces compatibility constraints between components, preventing invalid configurations': 'Q17',
    'The Python code generated at the end of the workflow includes all necessary dependencies and exactly reproduces the configured experiment.': 'Q18',
    
    # Q19: Componenti da migliorare
    'Which component of the platform would you like to see expanded or improved in terms of completeness?': 'Q19',
    
    # Q20-Q26: Usabilità
    'The information and commands needed to perform benchmarking tasks are always visible and easy to find in the interface.': 'Q20',
    'The information and controls in the app are easy to understand and use.': 'Q21',
    'The interface allows you to immediately select the actions needed to achieve the goal (e.g. load a dataset, choose a model, start a benchmark)': 'Q22',
    'The application clearly communicates to me the status of operations (e.g. "benchmark in progress", "completed", "failed"), the effects of my actions and the information needed to evaluate the changes made.': 'Q23',
    'Using MANILA is quick and easy to learn, even for those unfamiliar with fairness benchmarking tools.': 'Q24',
    'Help features (online help, documentation) are easy to find and are related to the actions I\'m performing': 'Q25',
    'The application prevents errors (e.g., by disabling incompatible options). When an error occurs, it is clearly reported with instructions on how to fix it.': 'Q26',
    
    # Q27-Q28: Aspetti usabilità
    'Which aspect of MANILA\'s usability do you find most worthy of improvement?': 'Q27',
    'Which aspect of MANILA\'s usability do you find most critical?': 'Q28',
    
    # Q29-Q32: Funzionalità aggiuntive (scala di utilità)
    'Fairness metrics recommendation system': 'Q29',
    'Ability to export and import experiment configurations': 'Q30',
    'A repository of known fairness experiment results, aimed at improving the sustainability of experiments': 'Q31',
    'Automatic generation of a report describing experiment results via LLM': 'Q32',
    
    # Q33: Suggerimenti
    'Do you have any other suggestions or comments?': 'Q33',
    
    # Q34: Raccomandazione
    'Would you recommend MANILA to other Machine Learning researchers or professionals?': 'Q34',
    
    # Q35-Q36: Punti di forza/limitazioni
    'What is MANILA\'s main strength?': 'Q35',
    'What is the main limitation or aspect to improve?': 'Q36',
}

# Domande che NON sono sulla scala Likert (sono scale separate)
DOMANDE_NON_LIKERT = ['Q3', 'Q4']

# Scala per Q3 (esperienza con ML benchmarking tools)
SCALA_ESPERIENZA = {
    
    2: 'Basic',
    3: 'Intermediate level',
    4: 'Advanced'
    
}

# Scala per Q4 (conoscenza fairness e bias)
SCALA_CONOSCENZA = {
    2: 'Basic',
        3: 'Intermediate level',
        4: 'Advanced'
}