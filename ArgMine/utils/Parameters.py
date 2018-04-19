import os

class Parameters(object):
    
    def __init__(self):
        self.paths= {
                "FACT_FEEL_db": os.path.abspath("../data/features/db"),
                "taggerInput": os.path.abspath("../data/taggerInput"),
                "taggerOutput": os.path.abspath("../data/taggerOutput"),
                "wordEmbeddings": os.path.abspath("../data/embeddings"),
                "ArgMineCorpus": os.path.abspath("../data/ArgMineCorpus"),
                "ArgMineCorpusGoldAnnotations": os.path.abspath("../data/ArgMineCorpus_Gold"),
                "AAECCorpus": os.path.abspath("../data/AAEC"),
                "results": os.path.abspath("../data/results"),
                "datasetPrint": os.path.abspath("../data/results/datasetPrint"),
                "keywords_pt": os.path.abspath("../data/keywords/pt"),
                "keywords_en": os.path.abspath("../data/keywords/en"),
                "textualEntailmentPredictions": os.path.abspath("../data/results/tePredictions"),
                "models": '/models',
                "externalModels": '../models',
                "predictionsOutput": "/predictions",
                "datasetLoader": "/datasetLoader",
                "predictionsWebsite": os.path.abspath("../data/results/predictionsWebsite"),
                "websiteNews": os.path.abspath("../data/results/predictionsWebsite/news"),
                "data": os.path.abspath("../data"),
                "segmentedNews": os.path.abspath("../data/taggerOutput/segmentedNews"),
                "database": os.path.abspath("../data/db"),
                "citiusTagger": os.path.abspath("../externaltools/CitiusPOSTagger"),
                "pathASD": os.path.abspath("../data/results/asd"),
                "pathASD_en": os.path.abspath("../data/results/asd_en"),
                "pathFFD_en": os.path.abspath("../data/results/ffd_en"),
                "pathPBL": os.path.abspath("../data/results/pbl"),
                "pblModel": os.path.abspath("../data/CSTNews/modelIntraPBL"),
                "wordnetPT": os.path.abspath("../data/wordNetPT"),
                "CSTNewsCorpus": os.path.abspath("../data/CSTNewsCorpus"),
                "CSTNewsRSTCorpus": os.path.abspath("../data/CSTNewsCorpus/rst"),
                "CSTNewsRSTCorpusPBLAnnotations": os.path.abspath("../data/CSTNewsCorpus/pblAnnotations2html"),
                "stanfordPoSTaggerJar": "/home/gil/Documents/Programs/stanford-postagger-full-2016-10-31/stanford-postagger.jar",
                #"stanfordPoSTaggerModel": "/home/gil/Documents/Programs/stanford-postagger-full-2016-10-31/models/english-bidirectional-distsim.tagger"
                "stanfordPoSTaggerModel": "/home/gil/Documents/Programs/stanford-postagger-full-2016-10-31/models/english-left3words-distsim.tagger"
                }
        self.filenames= {

                "premiseKeywords": "premiseKeywords.txt",
                "conclusionKeywords": "conclusionKeywords.txt",
                "datasetLoaderPickle": "datasetLoaderFile.pkl",
                "stopWordsWordCouple": "StopWordsWordCouple.txt",
                "stopWords": "stopwords.txt",
                "gridSearchLogs": "GridSearchLogs.txt",
                "gridSearchLogsConsole": "GridSearchLogsAux.txt",
                "modalAuxiliary": "modalAuxiliary.txt",
                "featureAnalysis": "FeatureAnalysis.txt",
                "argmineDatabase": "argmine.db",
                "synsetsWordnetPT": "synsets.txt",
                "relationsWordnetPT": "triplos.txt",
                "textualEntailmentModel": "logreg_temodel.pkl",
                "modelASD": "modelASD.pkl",
                "modelASD_en": "modelASD_en.pkl",
                "modelPBL": "modelPBL.pkl",
                "modelFFD": "modelFFD.pkl",
                "datasetLoaderPickleASD": "datasetLoaderFileASD.pkl",
                "inputFilePoSTagger": "inputFilePoSTagger",
                "wordEmbeddingsModel_pt": "polyglot-pt.pkl",
                "wordEmbeddingsModel_en": "polyglot-en.pkl",
                }