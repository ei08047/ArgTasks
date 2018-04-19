import json
import os
import re
import numpy as np
from afinn import Afinn
from nltk.corpus import PlaintextCorpusReader as Reader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from utils.Parameters import Parameters
from utils.Document import Document


# fact  vs  feel style argumentation types
class FactFeel:
    corpora_path = '~/nltk_data/corpora'
    nlds_path = '/nlds/data-fact-feel-v2/'
    train_path = 'train'
    dev_path = "dev"
    test_path = "test"
    FACT = 'fact'
    FEEL = 'feel'
    file_path_expression = re.compile('{}|{}|{}'.format(train_path,dev_path,test_path))
    file_expression = re.compile( '.*(fact|feel)_.*' )
    feel_expression = re.compile('.*feel_.*')
    fact_expression = re.compile('.*fact_.*')
    labels = ['fact','feel']

    def __init__(self):
        parameters = Parameters()
        outputFolder = parameters.paths['FACT_FEEL_db']
        print('{}'.format('creating fact feel\n'))
        self.afinn = Afinn()
        data_path = os.path.expanduser(FactFeel.corpora_path)
        self.path = str(data_path + FactFeel.nlds_path)
        self.facts = []
        self.feels = []
        self.doclist = {FactFeel.FACT : [] , FactFeel.FEEL : []}
        self.doc_list_path = str(self.path + 'doclist.json')
        if self.doc_list_exists():
            self.doclist = self.parse_doc_list()
        else:
            self.generate_doc_list()
        list_of_documents = []

        for doc in self.doclist:
            pat = re.compile('(train|dev|test).(fact|feel).(fact|feel)(_\d+)(.txt)')
            for other in self.doclist[doc]:
                l = re.findall(pat,other)
                capturegroup = l.pop()
                parent = str(capturegroup[0:1][0])+ '/' + str(capturegroup[1:2][0])
                docleaf = str(capturegroup[2:3][0]) + str(capturegroup[3:4][0])
                extension = capturegroup[4:5]
                new_doc = Document(parent,docleaf,outputFolder)
                list_of_documents.append(new_doc)

        ## this line prepares a doclist to be used for our flask server that feeds the annotation viewer
        self.trasformed_doclist = self.transform_doclist()

        remakeDB = False
        if remakeDB:
            self.docs = self.create_db(self.path,list_of_documents)
        else:
            temp_docs = []
            for doc in list_of_documents:
                temp = self.read_db(doc)
                temp_docs.append(temp)
            self.docs = temp_docs
        self.facts = [fact for fact in self.docs if 'fact' in fact.get_doc_leaf()]
        self.feels = [feel for feel in self.docs if 'feel' in feel.get_doc_leaf()]

    '''
    score with several -Affin, Vader, Mpqa, Ukp_verb -METHODS SCORE ->  DOC and SENTENCES and WORDS
    and store it in a JSON 
    '''
    def create_db(self,corpora_path, list_of_documents):
        for doc in list_of_documents:
            path = corpora_path + doc.get_path() + '.txt'
            with open(path,'r', encoding="utf-8") as file:
                lines = file.readlines()
                raw=''
                for line in lines:
                    raw += line
                sent_list = sent_tokenize(raw)
                sid = SentimentIntensityAnalyzer()
                id = 0
                for sent in sent_list:
                    tokens = word_tokenize(sent)
                    sent_start = id
                    sent_end = id + len(sent) + 1
                    for word in tokens:
                        doc.add_text_node(id, word)
                        next_id = id + len(word)
                        vader = sid.polarity_scores(word)
                        doc.add_annotation_node({'StartNode':id,'EndNode':next_id, 'Type':'Afinn-word','score':self.afinn.score(word)})
                        doc.add_annotation_node({'StartNode': id, 'EndNode': next_id, 'Type': 'Vader-word','score': vader['compound']})
                        id = next_id + 1
                    ss = sid.polarity_scores(sent)
                    ss2 = self.afinn.score(sent)
                    doc.add_annotation_node({'StartNode': sent_start, 'EndNode': sent_end, 'Type': 'Afinn-sentence','score': ss2})
                    doc.add_annotation_node({'StartNode': sent_start, 'EndNode': sent_end, 'Type': 'Vader-sentence', 'score': ss['compound']})

        for doc in list_of_documents:
            doc.write_json()

        return list_of_documents

    def read_db(self,doc):
        data = Document(doc.parent, doc.get_doc_leaf(),doc.output)
        return data

    def doc_list_exists(self):
        #print('doc_list_exists at {} ? {}'.format(self.doc_list_path, os.path.exists(self.doc_list_path)))
        return os.path.exists(self.doc_list_path)

    def generate_doc_list(self):
        print('generate_doc_list!')
        all = Reader(self.path, '.*\.txt')
        for id in all.fileids():
            if re.match(FactFeel.file_path_expression,id):
                if re.match(FactFeel.file_expression, id):
                    if re.match(FactFeel.fact_expression,id):
                        #print('fact :{}'.format(id))
                        self.facts.append(id)
                    elif re.match(FactFeel.feel_expression,id):
                        #print('feel :{}'.format(id))
                        self.feels.append(id)
        ('fs:{}'.format(all.fileids()))
        print('feels{}'.format(self.feels))
        print('facts{}'.format(self.facts))
        self.doclist[FactFeel.FACT] = self.facts
        self.doclist[FactFeel.FEEL] = self.feels
        with open(self.doc_list_path, 'w+') as outfile:
            json.dump(self.doclist,outfile)

    def parse_doc_list(self):
        with open(self.doc_list_path, 'r') as f:
            json_data = f.read()
        data = json.loads(json_data)
        return data

    def transform_doclist(self):
        pattern = re.compile('(dev|train|test)\/(fact|feel)\/(fact|feel)(_\d+).txt')
        return_list = []
        for feel in self.doclist['feel']:
            all = re.findall(pattern,feel)
            for a in all:
                name = a[2] + a[3]
                return_list.append(name)
        for fact in self.doclist['fact']:
            all = re.findall(pattern,fact)
            for a in all:
                name = a[2] + a[3]
                return_list.append(name)
        return return_list#json.dumps(return_list)


    #_type = Afinn_word, Afinn_sentence, Vader_word, Vader_sentence
    def analyse_doc(self,docs, _type):
        num_docs = len(docs)
        _min = 0
        _max = 0
        _median = 0
        _mean = 0
        all_scores = []
        word_doc = [doc.text_nodes for doc in docs]
        print('num docs: {} \n '.format(len(word_doc)))
        score_word_doc = [ann.get_annotation_nodes_by_type(_type) for ann in docs]
        for doc in score_word_doc:
            for score in doc:
                all_scores.append(score)
        all_scores_without_zeros = [score for score in all_scores if score != 0]
        only_zeros = [score for score in all_scores if score == 0]
        ratio = len(all_scores_without_zeros)/len(only_zeros)
        print('all_scores_without_zeros:{} \t only_zeros:{} \t ratio:{}'.format(len(all_scores_without_zeros),len(only_zeros),ratio))
        _min = min(all_scores)
        _max = max(all_scores)
        _median = np.median(all_scores)
        _mean = np.mean(all_scores)
        print('min:{} \t max:{} \t median:{} \t mean:{}'.format(_min,_max,_median,_mean))
        only_pos_scores = [score for score in all_scores_without_zeros if score > 0]
        only_neg_scores = [score for score in all_scores_without_zeros if score < 0]
        _median_only_pos = np.median(only_pos_scores)
        _median_only_neg = np.median(only_neg_scores)
        print('_median_only_pos:{} \t _median_only_neg:{}'.format(_median_only_pos, _median_only_neg))
        _mean_only_pos = np.mean(only_pos_scores)
        _mean_only_neg = np.mean(only_neg_scores)
        print('_mean_only_pos:{} \t _mean_only_neg:{}'.format(_mean_only_pos, _mean_only_neg))
        _mean_no_zeros = np.mean(all_scores_without_zeros)
        print('_mean_no_zeros:{} \t '.format(_mean_no_zeros))

    def run_analysis(self):
        print('num documents: {} | fact:{} | mental {}'.format(len(self.docs), len(self.facts),len(self.feels)) )
        analysis = {}
        for doc in self.docs:
            analysis = doc.analyse_doc('Afinn-word')
            print(analysis,'\n\n')


'''
# Test
f = FactFeel()
f.run_analysis()

'''








