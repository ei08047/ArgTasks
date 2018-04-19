from re import match
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import os.path


class Verb:
    data_path = os.path.expanduser('~/nltk_data/corpora')
    ukp_verbs_path = '/ukp_verbs/'
    lemmatizer = WordNetLemmatizer()
    communication_verbs = 'communication_verbs.txt'
    mental_verbs = 'mental_verbs.txt'

    def __init__(self):
        self.communication = set()
        self.mental = set()
        self.read_files()

    def read_files(self):
        com_path = str(Verb.data_path + Verb.ukp_verbs_path + Verb.communication_verbs)
        men_path = str(Verb.data_path + Verb.ukp_verbs_path + Verb.mental_verbs)

        if(os.path.exists(com_path) and os.path.exists(men_path)):
            with open(com_path,'r') as file:
                lines = file.readlines()
                for word in lines:
                    word = word.replace('\n','')
                    self.communication.add(word)
            with open(men_path,'r') as file:
                lines = file.readlines()
                for word in lines:
                    word = word.replace('\n','')
                    self.mental.add(word)

    def count_mental_verb_in_sentence(self,sentence):
        count = 0
        sentence = word_tokenize(sentence)
        sentence = pos_tag(sentence)
        for w, tag in sentence:
            # print('w {} , tag {}'.format(w,tag))
            if (match('V\w+', tag)):
                w = Verb.lemmatizer.lemmatize(w, 'v')
                for v in self.mental:
                    if (v == w):
                        count += 1
        return count

    def count_communication_verb_in_sentence(self,sentence):
        count = 0
        sentence = word_tokenize(sentence)
        sentence = pos_tag(sentence)
        for w,tag in sentence:
            if(match('V\w+',tag)):
                w = Verb.lemmatizer.lemmatize(w,'v')
                for v in self.communication:
                    if(v == w):
                        count += 1
        return count

    def is_sentence_with_mental_verb(self,sentence):
        sentence = word_tokenize(sentence)
        sentence = pos_tag(sentence)
        for w,tag in sentence:
            #print('w {} , tag {}'.format(w,tag))
            if(match('V\w+',tag)):
                w = Verb.lemmatizer.lemmatize(w,'v')
                for v in self.mental:
                    if(v == w):
                        #print('FOUND! mental verb: v({}) == w({})'.format(v,w))
                        return True
        return False

    def is_sentence_with_communication_verb(self,sentence):
        sentence = word_tokenize(sentence)
        sentence = pos_tag(sentence)
        for w,tag in sentence:
            if(match('V\w+',tag)):
                w = Verb.lemmatizer.lemmatize(w,'v')
                for v in self.communication:
                    if(v == w):
                        #print('FOUND! communication verb: v({}) == w({})'.format(v,w))
                        return True
        return False

