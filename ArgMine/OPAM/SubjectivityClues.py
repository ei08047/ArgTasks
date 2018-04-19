import os
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

class SubjectivityClues:
    data_path = os.path.expanduser('~/nltk_data/corpora')
    subjectivity_clues_path = '/subjectivity_lexicon/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    clues_pattern = re.compile('type=(weaksubj|strongsubj).len=1.word1=(\w+|\w+-\w*|\w+-\w+-\w*).pos1=(\w+).stemmed1=(y|n).priorpolarity=(\w+)')

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.clues = dict()
        path = str(SubjectivityClues.data_path+SubjectivityClues.subjectivity_clues_path)
        if(os.path.exists(path)):
            with open(path,'r') as clues_file:
                lines = clues_file.readlines()
                for line in lines:
                    res = re.findall(SubjectivityClues.clues_pattern,line)
                    if(res == []):
                        print('error!!')
                    else:
                        (subj_type, word, pos, stemmed, polarity) = res.pop()
                        self.clues[word] = (subj_type,pos,stemmed,polarity)
        else:
            print('cant find')

    def find_word(self, word):
        entry = self.clues[word]
        return entry

    def get_word_subjectivity_type(self,word):
        (sub,_) = self.clues[word]
        return sub

    def get_word_polarity(self, word):
        (_,pol) = self.clues[word]
        return pol

    def analyse_sentence(self,sentence):
        ret = []
        sentence = word_tokenize(sentence)
        for w in sentence:
            try:
                try:
                    entry = self.find_word(w)
                    ret.append( entry )
                except KeyError:
                    st_w = self.stemmer.stem(w)
                    #print('original:  {} stemmed:  {}'.format(w,st_w))
                    entry = self.find_word(st_w)
                    ret.append(entry)
            except KeyError:
                #print('no such word in lexicon', w)
                pass
        formated_ret = []
        for (subj_type,pos,stemmed,polarity) in ret:
            formated_entry = '{}-{}'.format(subj_type,polarity)
            formated_ret.append(formated_entry)
        return formated_ret

    def is_sentence_with_strong_subjectivity(self,sentence):
        sentence = word_tokenize(sentence)
        for w in sentence:
            try:
                sub = self.get_word_subjectivity_type(w)
                if(sub == 'strongsubj'):
                    return True
            except KeyError:
                pass
        return False

    def is_sentence_with_weak_subjectivity(self,sentence):
        sentence = word_tokenize(sentence)
        for w in sentence:
            try:
                sub = self.get_word_subjectivity_type(w)
                if(sub == 'weaksubj'):
                    return True
            except KeyError:
                pass
        return False

    def count_weak(self,sentence):
        count = 0
        sentence = word_tokenize(sentence)
        for w in sentence:
            try:
                sub = self.get_word_subjectivity_type(w)
                if (sub == 'weaksubj'):
                    count += 1
            except KeyError:
                pass
        return count

    def count_strong(self,sentence):
        count = 0
        sentence = word_tokenize(sentence)
        for w in sentence:
            try:
                sub = self.get_word_subjectivity_type(w)
                if (sub == 'strongsubj'):
                    count += 1
            except KeyError:
                pass
        return count

    def subj_pattern(self,sentence):
        pattern = ''
        sentence = word_tokenize(sentence)
        for w in sentence:
            try:
                sub = self.get_word_subjectivity_type(w)
                if (sub == 'strongsubj'):
                    pattern += 'S'
                elif(sub == 'weaksubj'):
                    pattern += 'W'
            except KeyError:
                pass
        return pattern

test = False
if(test):
    s = SubjectivityClues()
    ret = s.analyse_sentence('this would be appallingly apotheosis and abolished')
    print(ret) #  [('would', 'weaksubj'), ('appallingly', 'strongsubj'), ('apotheosis', 'strongsubj')]




