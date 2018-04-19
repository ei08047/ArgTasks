import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.classify import accuracy
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure
import collections
from statistics import mean
import numpy as np
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer


class Experiment:

    def __init__(self, corpora, features_used, items, target, save_features=False, save_mode=False, save_results = False, load_features=False,load_model=False, load_results=False):
        self.corpora = corpora
        self.features_list = features_used
        self.items = items
        self.target = target

        self.save_features = save_features
        self.save_model = save_mode
        self.save_results = save_results
        self.load_features = load_features
        self.load_model = load_model
        self.load_results = load_results

    def load(self, something, what_is):
        print('load ', what_is)
        iris = load_iris()
        return iris

    def save(self, something, what_is):
        print('save ', what_is)

    def train(self,train_features):
        print('training..' , train_features )
        svmc = SklearnClassifier(SVC(), sparse=False).train(train_features)
        return svmc

    def run(self):
        kf = KFold(n_splits=10, shuffle=False, random_state=None)
        #features_set = [(self.corpora.get_sentence_by_id(key).opinion_finder_features(), value) for (key, value) in self.items]
        #features_set = [(self.corpora.get_sentence_by_id(key).arguing_features(), value) for (key, value) in self.items]
        #features_set = [(self.corpora.get_sentence_by_id(key).verb_features(), value) for (key, value) in self.items]
        #features_set = [(self.corpora.get_sentence_by_id(key).strong_subjectivity_feature(), value) for (key, value) in self.items]
        features_set = [(self.corpora.get_sentence_by_id(key).get_all_features(), value) for (key, value) in self.items]
        #print('features_set: ',features_set)
        accuracy_list = []
        arg_precision_list = []
        arg_recall_list = []
        arg_f_measure_list = []
        non_arg_precision_list = []
        non_arg_recall_list = []
        non_arg_f_measure_list = []
        for train_index, test_index in kf.split(features_set):
            #print("TRAIN:", train_index, "TEST:", test_index) #SVC(), sparse=False
            #svm = SVC(kernel='linear',degree = 10 )
            #classifier = SklearnClassifier(svm, sparse=False).train(features_set[train_index[0]:train_index[len(train_index) - 1]])
            #print('coef :',svm.coef_)
            #print('_________________________________________')
            classifier = nltk.NaiveBayesClassifier.train(features_set[train_index[0]:train_index[len(train_index) - 1]])
            #print('most_informative_features: ',classifier.most_informative_features(10))

            #print('training set:', features_set[train_index[0]:train_index[len(train_index) - 1]])

            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)

            for i, (feats, label) in enumerate(features_set[test_index[0]:test_index[len(test_index) - 1]]):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)
            arg_precision = precision(refsets['arg'], testsets['arg'])
            arg_recall = recall(refsets['arg'], testsets['arg'])
            arg_f_measure = f_measure(refsets['arg'], testsets['arg'])
            non_arg_precision = precision(refsets['non-arg'], testsets['non-arg'])
            non_arg_recall = recall(refsets['non-arg'], testsets['non-arg'])
            non_arg_f_measure = f_measure(refsets['non-arg'], testsets['non-arg'])
            accuracy_ =  accuracy(classifier, features_set[test_index[0]:test_index[len(test_index) - 1]])

            accuracy_list.append(accuracy_)
            arg_precision_list.append(arg_precision)
            arg_recall_list.append(arg_recall)
            arg_f_measure_list.append(arg_f_measure)
            non_arg_precision_list.append(non_arg_precision)
            non_arg_recall_list.append(non_arg_recall)
            non_arg_f_measure_list.append(non_arg_f_measure)
        print('median accuracy: ', accuracy_list)
        print('median arg_precision: ', arg_precision_list)
        print('median arg_recall: ', arg_recall_list)
        print('median arg_f_measure: ',arg_f_measure_list)
        print('median non_arg_precision: ', non_arg_precision_list )
        print('median non_arg_recall: ', non_arg_recall_list )
        print('median non_arg_f_measure: ', non_arg_f_measure_list )

    def run_multi_class(self):

        kf = KFold(n_splits=2, shuffle=False, random_state=None)
        features_set = [(self.corpora.get_sentence_by_id(key).get_features_in_list(self.features_list), value) for (key, value) in self.items]
        svm = LinearSVC(random_state=0,class_weight='balanced')
        ovr = OneVsRestClassifier(svm)


        acc_list = list()
        precision = dict()
        recall = dict()
        f1 = dict()

        for train_index, test_index in kf.split(features_set):
            classifier = SklearnClassifier(ovr, sparse=False).train(features_set[train_index[0]:train_index[len(train_index) - 1]])
            accuracy_ = accuracy(classifier, features_set[test_index[0]:test_index[len(test_index) - 1]])
            acc_list.append(accuracy_)
            for t in self.target:
                (target_precision, target_recall, target_f_measure) = self.get_results( classifier,  features_set[test_index[0]:test_index[len(test_index) - 1]], t)
                try:
                    temp = precision[t]
                    temp.append(target_precision)
                    precision[t] = temp
                    temp = recall[t]
                    temp.append(target_recall)
                    recall[t] = temp
                    temp = f1[t]
                    temp.append(target_f_measure)
                    f1[t] = temp
                except KeyError:
                    precision[t] = []
                    temp = precision[t]
                    temp.append(target_precision)
                    precision[t] = temp
                    recall[t] = []
                    temp = recall[t]
                    temp.append(target_recall)
                    recall[t] = temp
                    f1[t] = []
                    temp = f1[t]
                    temp.append(target_f_measure)
                    f1[t] = temp

        line = str(self.features_list) + ' ' + str(mean(acc_list))
        for t in self.target:
            temp = str('{} {} {}'.format(precision[t], recall[t], f1[t]))
            line += ' ' + temp
        line += '\n'
        file = open("results.txt", "a")
        file.write(line)
        file.close()

    def one_vs_other(self):
        kf = KFold(n_splits=2, shuffle=False, random_state=None)
        features_set = [(self.corpora.get_sentence_by_id(key).get_features_in_list(self.features_list), value) for (key, value) in self.items]

        svm = LinearSVC(random_state=0, class_weight='balanced')
        acc_list = list()
        precision = dict()
        recall = dict()
        f1 = dict()
        # learning instance
        for train_index, test_index in kf.split(features_set):
            classifier = SklearnClassifier(svm, sparse=False).train(features_set[train_index[0]:train_index[len(train_index) - 1]])
            accuracy_ = accuracy(classifier, features_set[test_index[0]:test_index[len(test_index) - 1]])
            acc_list.append(accuracy_)
            for t in self.target:
                (target_precision, target_recall, target_f_measure) = self.get_results( classifier,  features_set[test_index[0]:test_index[len(test_index) - 1]], t)
                try:
                    temp = precision[t]
                    temp.append(target_precision)
                    precision[t] = temp
                    temp = recall[t]
                    temp.append(target_recall)
                    recall[t] = temp
                    temp = f1[t]
                    temp.append(target_f_measure)
                    f1[t] = temp
                except KeyError:
                    precision[t] = []
                    temp = precision[t]
                    temp.append(target_precision)
                    precision[t] = temp
                    recall[t] = []
                    temp = recall[t]
                    temp.append(target_recall)
                    recall[t] = temp
                    f1[t] = []
                    temp = f1[t]
                    temp.append(target_f_measure)
                    f1[t] = temp

        line = str(self.features_list) + ' ' + str(mean(acc_list))
        for t in self.target:
            temp = str('{} {} {}'.format(mean(precision[t]), mean(recall[t]), mean(f1[t])))
            line += ' ' + temp
        line += '\n'
        file = open("results.txt", "a")
        file.write(line)
        file.close()

    def test_pipeline(self):
        kf = KFold(n_splits=2, shuffle=False, random_state=None)
        features_set = [(self.corpora.get_sentence_by_id(key).get_features_in_list(self.features_list), value) for
                        (key, value) in self.items]
        svm = LinearSVC(random_state=0, class_weight='balanced')
        acc_list = list()
        precision = dict()
        recall = dict()
        f1 = dict()
        # learning instance
        for train_index, test_index in kf.split(features_set):
            classifier = SklearnClassifier(svm, sparse=False).train(
                features_set[train_index[0]:train_index[len(train_index) - 1]])
            accuracy_ = accuracy(classifier, features_set[test_index[0]:test_index[len(test_index) - 1]])
            acc_list.append(accuracy_)
            for t in self.target:
                print('target {}'.format(t))
                (target_precision, target_recall, target_f_measure) = self.get_results(classifier, features_set[test_index[0]:test_index[len(test_index) - 1]],t)
                try:
                    temp = precision[t]
                    temp.append(target_precision)
                    precision[t] = temp
                    temp = recall[t]
                    temp.append(target_recall)
                    recall[t] = temp
                    temp = f1[t]
                    temp.append(target_f_measure)
                    f1[t] = temp
                except KeyError:
                    precision[t] = []
                    temp = precision[t]
                    temp.append(target_precision)
                    precision[t] = temp
                    recall[t] = []
                    temp = recall[t]
                    temp.append(target_recall)
                    recall[t] = temp
                    f1[t] = []
                    temp = f1[t]
                    temp.append(target_f_measure)
                    f1[t] = temp

        line = str(self.features_list) + ' ' + str(mean(acc_list))
        for t in self.target:
            temp = str('{} {} {}'.format(precision[t], recall[t],f1[t]))
            line += ' ' + temp
        line += '\n'
        file = open("pipeline.txt", "a")
        file.write(line)
        file.close()

    def get_results(self, classifier, test_set, target):
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats, label) in enumerate(test_set):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
        target_precision = precision(refsets[target], testsets[target])
        target_recall = recall(refsets[target], testsets[target])
        target_f_measure = f_measure(refsets[target], testsets[target])
        results = (target_precision, target_recall, target_f_measure)
        return(results)


