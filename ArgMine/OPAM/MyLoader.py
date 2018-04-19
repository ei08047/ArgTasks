from sklearn.datasets.base import Bunch
from sklearn.model_selection import train_test_split

from OPAM.ArgumentativeEssays import ArgumentativeEssays
from asd_en.ml import DatasetLoader

target_dict = { 'Premise':0, 'Claim':1, 'Non-Arg':2 }

class MyLoader(DatasetLoader):

    def addLearningInstancesToDataset(self):
        ae_corpora = ArgumentativeEssays()
        print('1 ArgumentativeEssays')
        ae_corpora.read_doc_list()
        print('2 read_doc_list')
        ae_corpora.read_all_raw()
        print('3 read_all_raw')
        ae_corpora.read_all_arg_annotations()
        print('4 read_all_arg_annotations')
        ae_corpora.read_all_opinion_finder_annotations()
        print('5 read_all_opinion_finder_annotations')
        ae_corpora.produce_sentences_in_essays()
        print('6 produce_sentences_in_essays')
        ae_corpora.produce_all_arguing_lexicon_annotations()
        print('7 produce_all_arguing_lexicon_annotations')
        ae_corpora.produce_all_subjectivity_lexicon_annotations()
        print('8 produce_all_subjectivity_lexicon_annotations')
        ae_corpora.assign_features()
        print('9 assign_features')
        sentences = ae_corpora.get_all_sentences()
        print('10 get_all_sentences')
        print('sentences {}'.format(len(sentences)))
        target = ('Premise', 'Non-Arg')
        labeled_sentences = ae_corpora.get_labeled_sentences_by_target(sentences, target)
        temp_data = []
        temp_target = []
        for (sentence_id,sentence_gold) in labeled_sentences:
            ## get sentence from id
            s = ae_corpora.get_sentence_by_id(sentence_id)
            temp_data.append(s)
            temp_target.append( target_dict[sentence_gold] )
        self.doclist = ae_corpora.doclist
        self.dataset.data = temp_data

        self.dataset.target = temp_target
        self.dataset.target_names= [target[0],target[1]]
        print('data: {}  | target {}'.format(len(self.dataset.data),len(self.dataset.target)))

    def getTainingTestSetSplit(self, trainingSetPercentageSplit= 0.6, randomStateSeed= 12345):
        trainingSet = Bunch()
        testSet = Bunch()
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.data, self.dataset.target, test_size = 1 - trainingSetPercentageSplit, random_state = randomStateSeed)
        trainingSet.data = X_train
        trainingSet.target = y_train
        trainingSet.target_names = (self.dataset).target_names

        testSet.data = X_test
        testSet.target = y_test
        testSet.target_names = (self.dataset).target_names
        return (trainingSet, testSet)