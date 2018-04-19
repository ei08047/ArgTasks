#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import math
import operator
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import scipy.sparse as spp
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict, GridSearchCV, learning_curve, validation_curve
from sklearn.pipeline import Pipeline

from asd_en.DenseTransformer import DenseTransformer
from asd_en.FeatureScaler import FeatureScaler
from asd_en.ml.Configuration import Configuration
from asd_en.ml.DatasetLoader import DatasetLoader
from utils.Parameters import Parameters

# paths
currentDirectory= os.path.abspath(__file__)
parameters= Parameters()
paths= parameters.paths
filenames= parameters.filenames


class MachineLearningTask:
    __metaclass__= ABCMeta
    
    def __init__(self, datasetLoader, configuration, taskName= "Machine Learning Task", outputPath= paths["results"]):
        self.taskName= str(taskName)
        print ("\n\n+++++   " + self.taskName + "   +++++\n\n")
        print ("\n\n++++   " + configuration.type + "   ++++\n\n")
        self.outputPath= outputPath
        self.predictionsOutputPath= self.outputPath + paths["predictionsOutput"]
        self.datasetLoaderPath= self.outputPath + paths["datasetLoader"]
        if not (os.path.exists(self.outputPath)):
            os.makedirs(self.outputPath)
        if not (os.path.exists(self.predictionsOutputPath)):
            os.makedirs(self.predictionsOutputPath)
        if not (os.path.exists(self.datasetLoaderPath)):
            os.makedirs(self.datasetLoaderPath)
        # Loading dataset
        self.datasetLoader = None
        self.getDatasetLoader(datasetLoader)
        # Loading configuration
        self.configuration = None
        self.loadConfigurations(configuration)

        (self.trainingSet, self.testSet) = self.datasetLoader.getTrainingTestSetSplit()
        self.fittedPipeline = None
        self.pipeline = None
        # Update after running "learn()"
        self.predictionsTestSet = None
        self.probaPredictionsTestSet = None

    #self.outputPredictions(outputsPath + paths["models"], savedModelFilename, outputsPath + paths["predictionsOutput"])
    @abstractmethod
    def outputPredictions(self, modelPath, modelFilename):
        """
        """

    def getDatasetLoader(self, datasetLoader):
        # Loading dataset
        self.datasetLoader= None
        if isinstance(datasetLoader, DatasetLoader):
            self.datasetLoader = datasetLoader
            print ("\nSaving DatasetLoader to pickled file ...in directory: "+self.datasetLoaderPath)
            joblib.dump(self.datasetLoader, self.datasetLoaderPath + "/" + filenames["datasetLoaderPickle"], compress= 3)
        elif (datasetLoader is None) and (os.path.isfile(self.datasetLoaderPath + "/" + filenames["datasetLoaderPickle"])):
            print ("\nLoading DatasetLoaderFFD from pickled file ...")
            self.datasetLoader= joblib.load(self.datasetLoaderPath + "/" + filenames["datasetLoaderPickle"])
        else:
            raise Exception("DatasetLoader class not instantiated!")

    def loadConfigurations(self, configuration):
        # Loading configuration
        self.configuration= None
        if isinstance(configuration, Configuration):
            self.configuration= configuration
        else:
            raise Exception("Configuration class not instantiated!")

    def learn(self, crossValidationStrategy = 5, savedModelFilename= "myModel", featureAnalysis= True, performROCAnalysis= True):
        learningProcessStartTime= time.time()
        ########################
        #####   Features   #####
        ########################
        """
        FeatureUnion: composite feature spaces
        
        FeatureUnion combines several transformer objects into a new transformer that combines their output. A FeatureUnion takes 
        a list of transformer objects. During fitting, each of these is fit to the data independently. For transforming data, the 
        transformers are applied in parallel, and the sample vectors they output are concatenated end-to-end into larger vectors.
        
        FeatureUnion serves the same purposes as Pipeline - convenience and joint parameter estimation and validation.
        
        NOTE: A FeatureUnion has no way of checking whether two transformers might produce identical features. It only produces a 
        union when the feature sets are disjoint, and making sure they are is the caller's responsibility.
        
        Usage:
        built using a list of (key, value) pairs, where the key is the name you want to give to a given transformation (an 
        arbitrary string; it only serves as an identifier) and value is an estimator object
        """
        combinedFeatures = (self.configuration).loadFeatureSet()
        # get configurations
        featuresConfig= (self.configuration).loadFeaturesConfigs()
        algorithms= (self.configuration).loadClassifiersConfigs()
        filterMethods= (self.configuration).loadFilterMethodsConfigs()
        ########################
        #####   Pipeline   #####
        ########################
        # Info: http://scikit-learn.org/stable/modules/pipeline.html#combining-estimators
        """
        All estimators in a pipeline, except the last one, must be transformers (i.e. must have a transform method). The last 
        estimator may be any type (transformer, classifier, etc.).
        
        The Pipeline is build using a list of (key, value) pairs, where the key is a string containing the name you want to give 
        this step and value is an estimator object.
        
        The estimators of a pipeline are stored as a list -> see previous link if you want to obtain a specific estimator contained
        in the pipeline
        
        NOTE: Calling fit on the pipeline is the same as calling fit on each estimator in turn, transform the input and pass it 
        on to the next step. The pipeline has all the methods that the last estimator in the pipeline has, i.e. if the last 
        estimator is a classifier, the Pipeline can be used as a classifier. If the last estimator is a transformer, again, so is 
        the pipeline.
        """
        print (" Building Pipeline ...\n")
        self.pipeline= None
        if (self.configuration).fixed:
            # Structure: [name, object, configuration]
            if not (algorithms[0][0] == 'mnb' ):
                self.pipeline = Pipeline([("features", combinedFeatures), ('toDense', DenseTransformer()), ('filter', filterMethods[0][1]), ("clf", algorithms[0][1])])
            else:
                print('FeatureScaler(scalerType=2)')
                #pipeline = Pipeline([("features", combinedFeatures), ('finalFeatureNormalization', FeatureScaler.FeatureScaler(scalerType=0)), ('filter', filterMethods[0][1]), ("clf", algorithms[0][1])])
                self.pipeline = Pipeline([("features", combinedFeatures), ('finalFeatureNormalization', FeatureScaler(scalerType=2)),  ('filter', filterMethods[0][1]), ("clf", algorithms[0][1])])


            # complete configuration
            completeParametersSet= (self.configuration).getCompleteParametersSet(featuresConfig, filterMethods[0][2], algorithms[0][2])
            (self.pipeline).set_params(**completeParametersSet)
        else:
            print('performGridSearch!')
            self.pipeline= self.performGridSearch(combinedFeatures, featuresConfig, filterMethods, algorithms)
        ################################
        #####   Cross Validation   #####
        ################################
        # Info: http://scikit-learn.org/stable/modules/cross_validation.html
        """
        # Estimating the estimator accuracy
        # Input: estimator/pipeline and dataset
        
        print " Performing Cross Validation ...\n"
        
        # When the cross-validation argument is an integer, cross_val_score uses the KFold or StratifiedKFold 
        # strategies by default, the latter being used if the estimator derives from ClassifierMixin.
        scores= cross_validation.cross_val_score(pipeline, np.asarray((self.trainingSet).data), (self.trainingSet).target, cv=5)
        print "scores:"
        print scores
        print ""
        
        # mean score and the 95% confidence interval of the score estimate
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        """
        """
        The function cross_val_predict has a similar interface to cross_val_score, but returns, for each element in the input, 
        the prediction that was obtained for that element when it was in the test set. Only cross-validation strategies that 
        assign all elements to a test set exactly once can be used (otherwise, an exception is raised).
        These prediction can then be used to evaluate the classifier:
        
        NOTE: the result of this computation may be slightly different from those obtained using cross_val_score as the elements 
        are grouped in different ways.
        """
        print (" Obtaining Predictions ...\n")
        cvTimeStart= time.time()
        print('len(self.trainingSet.data): {} || len(self.trainingSet.target): {}'.format(len(self.trainingSet.data),len(self.trainingSet.target)))
        predicted= cross_val_predict(self.pipeline, self.trainingSet.data, y=self.trainingSet.target, cv= crossValidationStrategy, verbose=10)
        elapsedTimeCV = time.time() - cvTimeStart
        print ("\n[Cross-validation performed in " + str(elapsedTimeCV) + " sec.]\n")
        #######################
        #####   Results   #####
        #######################
        self.evaluationMetrics("Cross Validation Metrics - Training data:", (self.trainingSet).target, predicted, (self.trainingSet).target_names, True)
        # Fit Model for predictions
        self.fittedPipeline= (self.pipeline).fit(np.array((self.trainingSet).data), np.array((self.trainingSet).target))
        #print('model info{}',type(self.fittedPipeline))
        #################################
        #####   Model Persistency   #####
        #################################
        # After training the model -> persist the model for future use without having to retrain.
        print ("\n Saving model (pipeline) ...")
        # Saving the pipeline
        if os.path.isfile(self.outputPath + paths["models"] + "/" + savedModelFilename):
            os.remove(self.outputPath + paths["models"] + "/" + savedModelFilename)
        if not (os.path.exists(self.outputPath + paths["models"])):
            os.makedirs(self.outputPath + paths["models"])
        try:
            joblib.dump(self.fittedPipeline, self.outputPath + paths["models"] + "/" + savedModelFilename, compress= 3)
        except:
            print('trouble in joblib.dump')
        ################################
        #####   Feature Analysis   #####
        ################################
        if featureAnalysis:
            self.performFeatureAnalysis(self.fittedPipeline, self.outputPath, filenames["featureAnalysis"])
        ######################################
        #####   Predictions - Test Set   #####
        ######################################
        print ("Test data length: " + str(len((self.testSet).data)))
        self.predictionsTestSet = self.fittedPipeline.predict(np.asarray((self.testSet).data))
        self.probaPredictionsTestSet= self.fittedPipeline.predict_proba(np.asarray((self.testSet).data))
        print('self.predictionsTestSet', self.predictionsTestSet)
        print('self.probaPredictionsTestSet', self.probaPredictionsTestSet)
        self.evaluationMetrics("Evaluation Metrics - Test Set:", (self.testSet).target, self.predictionsTestSet, (self.testSet).target_names)
        ############################
        #####   ROC Analysis   #####
        ############################
        if performROCAnalysis:
            self.performROCAnalysis("ROC Analysis:", (self.testSet).target, self.probaPredictionsTestSet, (self.testSet).target_names)
        elapsedTimeLearningProcess = time.time() - learningProcessStartTime
        print ("\n[Learning Process performed in " + str(elapsedTimeLearningProcess) + " sec.]\n")
        print ("\n\nThe end!")
    #TODO: Check if we should use "sklearn.model_selection.ParameterGrid" and "sklearn.model_selection.ParameterSampler" instead of using  "sklearn.model_selection.GridSearchCV"


    def performGridSearch(self, featureSet, featuresConfig, filtersConfig, classifiersConfig, outputPath= paths["results"]):
        pipeline = None
        print ("Performing Grid Search ...\n")
        currentBestScore= -1.0
        currentBestPipeline= None
        currentBestParametersSet= None
        startTime= time.time()
        gridSearchLogs = open(outputPath + "/" + filenames["gridSearchLogs"], "w")
        sys.stdout= open(outputPath + "/" + filenames["gridSearchLogsConsole"], "w")
        for name, model, tuneParameters in classifiersConfig:
            gridSearchLogs.write("\n> Model: " + str(name)  + "\n")
            startTimeClassifier= time.time()
            currentParametersSet=  featuresConfig.copy()
            for k,v in tuneParameters.items():
                if k not in currentParametersSet:
                    currentParametersSet[k] = v
            for filterName, filterMethod, filterTuneParameters in filtersConfig:
                gridSearchLogs.write("\n    >> Filter: " + str(filterName)  + "\n")
                currentParametersSet2= currentParametersSet.copy()
                for filterKey,filterValue in filterTuneParameters.items():
                    if filterKey not in currentParametersSet2 and (not (name == 'mnb')) :
                        currentParametersSet2[filterKey] = filterValue
                #currentPipeline = Pipeline([("features", combined_features), ('fa', FeatureAnalysis.FeatureAnalysis()), ("clf", model)])
                if not (name == 'mnb' ):
                    print('current pipeline:{}'.format(name))
                    currentPipeline = Pipeline([("features", featureSet), ('finalFeatureNormalization', FeatureScaler(scalerType=0)), ('toDense', DenseTransformer()), ('filter', filterMethod), ("clf", model)])
                else:
                    print('setting currentPipeline\n\n')
                    currentPipeline = Pipeline([("features", featureSet), ('finalFeatureNormalization', FeatureScaler(scalerType=0)), ("clf", model)])
                #currentPipeline.set_params(features__textStatisticsScaled__scaler= currentScaler)
                #gridSearch = RandomizedSearchCV(currentPipeline, param_distributions=currentParametersSet2, n_jobs= -1, verbose=10, cv= 3, scoring= 'f1_weighted', refit= True, error_score=0, n_iter= 40)
                gridSearch = GridSearchCV(currentPipeline, param_grid= currentParametersSet2, n_jobs= 1, verbose=10, cv= 3, scoring= 'f1_weighted', refit= True, error_score=0)
                currentScore= -1.0
                try:
                    gridSearch.fit(np.asarray((self.trainingSet).data), (self.trainingSet).target)
                    currentScore= gridSearch.best_score_
                    gridSearchLogs.write("\n            >>> Best configuration:\n")
                    gridSearchLogs.write("                    score= " + str(currentScore) + "\n")
                    gridSearchLogs.write("                    configuration set:\n")
                    best_parameters = gridSearch.best_estimator_.get_params()
                    for param_name in sorted(currentParametersSet2.keys()):
                        gridSearchLogs.write("                        %s: %r" % (param_name, best_parameters[param_name]))
                        gridSearchLogs.write("\n")
                    if currentScore > currentBestScore:
                        currentBestPipeline = gridSearch.best_estimator_
                        currentBestScore = gridSearch.best_score_
                        currentBestParametersSet= currentParametersSet2
                except Exception as e:
                    print ("\n\nException caught!")
                    print ("Message:")
                    print (e)
            elapsedTimeClassifier= time.time() - startTimeClassifier
            gridSearchLogs.write("\n[Time consumed to perform Grid Search: " + str(elapsedTimeClassifier) + " sec.]\n")
            gridSearchLogs.write("\n\n\n---------------------------------------------------------------------------------------\n\n\n")
        sys.stdout.close()
        sys.stdout= sys.__stdout__
        # update pipeline
        pipeline_is_none = False
        currentBestPipeline_is_none = False
        if pipeline == None:
            pipeline_is_none=True
        if currentBestPipeline == None:
            currentBestPipeline_is_none=True

        print('update pipeline:  pipeline_is_none {} || currentBestPipeline_is_none {}'.format(pipeline_is_none,currentBestPipeline_is_none))
        pipeline = currentBestPipeline

        gridSearchLogs.write("\n\n\n")
        gridSearchLogs.write("*****************************\n")
        gridSearchLogs.write("*****   Best Pipeline   *****\n")
        gridSearchLogs.write("*****************************\n")
        gridSearchLogs.write("score: %0.3f" % currentBestScore)
        gridSearchLogs.write("\n")
        gridSearchLogs.write("configuration set:\n")
        best_parameters = pipeline.get_params()
        
        for param_name in sorted(currentBestParametersSet.keys()):
            gridSearchLogs.write("    %s: %r\n" % (param_name, best_parameters[param_name]))
        
        
        elapsedTime= time.time() - startTime
        gridSearchLogs.write("\n\n[Time consumed to perform complete Grid Search: " + str(elapsedTime) + " sec.]\n")
        
        gridSearchLogs.close()
        
        return pipeline

    def performFeatureAnalysis(self, pipeline, outputFilePath, outputFileName):
        print('performFeatureAnalysis',outputFilePath,outputFileName)

        """
        # Create clone of original Pipeline
        # To avoid override of information and changes to the pipeline that will be used in the training phase
        # E.g. to print the feature set, the pipeline must be fitted with some training data. If not cloned, complications 
        # related to fitting the data twice or similar problems may occur.
        pipelineTest= clone(pipeline)
        
        # Note: after cloning any pipeline, the resulting pipeline is not fitted
        # therefore, we have to fit it again on the (complete) dataset and, then call 'transform' on the test data to obtain 
        # the desired feature space
        pipelineTest = (pipelineTest).fit((self.trainingSet).data, (self.trainingSet).target)
        currentFeatureSpaceBeforeFilter= (pipelineTest.named_steps['features']).transform((self.trainingSet).data)
        
        rfecv = RFECV(estimator=pipelineTest.named_steps['clf'], step=1, cv=5, scoring='accuracy', verbose=2)
        pipelineTest= rfecv.fit(currentFeatureSpaceBeforeFilter, (self.trainingSet).target)
        
        print("Optimal number of features : %d" % rfecv.n_features_)
        
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
        
        print "Features ranking:"
        print rfecv.ranking_
        """
        
        #print "Classes order: " + str(pipeline.named_steps['svm'].classes_)
        
        #print "\n# features:"
        #print (pipeline.named_steps['filter']).shape
        #print "\nsvm coefs shape:"
        #print ((pipeline.named_steps['filter']).coef_).shape
        
        pipelineTest= clone(pipeline)
        # after clone, the resulting pipeline is not fitted
        # therefore, we have to fit it again on the complete dataset and then call 'transform' on the test data to obtain the desired feature space
        pipelineTest = pipelineTest.fit(np.asarray((self.trainingSet).data), (self.trainingSet).target)
        currentFeatureSpace= (pipelineTest.named_steps['features']).transform(np.asarray((self.testSet).data))

        #currentFeatureSpaceAfterFilter= (pipelineTest.named_steps['finalFeatureNormalization']).transform(currentFeatureSpace)
        #pipelineTest = (pipelineTest).fit((self.trainingSet).data[0:20], (self.trainingSet).target[0:20])
        #currentFeatureSpace= (pipelineTest.named_steps['features']).transform((self.trainingSet).data[0:20])
        
        """
        print ("Feature Set shape:")
        print (currentFeatureSpace.shape)
        
        
        print ("\nFeature Names:")
        print (pipelineTest.named_steps['features'].get_feature_names())
        """
        print ("\nWriting to feature analysis document ...\n")
        # Loop over all individual Feature Sets and retrieve the name (encoded in utf-8) of all the features currently available
        featureNamesList= []
        
        for featureName in pipelineTest.named_steps['features'].get_feature_names():
            featureNamesList.append(featureName)
        # concatenate all the features names created by the pipeline
        """
        # Article Vector Space Model Feature
        sortedVocabularyByValue= sorted(((pipelineTest.named_steps['features']).transformer_list[0][1].steps[0][1].vocabulary_).items(), key= operator.itemgetter(1))
        for x,y in sortedVocabularyByValue:
            #featureNamesList.append(str(x.encode("utf-8")))
            featureNamesList.append("ArticleVSM: " + str(x.encode("utf-8")))
        
        # Lexical Level
        for e in (pipelineTest.named_steps['features']).transformer_list[1][1].get_feature_names():
            featureNamesList.append(e)
        
        # Semantic Level
        for e in (pipelineTest.named_steps['features']).transformer_list[2][1].get_feature_names():
            featureNamesList.append(e)
        """
        # Target Learning Instance -> the values that will be displayed for each individual feature correspond to the values 
        # of the #"learningInstanceId" learning instance
        learningInstanceId= 2
        # File where the Features for the #"learningInstanceId"learning instance will be printed
        featureAnalysisDocument = codecs.open(filename= outputFilePath + "/" + outputFileName, mode= "w", encoding="utf-8")
        # Print Learning Instance information
        featureAnalysisDocument.write("\n Data Instance #" + str(learningInstanceId) + ":\n")
        featureAnalysisDocument.write(str((self.testSet).data[learningInstanceId])) # .decode("utf-8")
        # Print features information and values for the target Learning Instance
        featureAnalysisDocument.write("\n\nFeature Space:\n")
        featureIndexAndImportancePairList= []

        #xrange
        for featureIndex in range(len(featureNamesList)):
            featureImportancesForCurrentFeatureIndex= []
            featureImportancesForCurrentFeatureIndex.append(((pipelineTest.named_steps['clf']).coef_).tolist()[0][featureIndex])
            #featureImportancesForCurrentFeatureIndex.append(((pipelineTest.named_steps['clf']).coef_).tolist()[1][featureIndex])
            #featureImportancesForCurrentFeatureIndex.append(((pipelineTest.named_steps['clf']).coef_).tolist()[2][featureIndex])
            #featureImportancesForCurrentFeatureIndex.append(((pipelineTest.named_steps['clf']).feature_importances_)[featureIndex])

            if spp.issparse(currentFeatureSpace):
                featureAnalysisDocument.write("[" + str(featureIndex) + "] Feature Value= " + str((currentFeatureSpace).toarray()[learningInstanceId][featureIndex]) + " (" + featureNamesList[featureIndex] + ")" + " --> Importance= " + str(featureImportancesForCurrentFeatureIndex) +  "\n")
            else:
                featureAnalysisDocument.write("[" + str(featureIndex) + "] Feature Value= " + str((currentFeatureSpace).tolist()[learningInstanceId][featureIndex]) + " (" + featureNamesList[featureIndex] + ")" + " --> Importance= " + str(featureImportancesForCurrentFeatureIndex) +  "\n")

            featureIndexAndImportancePairList.append( (featureIndex, featureImportancesForCurrentFeatureIndex ))

        
        featureAnalysisDocument.write("\n\n##### Feature set ordered by feature importance #####\n\n")
        
        if len(featureIndexAndImportancePairList[0][1]) == 1:
            # binary classification
            
            featureIndexAndImportancePairList= sorted((featureIndexAndImportancePairList), key= operator.itemgetter(1), reverse= True)
            for x,y in featureIndexAndImportancePairList:
                if not( y == 0):
                    if spp.issparse(currentFeatureSpace):
                        featureAnalysisDocument.write("[" + str(x) + "] Feature Value= " + str((currentFeatureSpace).toarray()[learningInstanceId][x]) + " (" + featureNamesList[x] + ")" + " --> Importance= " + str(y) +  "\n")
                    else:
                        featureAnalysisDocument.write("[" + str(x) + "] Feature Value= " + str((currentFeatureSpace).tolist()[learningInstanceId][x]) + " (" + featureNamesList[x] + ")" + " --> Importance= " + str(y) +  "\n")
        else:
            # multiclass classification
            #TODO: Not tested
            for classIndex in range(len(featureIndexAndImportancePairList[0][1])):
                featureAnalysisDocument.write("\n\n### Ordered regarding class #" + str(classIndex) +  " #####\n\n")
                
                featureIndexAndImportancePairListClassIndex= [ (elem[0], elem[1][classIndex]) for elem in featureIndexAndImportancePairList]
                
                featureIndexAndImportancePairListClassIndex= sorted((featureIndexAndImportancePairListClassIndex), key= operator.itemgetter(1), reverse= True)
                for x,y in featureIndexAndImportancePairListClassIndex:
                    if not( y == 0):
                        if spp.issparse(currentFeatureSpace):
                            featureAnalysisDocument.write("[" + str(x) + "] Feature Value= " + str((currentFeatureSpace).toarray()[learningInstanceId][x]) + " (" + featureNamesList[x] + ")" + " --> Importance= " + str(y) +  "\n")
                        else:
                            featureAnalysisDocument.write("[" + str(x) + "] Feature Value= " + str((currentFeatureSpace).tolist()[learningInstanceId][x]) + " (" + featureNamesList[x] + ")" + " --> Importance= " + str(y) +  "\n")
        featureAnalysisDocument.close()

    def evaluationMetrics(self, title, y, predicted, targetNames, graphicalOutput= False):
        
        print ("\n" + title + "\n")
        
        # comparing prediction with the known categories in order to determine accuracy
        datasetAccuracy= np.mean(predicted == y)
        
        print ("Accuracy= " + str(datasetAccuracy))
        
        # +++ Detailed Performance Analysis +++
        
        # Confusion Matrix table
        print ("\nConfusion Matrix:")
        print (metrics.confusion_matrix(y, predicted))

        
        print(metrics.classification_report(y, predicted, target_names= targetNames))
        
        
        if graphicalOutput:
            # Confusion Matrix Graphic
            pl.imshow(metrics.confusion_matrix(y, predicted), interpolation= "nearest")
            pl.title('Confusion matrix')
            pl.colorbar()
            tick_marks = np.arange(len(targetNames))
            pl.xticks(tick_marks, targetNames, rotation=45)
            pl.yticks(tick_marks, targetNames)
            pl.tight_layout()
            pl.ylabel('True label')
            pl.xlabel('Predicted label')
            pl.show()

    def performROCAnalysis(self, title, y, probaPredictions, targetNames):
        print ("\n" + title + "\n")
        # ROC curve analysis to adjust classifier threshold
        fpr, tpr, thresholds= metrics.roc_curve(y, probaPredictions[:, 1], pos_label= 1, drop_intermediate= True)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate (1 - specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.grid(True)
        plt.show()
        print (str(len(fpr)))
        i= 0
        for i in range(len(fpr)):
            print ("fpr= " + str(fpr[i]) + ", tpr= " + str(tpr[i]) + " --> threshold= " + str(thresholds[i]))
        bestThreshold= 0.0
        v= 0.0
        i= 0
        for i in range(len(fpr)):
            currentV= tpr[i] - fpr[i]
            if currentV > v:
                v= currentV
                bestThreshold= thresholds[i]
        print ("Best threshold= " + str(bestThreshold))
        newPredictions= []
        for probaPred in probaPredictions[:, 1]:
            if probaPred > bestThreshold:
                newPredictions.append(1)
            else:
                newPredictions.append(0)
        self.evaluationMetrics("Evaluation Metrics after ROC Analysis - Modified threshold= " + str(bestThreshold), y, newPredictions, targetNames)

    # Perform feature reduction to output a 3d or 2d visualization of the features
    # Used to perform feature analysis
    #TODO: code snippet with some computations that might be useful for feature analysis and reduction. Not tested in the current version!
    def featureVisualization(self, pipeline):
        
        
        pipeline2= clone(pipeline)
        pipeline3= clone(pipeline)
        
        currentFeatureSpaceBeforeFilter= (pipeline2.named_steps['features']).fit((self.trainingSet).data, (self.trainingSet).target).transform((self.trainingSet).data)
        
        print ("Feature Set shape before Filter:")
        #print (pipeline.named_steps['features'].transformer_list[-1].shape)
        print (pipeline2.named_steps['features'])
        print (currentFeatureSpaceBeforeFilter)
        print (currentFeatureSpaceBeforeFilter.shape)
        print (currentFeatureSpaceBeforeFilter.shape[-1])
        X_indices = np.arange(currentFeatureSpaceBeforeFilter.shape[-1])
        
        print ("Feature Set shape After Filter:")
        print (pipeline3.steps[:-1])
        #print pipeline.steps[:-1].trasnform((self.trainingSet).data)
        #print pipeline.steps[:-1].trasnform((self.trainingSet).data).shape
        
        currentFeatureSpace= (self.trainingSet).data
        
        print ("\nFeature space after each step:")
        ci= 0
        for trasnformerName, transformerObject in pipeline3.steps[:-1]:
            currentFeatureSpace= transformerObject.fit(currentFeatureSpace, (self.trainingSet).target).transform(currentFeatureSpace)
            print ("step #" + str(ci) + ":")
            print (currentFeatureSpace.shape)
            ci= ci + 1
        

        print (currentFeatureSpace)
        print (currentFeatureSpace.shape)
        
        
        # Plot data with final feature space dimensional is <= 3d
        if (currentFeatureSpace.shape[1] == 3):
            fig = plt.figure('3d Data Visualization')
            ax = Axes3D(fig, elev=-150, azim=110)
            
            ax.scatter(currentFeatureSpace[:, 0], currentFeatureSpace[:, 1], currentFeatureSpace[:, 2], c= (self.trainingSet).target, cmap=plt.cm.get_cmap())
            ax.set_title("First three latent directions")
            ax.set_xlabel("Variable 1")
            ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel("Variable 2")
            ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel("Variable 3")
            ax.w_zaxis.set_ticklabels([])
            
            plt.show()
        elif (currentFeatureSpace.shape[1] == 2):
            x_min, x_max = currentFeatureSpace[:, 0].min() - .5, currentFeatureSpace[:, 0].max() + .5
            y_min, y_max = currentFeatureSpace[:, 1].min() - .5, currentFeatureSpace[:, 1].max() + .5
            
            plt.figure('2d Data Visualization')
            plt.clf()
            
            # Plot the training points
            plt.scatter(currentFeatureSpace[:, 0], currentFeatureSpace[:, 1], c= (self.trainingSet).target, cmap=plt.cm.get_cmap())
            plt.xlabel('Variable 1')
            plt.ylabel('Variable 2')
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            
            plt.show()
        
        
        
        
        # Univariate feature selection with F-test for feature scoring
        # We use the default selection function: the 10% most significant features
        
        #selector = SelectPercentile(f_classif, percentile=10)
        #selector.fit(X, y)
        
        selector= (pipeline3.named_steps['filter'])
        
        # Percentile
        #scores = -np.log10(selector.pvalues_)
        #scores /= scores.max()
        
        # Variance
        #scores= selector.variances_
        #print "Scores Length= " + str(len(scores))
        #print selector.get_support()
        
        # Factor Analysis and ICA
        #scores= selector.components_[0]
        #print "Scores Length= " + str(len(scores))
        
        # LDA
        #scores= selector.intercept_
        #print "Scores Length= " + str(len(scores))
        
        
        
        # PCA and RandomizedPCA
        scores= selector.components_[0]
        print (scores)
        print ("Explained variance ratio:")
        print (selector.explained_variance_ratio_)
        
        plt.bar(X_indices, scores, width=.7, label=r'Univariate score ($-Log(p_{value})$)', color='g')
        
        #print "Feature Set shape After Filter:"
        #print (pipeline.named_steps['filter'].transform((self.trainingSet).data).shape)
        #print featureSpace.shape
        
        low = min(scores)
        print ("min= " + str(low))
        high = max(scores)
        print ("max= " + str(high))
        
        for r in scores:
            print ("[" + str(r) + "]")
        
        plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
        
        plt.title("Comparing feature selection")
        plt.xlabel('Feature number')
        #plt.yticks(())
        plt.axis('tight')
        plt.legend(loc='upper right')
        plt.show()

    def plotLearningCurve(self, estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.
        
        ConfigurationASD
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.
        
        title : string
            Title for the chart.
    
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
    
        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
    
        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
    
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.
    
            For integer/None inputs, if ``y`` is binary or multiclass,
            class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
    
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.
    
        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        
        plt.legend(loc="best")
        return plt

    def plotValidationCurve(self, estimator, param_name, param_range, title, X, y, cv=None):
        
        
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name= param_name, param_range= param_range,
            cv=cv, scoring="accuracy")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.title(title)
        plt.xlabel(str(param_name))
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        return plt
    

