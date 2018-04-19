#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import DatasetLoaderASD
from pymongo import MongoClient
from sklearn.externals import joblib

from asd.ConfigurationASD import ConfigurationASD
from asd_en.ml import MachineLearningTask
from utils.Parameters import Parameters

# paths
currentDirectory= os.path.abspath(__file__)
parameters= Parameters()


class ArgumentativeSentenceDetection(MachineLearningTask):
    
    
    #TODO: rewrite visualization
    def outputPredictions(self, modelPath, modelFilename):
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        dbArgMine = mongoClient.ArgMineCorpus
        
        # Sentence's table
        sentenceCollection= dbArgMine.sentence
        
        print "\n\n Output Predictions for test data  ..."
        
        print "\n\n Loading saved model ..."
        
        # Load back the pickled model
        modelPipeline= joblib.load(modelPath + "/" + modelFilename)
        
        print "\n\n Model loaded successfully"
        
        print "\n\n Obtaining Prediction for test data ..."
        
        predictions= modelPipeline.predict((self.testSet).data)
        probaPredictions= modelPipeline.predict_proba((self.testSet).data)
        
        print "\n\n Predictions: done."
        
        # Test set Scores
        self.evaluationMetrics("Test set scores:", (self.testSet).target, predictions, (self.testSet).target_names)
        
        # Output predictions
        
        lastNewsId= -1
        
        newTestData= []
        newTestTarget= []
        newTestProbaTarget= []
        newTestTrueTarget= []
        
        currentNewsData= []
        currentNewsTarget= []
        currentNewsProbaTarget= []
        currentNewsTrueTarget= []
        
        for i in range(len((self.testSet).data)):
            if (not (((self.testSet).data)[i][0] == lastNewsId)) and (not (lastNewsId == -1)):
                newTestData.append(currentNewsData)
                currentNewsData= []
                newTestTarget.append(currentNewsTarget)
                currentNewsTarget= []
                newTestProbaTarget.append(currentNewsProbaTarget)
                currentNewsProbaTarget= []
                newTestTrueTarget.append(currentNewsTrueTarget)
                currentNewsTrueTarget= []
            
            currentNewsData.append(((self.testSet).data)[i])
            
            currentNewsTarget.append(predictions[i])
            
            
            currentNewsProbaTarget.append(probaPredictions[i])
            currentNewsTrueTarget.append(((self.testSet).target)[i])
            
            lastNewsId= ((self.testSet).data)[i][0]
        
        newTestData.append(currentNewsData)
        newTestTarget.append(currentNewsTarget)
        newTestProbaTarget.append(currentNewsProbaTarget)
        newTestTrueTarget.append(currentNewsTrueTarget)
        
        for currentIndex in range(len(newTestData)):
            
            currentSetOfPropositions= newTestData[currentIndex]
            currentSetOfTargets= newTestTarget[currentIndex]
            currentSetOfProbaTargets= newTestProbaTarget[currentIndex]
            currentSetOfTrueTargets= newTestTrueTarget[currentIndex]
            
            
            htmlDocument = open((self.predictionsOutputPath + "/" + str(currentSetOfPropositions[0][0]) + "_argzone.html"), "w")
            
            line = '<html> <head> <meta charset="UTF-8"> </head> <body> <h1> '
            line = line + "Title not found" + " " 
            line = line + '</h1> <br> <br> <p align="justify"> '
            
            for i in range(len(currentSetOfPropositions)):
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": currentSetOfPropositions[i][0]}, {"sentenceId": currentSetOfPropositions[i][1]}]})
                
                if currentSetOfTrueTargets[i] == 1:
                    line = line + ' <font color="green"> '
                    
                    if currentSetOfTargets[i] == 0:
                        line = line + (currentSentence["originalText"])  # .replace('\n', '<br>')
                    else:
                        line = line + "<u>" + (currentSentence["originalText"]) +  "</u> "  # .replace('\n', '<br>')
                    
                    line = line + "</font>"
                    
                else:
                    
                    if currentSetOfTargets[i] == 0:
                        line = line + (currentSentence["originalText"])  # .replace('\n', '<br>')
                    else:
                        line = line + "<u>" + (currentSentence["originalText"]) +  "</u> "  # .replace('\n', '<br>')
                    
                
                # add predicted probabilities
                if currentSetOfTargets[i] == 0:
                    line = line + " " + "[" + str(currentSetOfProbaTargets[i][0]) + "]"  + "<br>"
                else:
                    line = line + " " + "[" + str(currentSetOfProbaTargets[i][1]) + "]"  + "<br>"
            
            line = line + '</p> </body> </html>'
            
            htmlDocument.write(line.encode("utf-8"))
            htmlDocument.close()
            
        
        print "\n\n Output Predictions for test data: Done!"
    




##############################
##########   MAIN   ##########
##############################


asdConfiguration= ConfigurationASD(type= "best")


asdDatasetLoader= DatasetLoaderASD.DatasetLoaderASD()

if(False):
    argDetector= ArgumentativeSentenceDetection(asdDatasetLoader, asdConfiguration, taskName= "Argumentative Sentence Detection", outputPath= parameters.paths["pathASD"])

    argDetector.learn(crossValidationStrategy= 5, savedModelFilename= parameters.filenames["modelASD"], featureAnalysis= True, performROCAnalysis= False)

    argDetector.outputPredictions(argDetector.outputPath + parameters.paths["models"], parameters.filenames["modelASD"])

    """
    print " Ploting the Learning Curve ...\n"
    learningCurvePlot= argDetector.plotLearningCurve(argDetector.pipeline, argDetector.taskName + " - Learning Curve", np.asarray((argDetector.trainingSet).data), (argDetector.trainingSet).target, ylim=(0.3, 1.01), cv= 3)
    learningCurvePlot.show()
    """

    """
    print "\n Plotting Validation Curve ...\n"
    #print "available keys:"
    #print pipeline.get_params().keys()
    #validationCurvePlot= plot_validation_curve(estimator= pipeline, param_name= "clf__C", param_range= np.linspace(0.1, 0.99, 5), title= "Validation Curve", X= np.asarray((self.trainingSet).data), y= (self.trainingSet).target, cv=5)
    validationCurvePlot= argDetector.plotValidationCurve(estimator= argDetector.pipeline, param_name= "features__wordCouple__numberOFKeywords", param_range= [0,1,2], title= argDetector.taskName + " - Validation Curve", X= np.asarray((argDetector.trainingSet).data), y= (argDetector.trainingSet).target, cv=3)
    validationCurvePlot.show()
    """
