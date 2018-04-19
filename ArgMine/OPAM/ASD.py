from OPAM.MyConfiguration import MyConfiguration
from OPAM.MyLoader import MyLoader
from asd_en.ml import MachineLearningTask
from utils.Parameters import Parameters


class ASD(MachineLearningTask):
    def outputPredictions(self, modelPath, modelFilename):
        return True

my_parameters= Parameters()


my_configuration = MyConfiguration(type= "baseline") # , fixed=False
my_loader = MyLoader()
argDetector = ASD(my_loader, my_configuration,taskName="Argumentative Sentence Detection",outputPath=my_parameters.paths["pathASD"])
argDetector.learn(crossValidationStrategy=10, savedModelFilename=my_parameters.filenames["modelASD"], featureAnalysis=True,performROCAnalysis=False)
#argDetector.outputPredictions(argDetector.outputPath + my_parameters.paths["models"], my_parameters.filenames["modelASD"])








