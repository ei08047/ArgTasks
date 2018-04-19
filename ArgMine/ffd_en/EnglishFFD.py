from ffd_en.ml.MachineLearningTask import MachineLearningTask
from ffd_en.ConfigurationFFD import ConfigurationFFD
from ffd_en.DatasetLoaderFFD import DatasetLoaderFFD
from utils.Parameters import Parameters


class FFD(MachineLearningTask):
    def outputPredictions(self, modelPath, modelFilename):
        return True

my_parameters= Parameters()
my_configuration = ConfigurationFFD(fixed=True, type= "one_gram")

my_loader = None
load_from_pickle = True
if not load_from_pickle:
    my_loader = DatasetLoaderFFD()
    my_loader.getDatasetInfo()


argDetector = FFD(my_loader, my_configuration,taskName="Fact Feel Detection",outputPath=my_parameters.paths["pathFFD_en"])
argDetector.learn(crossValidationStrategy=4, savedModelFilename=my_parameters.filenames["modelFFD"], featureAnalysis=True,performROCAnalysis=False)
argDetector.outputPredictions(argDetector.outputPath + my_parameters.paths["models"], my_parameters.filenames["modelFFD"])








