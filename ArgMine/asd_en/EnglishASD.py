from asd_en.ConfigurationASD import ConfigurationASD
from asd_en.DatasetLoaderASD import DatasetLoaderASD
from asd_en.ml.MachineLearningTask import MachineLearningTask
from utils.Parameters import Parameters
import sys


def to_bool(param):
    if param == 'False':
        return False
    else:
        return True

class ASD(MachineLearningTask):
    def outputPredictions(self, modelPath, modelFilename):
        return True

my_parameters= Parameters()

load_data_from_file = to_bool(sys.argv[1])
isFixed = to_bool(sys.argv[2])
type = sys.argv[3]
feature_analysis = to_bool(sys.argv[4])
roc_analysis = to_bool(sys.argv[5])

my_configuration = ConfigurationASD(fixed=isFixed, type=type)

my_loader = None
if not load_data_from_file:
    my_loader = DatasetLoaderASD()
    my_loader.getDatasetInfo()
    argDetector = ASD(my_loader, my_configuration,taskName="Argumentative Sentence Detection",outputPath=my_parameters.paths["pathASD"])
else:
    argDetector = ASD(my_loader, my_configuration, taskName="Argumentative Sentence Detection",outputPath=my_parameters.paths["pathASD"])

argDetector.learn(crossValidationStrategy=3, savedModelFilename=my_parameters.filenames["modelASD"], featureAnalysis=feature_analysis,performROCAnalysis=roc_analysis)
argDetector.outputPredictions(argDetector.outputPath + my_parameters.paths["models"], my_parameters.filenames["modelASD"])








