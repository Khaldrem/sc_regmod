"""
models/1
"""

#Imports
import os, sys
path_to_package = os.path.abspath(os.path.join('../'))
if path_to_package not in sys.path:
    sys.path.append(path_to_package)


from src.io import get_filepaths
from src.utils import get_anova_filepaths, check_working_os, get_models_path
from src.models import data_preparation, automated_model_train

# PATH DEFINITIONS
if check_working_os():
    ANOVA_DATASET_PATH = "/home/khaldrem/code/sc_regmod/dataset/anova"
    INDEX_PATH = "/home/khaldrem/code/sc_regmod/dataset/index"
    PHENOTYPES_PATH = "/home/khaldrem/code/sc_regmod/dataset/phenotypes/clean_phenotypes.csv"
    MODELS_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/models"
else:
    ANOVA_DATASET_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\anova"
    INDEX_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\index"
    PHENOTYPES_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\phenotypes\\clean_phenotypes.csv"
    MODELS_BASE_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\models"


DATASET_FILEPATHS = get_anova_filepaths(base_path=ANOVA_DATASET_PATH, type_anova="at_least_one", p_value=0.05, chromosome="all")
MODELS_PATH = get_models_path(MODELS_BASE_PATH)

automated_model_train(choosen_phenotypes=["SM300-Efficiency"],
                    DATASET_PATH=DATASET_FILEPATHS,
                    MODELS_PATH=MODELS_PATH,
                    PHENOTYPES_PATH=PHENOTYPES_PATH)