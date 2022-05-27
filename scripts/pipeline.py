import os, sys
path_to_package = os.path.abspath(os.path.join('../'))
if path_to_package not in sys.path:
    sys.path.append(path_to_package)

from time import time

from src.io import check_if_anova_files_exists, check_if_models_exists
from src.utils import get_anova_filepaths
from src.anova import do_multiprocess_anova
from src.models import filter_model_step

"""
    Pipeline:
        Paso 1: Realizar/Verificar datos Anova
        Paso 2: Entrenar modelo para filtrar datos del paso anterior
        Paso 3: Entrenar modelo final
"""


class Pipeline():
    def __init__(self,
            exp_number = "",
            p_value_threshold = 0.0, 
            type_anova = "", 
            chromosome = "",
            filter_model_name = "",
            final_model_name = "") -> None:
        #Se debe cambiar los path si se corre en windows
        self.CLEAN_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/clean"
        self.ANOVA_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/anova"
        self.INDEX_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/index"
        self.MODELS_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/models" 
        self.PHENOTPES_FILE_PATH = "/home/khaldrem/code/sc_regmod/dataset/phenotypes/clean_phenotypes.csv"

        #Verificar parametros

        #Asignar parametros
        self.exp_number = exp_number
        self.p_value_threshold = p_value_threshold
        self.type_anova = type_anova
        self.chromosome = chromosome
        self._filter_model_name = filter_model_name
        self.final_model_name = final_model_name


    def run(self):
        start = time()

        print("===== Running experiment =====")
        print(f"    Config:")
        print(f"        - p_value_threshold: {self.p_value_threshold}")
        print(f"        - type_anova:        {self.type_anova}")
        print(f"        - chromosome:        {self.chromosome}")
        print()

        #Check if anova data exists
        if not check_if_anova_files_exists(self.ANOVA_BASE_PATH, self.type_anova, self.p_value_threshold, self.chromosome):
            print(f" + Anova files with that configuration doesn't exists.")
            do_multiprocess_anova(type_anova=self.type_anova, phenotype="", 
                                p_value=self.p_value_threshold, chromosome=self.chromosome, pool_workers=32, 
                                ANOVA_DATASET_PATH=self.ANOVA_BASE_PATH, 
                                CLEAN_DATASET_PATH=self.CLEAN_BASE_PATH, 
                                PHENOTYPES_PATH=self.PHENOTPES_FILE_PATH, 
                                INDEX_PATH=self.INDEX_BASE_PATH)
        else:
            print(f" + Anova files already exists.")
        

        #Set ANOVA_DATASET var
        ANOVA_DATASET = get_anova_filepaths(base_path = self.ANOVA_BASE_PATH, type_anova=self.type_anova, 
                                            p_value=self.p_value_threshold, chromosome=self.chromosome)

        #Check if filter models were created
        if not check_if_models_exists(self.exp_number, self.MODELS_BASE_PATH, "filter"):
            print(f" + Training filter models.")

            choosen_phenotypes = [
                "SM300-Efficiency"
            ]

            filter_model_step(choosen_phenotypes=choosen_phenotypes, DATASET_PATH=ANOVA_DATASET, PHENOTYPES_PATH=self.PHENOTPES_FILE_PATH)





         


        end = time()
        print(f"===== Completed in {round((end-start)/60, 2)} min. =====")



# DATASET_FILEPATHS = get_anova_filepaths(base_path=ANOVA_DATASET_PATH, 
#                                         type_anova="at_least_one", 
#                                         p_value=0.05, 
#                                         chromosome="all")

# filter_model_step(choosen_phenotypes=VALID_PHENOTYPES, DATASET_PATH=DATASET_FILEPATHS)

p = Pipeline(exp_number="1", p_value_threshold=0.05, type_anova="at_least_one", chromosome="all")
p.run()

# 483279