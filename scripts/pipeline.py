import os, sys
path_to_package = os.path.abspath(os.path.join('../'))
if path_to_package not in sys.path:
    sys.path.append(path_to_package)

from time import time

from src.io import create_id_row_file, create_file_data_csv, check_if_anova_files_exists, check_if_models_exists, check_dataset_for_model_step, get_filepaths
from src.utils import get_anova_filepaths
from src.anova import do_multiprocess_anova
from src.models import prepare_dataset, filter_model_step, final_model_step, prepare_dataset_final

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
            top_n_feature = 0) -> None:
        #Se debe cambiar los path si se corre en windows
        self.CLEAN_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/clean"
        self.ANOVA_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/anova"
        self.INDEX_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/index"
        self.MODELS_BASE_PATH = "/home/khaldrem/code/sc_regmod/dataset/models"
        self.MODELS_DATASET_PATH = "/home/khaldrem/code/sc_regmod/dataset/models_dataset"
        self.PHENOTPES_FILE_PATH = "/home/khaldrem/code/sc_regmod/dataset/phenotypes/clean_phenotypes.csv"

        #Asignar parametros
        self.exp_number = exp_number
        self.p_value_threshold = p_value_threshold
        self.type_anova = type_anova
        self.chromosome = chromosome
        self.top_n_feature = top_n_feature
        self.choosen_phenotypes = [
            "SM300-Efficiency", "SM300-Rate", "SM300-Lag", "SM300-AUC",
            "SM60-Efficiency",  "SM60-Rate",  "SM60-Lag",  "SM60-AUC",
            "Ratio-Efficiency", "Ratio-Rate", "Ratio-Lag", "Ratio-AUC",
        ]


    def run(self):
        start = time()

        print("===== Running experiment =====")
        print(f"    Config:")
        print(f"        - p_value_threshold: {self.p_value_threshold}")
        print(f"        - type_anova:        {self.type_anova}")
        print(f"        - chromosome:        {self.chromosome}")
        print(f"        - top_n_feature:     {self.top_n_feature}")
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

        #Create id file
        create_id_row_file(get_filepaths(ANOVA_DATASET)[0], self.MODELS_BASE_PATH, self.exp_number)

        #Creates files data
        create_file_data_csv(ANOVA_DATASET, self.MODELS_BASE_PATH, self.exp_number)

        #Prepare dataset if necesary
        prepare_dataset("filter", self.MODELS_BASE_PATH, 
                    self.MODELS_DATASET_PATH, self.PHENOTPES_FILE_PATH,
                    self.exp_number, self.top_n_feature)


        # FILTER STEP
        if not check_if_models_exists(self.exp_number, self.MODELS_BASE_PATH, "filter"):
            filter_model_step(
                choosen_phenotypes=self.choosen_phenotypes,
                top_n_features=self.top_n_feature,
                exp_number = self.exp_number,
                MODELS_BASE_PATH = self.MODELS_BASE_PATH,
                MODELS_DATASET_PATH=self.MODELS_DATASET_PATH,
                INDEX_PATH=self.INDEX_BASE_PATH)
        else:
            print(f" + Models files already exists.")

        
        #Prepare data for final step
        prepare_dataset_final(choosen_phenotypes=self.choosen_phenotypes, top_n_features=self.top_n_feature,
                            exp_num=self.exp_number, models_base_path=self.MODELS_BASE_PATH, 
                            models_dataset_path=self.MODELS_DATASET_PATH, index_dir_path=self.INDEX_BASE_PATH, 
                            phenotypes_dataset_path=self.PHENOTPES_FILE_PATH)

        # FINAL STEP
        if not check_if_models_exists(self.exp_number, self.MODELS_BASE_PATH, "final"):
            print(f" + Training final model.")

            final_model_step(choosen_phenotypes=self.choosen_phenotypes, exp_number=self.exp_number,
                        MODELS_BASE_PATH=self.MODELS_BASE_PATH, MODELS_DATASET_PATH=self.MODELS_DATASET_PATH)
        else:
            print(f" + Final model already exists.")

        end = time()
        print(f"===== Completed in {round((end-start)/60, 2)} min. =====")


# ----------------------------------------------------------------- #
# ------------------------- Experimento 1 ------------------------- #
# ----------------------------------------------------------------- #

# config = {
#     "exp_number": "1",
#     "p_value_threshold": 0.01,
#     "type_anova": "at_least_one",
#     "chromosome": "all",
#     "top_n_feature": 10
# }

# p1 = Pipeline(exp_number=config["exp_number"], 
#             p_value_threshold=config["p_value_threshold"], 
#             type_anova=config["type_anova"], 
#             chromosome=config["chromosome"],
#             top_n_feature=config["top_n_feature"])
# p1.run()



# ----------------------------------------------------------------- #
# ------------------------- Experimento 2 ------------------------- #
# ----------------------------------------------------------------- #

# config = {
#     "exp_number": "2",
#     "p_value_threshold": 0.01,
#     "type_anova": "at_least_one",
#     "chromosome": "haploide-euploide",
#     "top_n_feature": 10
# }

# p2 = Pipeline(exp_number=config["exp_number"], 
#             p_value_threshold=config["p_value_threshold"], 
#             type_anova=config["type_anova"], 
#             chromosome=config["chromosome"],
#             top_n_feature=config["top_n_feature"])
# p2.run()



# # ----------------------------------------------------------------- #
# # ------------------------- Experimento 3 ------------------------- #
# # ----------------------------------------------------------------- #

# config = {
#     "exp_number": "3",
#     "p_value_threshold": 0.01,
#     "type_anova": "at_least_one",
#     "chromosome": "diploides-euploides",
#     "top_n_feature": 10
# }

# p3 = Pipeline(exp_number=config["exp_number"], 
#             p_value_threshold=config["p_value_threshold"], 
#             type_anova=config["type_anova"], 
#             chromosome=config["chromosome"],
#             top_n_feature=config["top_n_feature"])
# p3.run()



# ----------------------------------------------------------------- #
# ------------------------- Experimento 4 ------------------------- #
# ----------------------------------------------------------------- #

config = {
    "exp_number": "4",
    "p_value_threshold": 0.01,
    "type_anova": "at_least_one",
    "chromosome": "---",
    "top_n_feature": 10
}

p4 = Pipeline(exp_number=config["exp_number"], 
            p_value_threshold=config["p_value_threshold"], 
            type_anova=config["type_anova"], 
            chromosome=config["chromosome"],
            top_n_feature=config["top_n_feature"])
p4.run()
