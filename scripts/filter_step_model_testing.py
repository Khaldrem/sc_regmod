import os, sys

path_to_package = os.path.abspath(os.path.join('../'))
if path_to_package not in sys.path:
    sys.path.append(path_to_package)

import pandas as pd
import numpy as np
import time

from src.io import get_filepaths, read_phylip_file, read_phenotypes_file
from src.utils import get_filename
from src.anova import order_phenotypes_by_id_rows

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def get_id_values(example_fp):
    data = read_phylip_file(example_fp)
    id_rows = []
    for row in data:
        id_rows.append(row.id)
    
    return id_rows


#Como todos los archivos comparten la misma cantidad de filas, se filtran las filas que no se utilizaran en el df de fenotipos
def phenotypes_dataframe_preparation(phenotypes_path, id_rows):
    phenotypes_df = read_phenotypes_file(phenotypes_path)
    phenotypes_df = order_phenotypes_by_id_rows(phenotypes_df, id_rows)

    return phenotypes_df


def data_preparation(filepath, phenotypes_df):
    data = read_phylip_file(filepath)

    cols_data = {}
    for col in range(data.get_alignment_length()):
        key_name = "x" + str(col)
        cols_data[key_name] = list(data[:, col])

    df = pd.DataFrame.from_dict(cols_data, orient="columns")
   
    #Obtengo las llaves
    df_keys = df.keys().tolist() + phenotypes_df.keys().tolist()

    df = pd.concat([df, phenotypes_df], axis='columns', ignore_index=True)
    df.columns = df_keys

    return df


def prepare_dataset(filepaths, PATH, mode, exp_num):
    #If not exists, then the dataset is not prepared
    if not os.path.exists("file_data.csv"):
        #Preparing dataframe
        id_rows = get_id_values(filepaths[0])
        phenotypes_df = phenotypes_dataframe_preparation(PHENOTYPE_DATASET, id_rows)


        #Prepare data

        file_data = {
            "filename": [],
            "data_length": []
        }

        idx = 0
        for f in filepaths:
            filename = get_filename(f)
            data = read_phylip_file(f)

            print()
            print(f"({idx}/{len(filepaths)}) filename: {filename}, seq_len: {data.get_alignment_length()}")
            idx += 1


            file_data["filename"].append(filename)
            file_data["data_length"].append(data.get_alignment_length())

            #Top_n_features filter
            if data.get_alignment_length() > 10:
                df = data_preparation(f, phenotypes_df)
                df.to_csv(f"dataframes/{mode}/{exp_num}/{filename}.csv")
            else:
                print("No supera el top_n_features.")

        file_data_df = pd.DataFrame.from_dict(file_data)
        file_data_df.to_csv("file_data.csv")




def get_metrics(model, evaluation_metrics, y_test, y_pred):
    model_name = str(model).split("(")[0]

    evaluation_metrics[f"r2_{model_name}"].append(r2_score(y_test, y_pred))
    evaluation_metrics[f"MAE_{model_name}"].append(mean_absolute_error(y_test, y_pred))
    evaluation_metrics[f"MSE_{model_name}"].append(mean_squared_error(y_test, y_pred))
    evaluation_metrics[f"RMSE_{model_name}"].append(mean_squared_error(y_test, y_pred, squared=True))


# Config
# Anova: p_value 0.01
# chromosome: haploide-euploide

ANOVA_DATASET = "/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_01/haploide-euploide"
PHENOTYPE_DATASET = "/home/khaldrem/code/sc_regmod/dataset/phenotypes/clean_phenotypes.csv"

# fps = get_filepaths(ANOVA_DATASET)
# prepare_dataset(fps)


files_data_df = pd.read_csv("/home/khaldrem/code/sc_regmod/dataset/models/exp_1_file_data.csv")

overall_time = time.time()
phenotypes = [
    "SM300-Efficiency", "SM300-Rate", "SM300-Lag", "SM300-AUC",
    "SM60-Efficiency",  "SM60-Rate",  "SM60-Lag",  "SM60-AUC",
    "Ratio-Efficiency", "Ratio-Rate", "Ratio-Lag", "Ratio-AUC",
]

phenotypes = phenotypes[:1]

start_phenotype = time.time()
evaluation_metrics = {
    "filename": [],
    "phenotype": [],
    "r2_LinearRegression": [],
    "r2_Ridge": [],
    "r2_Lasso": [],
    "r2_RandomForestRegressor": [],
    # "r2_GradientBoostingRegressor": [],
    "r2_SVR": [],
    "MAE_LinearRegression": [],
    "MAE_Ridge": [],
    "MAE_Lasso": [],
    "MAE_RandomForestRegressor": [],
    # "MAE_GradientBoostingRegressor": [],
    "MAE_SVR": [],
    "MSE_LinearRegression": [],
    "MSE_Ridge": [],
    "MSE_Lasso": [],
    "MSE_RandomForestRegressor": [],
    # "MSE_GradientBoostingRegressor": [],
    "MSE_SVR": [],
    "RMSE_LinearRegression": [],
    "RMSE_Ridge": [],
    "RMSE_Lasso": [],
    "RMSE_RandomForestRegressor": [],
    # "RMSE_GradientBoostingRegressor": [],
    "RMSE_SVR": []
}

for idr, row in files_data_df.iterrows():
    time_per_file = time.time()


    #Filter files that doesnt have top_n_features
    if row["data_length"] > 10:
        df = pd.read_csv(f"/home/khaldrem/code/sc_regmod/dataset/models_dataset/filter/1/{row['filename']}.csv")
        
        #X features
        features_ohe = pd.get_dummies(df.iloc[:, 0:row["data_length"]])
        X = pd.DataFrame(np.array(features_ohe), columns=features_ohe.columns)

        for phenotype in phenotypes:
            evaluation_metrics["filename"].append(row['filename'])
            evaluation_metrics["phenotype"].append(phenotype)

            #Y labels
            y = np.array(df[phenotype])

            #Split Train/Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

            # Iteration over models
            regressors = [
                LinearRegression(),
                Ridge(),
                Lasso(),
                RandomForestRegressor(),
                # GradientBoostingRegressor(),
                SVR()
            ]

            for model in regressors:
                start = time.time()

                model.fit(X_train, y_train)
                train_time = time.time() - start

                y_pred = model.predict(X_test)
                predict_time = time.time() - start

                get_metrics(model, evaluation_metrics, y_test, y_pred)

    print(f"    - id: {idr} - time: {round(time.time() - time_per_file, 4)} sec")

metrics_df = pd.DataFrame.from_dict(evaluation_metrics)
metrics_df.to_csv(f"evaluation_metrics/exp_1_evaluation_metrics.csv")

print(f"Time: {round((time.time() - overall_time)/60, 4)} min.")