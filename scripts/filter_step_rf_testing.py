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


def prepare_dataset(filepaths):
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
                df.to_csv(f"dataframes/{filename}.csv")
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

fps = get_filepaths(ANOVA_DATASET)
prepare_dataset(fps)
files_data_df = pd.read_csv("file_data.csv")

overall_time = time.time()
phenotypes = [
    "SM300-Efficiency", "SM300-Rate", "SM300-Lag", "SM300-AUC",
    "SM60-Efficiency",  "SM60-Rate",  "SM60-Lag",  "SM60-AUC",
    "Ratio-Efficiency", "Ratio-Rate", "Ratio-Lag", "Ratio-AUC",
]

phenotypes = phenotypes[:1]

for phenotype in phenotypes:
    start_phenotype = time.time()
    best_params_metric = {
        "filename": [],
        "n_estimators": [],
        "max_features": [],
        "max_depth": [],
        "criterion": []
    }
    
    for idx, row in files_data_df.iterrows():
        #Filter files that doesnt have top_n_features
        if row["data_length"] > 10:
            best_params_metric["filename"].append(row['filename'])
            df = pd.read_csv(f"dataframes/{row['filename']}.csv")

            #Y labels
            y = np.array(df[phenotype])

            #X features
            features_ohe = pd.get_dummies(df.iloc[:, 0:row["data_length"]])
            X = pd.DataFrame(np.array(features_ohe), columns=features_ohe.columns)

            #Split Train/Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

            #GridsearchCV over RandomForest
            grid = { 
                'n_estimators': [200,300,400,500],
                'max_features': ['sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8],
                'criterion' :['absolute_error'],
            }
            
            scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
            
            grid_RF = GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid=grid,
                        cv=5, 
                        n_jobs=-1
                        )
                                    
            grid_RF.fit(X_train, y_train)

            print(grid_RF.best_params_)

            best_params_metric["n_estimators"].append(grid_RF.best_params_["n_estimators"])
            best_params_metric["max_features"].append(grid_RF.best_params_["max_features"])
            best_params_metric["max_depth"].append(grid_RF.best_params_["max_depth"])
            best_params_metric["criterion"].append(grid_RF.best_params_["criterion"])
    
    print(f"{phenotype} - Time per phenotype: {round((time.time() - start_phenotype)/60, 4)} min.")

    best_params_df = pd.DataFrame.from_dict(best_params_metric)
    best_params_df.to_csv(f"params/{phenotype}_rf_best_params.csv")

print(f"Time: {round((time.time() - overall_time)/60, 4)} min.")



# y_pred = grid_Ridge.best_estimator_.predict(X_test)

# print(y_pred)
# print(y_test)

# print(f"Best params: {grid_Ridge.best_params_}")
# print()
# print(f"Score train set: {grid_Ridge.best_estimator_.score(X_train, y_train)}")
# print(f"Score test set: {grid_Ridge.best_estimator_.score(X_test, y_test)}")
# print()

# print("On test set ...")
# print(f"r2: {r2_score(y_test, y_pred)}")
# print(f"MAE: {mean_absolute_error(y_test, y_pred)}")



