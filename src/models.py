
from matplotlib.pyplot import axes
import numpy as np
import pandas as pd
from itertools import product

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from src.anova import order_phenotypes_by_id_rows
from src.io import read_phylip_file, read_phenotypes_file, get_filepaths, write_pandas_csv, load_json
from src.indexes import insert_models_feature_importance
import time

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedKFold, KFold, cross_validate
from sklearn.linear_model import LassoCV, Ridge, Lasso, RidgeCV, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR

from src.utils import check_directory, check_working_os, get_filename

import joblib
import json
import os, glob


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


def prepare_dataset(step, models_path, models_dataset_path, phenotypes_dataset_path, exp_num, top_n_features):
    dataset_path = f"{models_dataset_path}/{step}/{exp_num}"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    #Load id_rows
    f = open(f"{models_path}/id_rows.json", "r")
    id_rows = json.load(f)
    f.close()

    #File_data
    file_data = pd.read_csv(f"{models_path}/exp_{exp_num}_file_data.csv")

    #Prepare the phenotypes df
    phenotypes_df = phenotypes_dataframe_preparation(phenotypes_dataset_path, id_rows[exp_num])

    if step == "filter":
        if not (file_data[file_data["data_length"] > top_n_features].shape[0] == len(glob.glob(f'{dataset_path}/*.csv'))):
            for _, row in file_data.iterrows():
                if row["data_length"] > top_n_features:
                    df = data_preparation(row["filepath"], phenotypes_df)
                    df.to_csv(f"{dataset_path}/{row['filename']}.csv")
        else:
            print(f" + Dataset files for filter step already exists.")

    if step == "final":
        pass


def prepare_dataset_final(choosen_phenotypes=[], top_n_features=10, 
                        exp_num=1, models_base_path="", models_dataset_path="", 
                        index_dir_path="", phenotypes_dataset_path=""):
    #Load id_rows
    f = open(f"{models_base_path}/id_rows.json", "r")
    id_rows = json.load(f)
    f.close()
    
    file_data = pd.read_csv(f"{models_base_path}/exp_{exp_num}_file_data.csv")
    phenotypes_df = phenotypes_dataframe_preparation(phenotypes_dataset_path, id_rows[exp_num])

    path_to_write = f"{models_dataset_path}/final/{exp_num}"
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)

    for phenotype in choosen_phenotypes:
        path_to_write_phenotype = f"{path_to_write}/{phenotype}_final_data.csv"
        if not os.path.exists(path_to_write_phenotype):
            data_to_write = {}

            for _, row in file_data.iterrows():
                if row["data_length"] > top_n_features:
                    #Load index data
                    index_data = load_json(index_dir_path, row["filename"])
                    selected_columns = sorted(index_data["models"]["filter"][phenotype][exp_num])

                    #Load data
                    df = pd.read_csv(f"{models_dataset_path}/filter/{exp_num}/{row['filename']}.csv")

                    #Iterate over selected columns
                    for col in selected_columns:
                        data_to_write[f"{row['filename']}_x{col}"] = df[f"x{col}"].tolist()

                    
            
            data_to_write_df = pd.DataFrame.from_dict(data_to_write)
            
            df_keys = data_to_write_df.keys().tolist() + phenotypes_df.keys().tolist()
            data_to_write_df = pd.concat([data_to_write_df, phenotypes_df], axis='columns', ignore_index=True)
            data_to_write_df.columns = df_keys

            data_to_write_df.to_csv(path_to_write_phenotype)
        




def filter_data_based_on_index_file(data=[], phenotype="", exp_number=0, index_dir_path="", filename=""):
    # Consider data as a DataFrame
    index_data = load_json(index_dir_path, filename)
    format_cols = []
    sorted_important_cols = sorted(index_data["models"]["filter"][phenotype][str(exp_number)])
    for col in sorted_important_cols:
        format_cols.append(f"x{str(col)}")
    
    return data[format_cols]



def set_scoring_metrics(phenotype, scoring_metrics, y_test, y_pred):
    scoring_metrics[f"{phenotype}_r2"].append(round(r2_score(y_test, y_pred), 4))
    scoring_metrics[f"{phenotype}_MAE"].append(round(mean_absolute_error(y_test, y_pred), 4))
    scoring_metrics[f"{phenotype}_MSE"].append(round(mean_squared_error(y_test, y_pred), 4))
    scoring_metrics[f"{phenotype}_RMSE"].append(round(mean_squared_error(y_test, y_pred, squared=True), 4))


def set_metrics_training_model_time(phenotype, metrics, start_time):
    key_format = f"{phenotype}_training_model_time"
    metrics[key_format].append(round((time.time() - start_time), 4))


def set_final_scoring_metrics(models_metric, model_name, phenotype, y_test, y_pred):
    models_metric["model_name"].append(model_name)
    models_metric["phenotype"].append(phenotype)

    models_metric["r2"].append(round(r2_score(y_test, y_pred), 8))
    models_metric["MAE"].append(round(mean_absolute_error(y_test, y_pred), 8))
    models_metric["MSE"].append(round(mean_squared_error(y_test, y_pred), 8))
    models_metric["RMSE"].append(round(mean_squared_error(y_test, y_pred, squared=True), 8))



def filter_model_step(choosen_phenotypes=[], top_n_features = 5, exp_number=0, 
                    MODELS_BASE_PATH="", MODELS_DATASET_PATH="", INDEX_PATH=""):
    file_data_df = pd.read_csv(f"{MODELS_BASE_PATH}/exp_{exp_number}_file_data.csv")
    
    scoring_metrics = {
        "filename": [],
        "SM300-Efficiency_r2": [],
        "SM300-Efficiency_MAE": [],
        "SM300-Efficiency_MSE": [],
        "SM300-Efficiency_RMSE": [],

        "SM300-Rate_r2": [],
        "SM300-Rate_MAE": [],
        "SM300-Rate_MSE": [],
        "SM300-Rate_RMSE": [],
    
        "SM300-Lag_r2": [],
        "SM300-Lag_MAE": [],
        "SM300-Lag_MSE": [],
        "SM300-Lag_RMSE": [],

        "SM300-AUC_r2": [],
        "SM300-AUC_MAE": [],
        "SM300-AUC_MSE": [],
        "SM300-AUC_RMSE": [],

        "SM60-Efficiency_r2": [],
        "SM60-Efficiency_MAE": [],
        "SM60-Efficiency_MSE": [],
        "SM60-Efficiency_RMSE": [],
        
        "SM60-Rate_r2": [],
        "SM60-Rate_MAE": [],
        "SM60-Rate_MSE": [],
        "SM60-Rate_RMSE": [],

        "SM60-Lag_r2": [],
        "SM60-Lag_MAE": [],
        "SM60-Lag_MSE": [],
        "SM60-Lag_RMSE": [],

        "SM60-AUC_r2": [],
        "SM60-AUC_MAE": [],
        "SM60-AUC_MSE": [],
        "SM60-AUC_RMSE": [],

        "Ratio-Efficiency_r2": [],
        "Ratio-Efficiency_MAE": [],
        "Ratio-Efficiency_MSE": [],
        "Ratio-Efficiency_RMSE": [],
        
        "Ratio-Rate_r2": [],
        "Ratio-Rate_MAE": [],
        "Ratio-Rate_MSE": [],
        "Ratio-Rate_RMSE": [],

        "Ratio-Lag_r2": [],
        "Ratio-Lag_MAE": [],
        "Ratio-Lag_MSE": [],
        "Ratio-Lag_RMSE": [],

        "Ratio-AUC_r2": [],
        "Ratio-AUC_MAE": [],
        "Ratio-AUC_MSE": [],
        "Ratio-AUC_RMSE": [],

    }

    time_metrics = {
        "filename": [],
        "total_per_file": [],

        "SM300-Efficiency_training_model_time": [],
        "SM300-Efficiency_test_model_time": [],

        "SM300-Rate_training_model_time": [],
        "SM300-Rate_test_model_time": [],

        "SM300-Lag_training_model_time": [],
        "SM300-Lag_test_model_time": [],

        "SM300-AUC_training_model_time": [],
        "SM300-AUC_test_model_time": [],

        "SM60-Efficiency_training_model_time": [],
        "SM60-Efficiency_test_model_time": [],

        "SM60-Rate_training_model_time": [],
        "SM60-Rate_test_model_time": [],

        "SM60-Lag_training_model_time": [],
        "SM60-Lag_test_model_time": [],

        "SM60-AUC_training_model_time": [],
        "SM60-AUC_test_model_time": [],

        "Ratio-Efficiency_training_model_time": [],
        "Ratio-Efficiency_test_model_time": [],

        "Ratio-Rate_training_model_time": [],
        "Ratio-Rate_test_model_time": [],

        "Ratio-Lag_training_model_time": [],
        "Ratio-Lag_test_model_time": [],

        "Ratio-AUC_training_model_time": [],
        "Ratio-AUC_test_model_time": []
    }

    for idx, row in file_data_df.iterrows():
        print(f"            ({idx}/{file_data_df.shape[0]})")

        if row["data_length"] > top_n_features:
            total_time = time.time()

            scoring_metrics["filename"].append(row['filename'])
            time_metrics["filename"].append(row['filename'])

            df = pd.read_csv(f"{MODELS_DATASET_PATH}/filter/{exp_number}/{row['filename']}.csv")

            #X features
            features_ohe = pd.get_dummies(df.iloc[:, 1:row["data_length"]])
            X = pd.DataFrame(np.array(features_ohe), columns=features_ohe.columns)

            for phenotype in choosen_phenotypes:
                #Y labels
                y = np.array(df[phenotype])
                
                #Split Train/Test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

                # GridSearchCV
                # Los diferentes rangos que puede abarcar alpha hacen que el algoritmo se demore
                # a veces demasiado. Por lo que solo he limitado la busqueda a 50 valores.
                # en general, cae siempre entre valores de 20 a 21, lo cual no se si es demasiado.
                params_grid_ridge = {
                    "alpha": np.linspace(0, 1, 50)
                    # "alpha": np.logspace(0.001, 100, 100)
                    # "alpha": np.arange(0.1, 25, 1)
                }

                scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']

                training_model_time = time.time()
                grid_Ridge = GridSearchCV(estimator=Ridge(), 
                                        param_grid=params_grid_ridge,
                                        cv=5, 
                                        n_jobs=-1, 
                                        scoring=scoring,
                                        refit='neg_mean_absolute_error')

                grid_Ridge.fit(X_train, y_train)
                time_metrics[f"{phenotype}_training_model_time"].append(round((time.time() - training_model_time), 4))


                #Save predictions
                test_model_time = time.time()
                y_pred = grid_Ridge.best_estimator_.predict(X_test)
                time_metrics[f"{phenotype}_test_model_time"].append(round((time.time() - test_model_time), 4))
                set_scoring_metrics(phenotype, scoring_metrics, y_test, y_pred)

                prediction_df = pd.DataFrame.from_dict({"y_test": y_test, "y_pred": y_pred})
                prediction_df.to_csv(f"{MODELS_BASE_PATH}/filter/{exp_number}/csv/prediction_results_{row['filename']}_{phenotype}.csv")

                # Save the model
                if check_working_os():
                    joblib.dump(grid_Ridge.best_estimator_, f"{MODELS_BASE_PATH}/filter/{exp_number}/{row['filename']}_{phenotype}.joblib")
                else:
                    joblib.dump(grid_Ridge.best_estimator_, f"{MODELS_BASE_PATH}\\filter\\{exp_number}\\{row['filename']}_{phenotype}.joblib")

                # Save cv_results
                cv_res = pd.DataFrame(grid_Ridge.cv_results_)
                cv_res.to_csv(f"{MODELS_BASE_PATH}/filter/{exp_number}/csv/cv_results_{row['filename']}_{phenotype}.csv")
                
                # Save coef & names
                coefs = pd.DataFrame(grid_Ridge.best_estimator_.coef_, 
                                columns=["coefficients"], 
                                index=grid_Ridge.best_estimator_.feature_names_in_)
                coefs.to_csv(f"{MODELS_BASE_PATH}/filter/{exp_number}/csv/coef_results_{row['filename']}_{phenotype}.csv")

                
                # Get most important features
                coefs = coefs.sort_values(by="coefficients", ascending=False)
                most_important_features = set()
                for i, _ in coefs.iterrows():
                    if len(most_important_features) == top_n_features:
                        break
                    else:
                        feature_index = int(i.split("_")[0].split("x")[1])
                        most_important_features.add(feature_index)

                # Write these features on the index
                insert_models_feature_importance(list(most_important_features), "models", phenotype, exp_number, INDEX_PATH, row['filename'])
            
            time_metrics["total_per_file"].append(round((time.time() - total_time), 4))
    
    #save scoring metrics
    scoring_metrics_df = pd.DataFrame.from_dict(scoring_metrics)
    scoring_metrics_df.to_csv(f"{MODELS_BASE_PATH}/filter/{exp_number}/csv/scoring_metrics_overall.csv")

    #save time metrics
    time_metrics_df = pd.DataFrame.from_dict(time_metrics)
    time_metrics_df.to_csv(f"{MODELS_BASE_PATH}/filter/{exp_number}/csv/time_metrics_overall.csv")



def final_model_step(choosen_phenotypes=[], exp_number=0, MODELS_BASE_PATH="", MODELS_DATASET_PATH=""):
    os.makedirs(f"{MODELS_BASE_PATH}/final/{exp_number}/pred")
    os.makedirs(f"{MODELS_BASE_PATH}/final/{exp_number}/models")

    models_metric = {
        "model_name": [],
        "phenotype": [],
        "r2": [],
        "MAE": [],
        "MSE": [],
        "RMSE": []
    }

    time_metrics = {
        "model_name": [],
        "phenotype": [],
        "total_time": [],
        "training_time": [],
        "test_time": []
    }

    models_to_train = ["Ridge", "Lasso", "RandomForestRegressor"]

    for phenotype in choosen_phenotypes:
        y_prediction = {
            "y_test": [],
            "Ridge_y_pred": [],
            "Lasso_y_pred": [],
            "RandomForestRegressor_y_pred": [],
        }
        
        total_time_start = time.time()
        
        df = pd.read_csv(f"{MODELS_DATASET_PATH}/final/{exp_number}/{phenotype}_final_data.csv")

        features_ohe = pd.get_dummies(df.iloc[:, 1:(df.shape[1] - 12)])
        X = pd.DataFrame(np.array(features_ohe), columns=features_ohe.columns)
        
        y = np.array(df[phenotype])
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

        for model in models_to_train:
            time_metrics["model_name"].append(model)
            time_metrics["phenotype"].append(phenotype)

            if model == "Ridge":
                ridge_model = RidgeCV(alphas = np.linspace(0.001, 1, 25), 
                            cv=KFold(n_splits=5, shuffle=False, random_state=None), 
                            scoring='neg_mean_absolute_error')

                ridge_cv = ridge_model.fit(X_train, y_train)
                time_metrics["training_time"].append(round((time.time() - total_time_start), 4))

                y_pred_ridge = ridge_cv.predict(X_test)
                time_metrics["test_time"].append(round((time.time() - total_time_start), 4))

                set_final_scoring_metrics(models_metric, model, phenotype, y_test, y_pred_ridge)
                y_prediction[f"{model}_y_pred"] = y_pred_ridge

                #Save model
                joblib.dump(ridge_cv, f"{MODELS_BASE_PATH}/final/{exp_number}/models/ridge_{phenotype}.joblib")

                #Save coeficientes
                coefs_dict = {
                    "features": ridge_cv.feature_names_in_,
                    "coefficients": ridge_cv.coef_,
                    "filename": [i.split('_')[0] for i in ridge_cv.feature_names_in_]
                }
            
                coefs = pd.DataFrame.from_dict(coefs_dict)
                coefs = coefs.sort_values(by="coefficients", ascending=False)
                coefs.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/csv/ridge_coef_results_{phenotype}.csv")

            if model == "Lasso":
                lasso_model = LassoCV(alphas = np.linspace(0.001, 1, 25),
                                    cv=KFold(n_splits=5, shuffle=False, random_state=None), 
                                    n_jobs=-1)
                
                lasso_cv = lasso_model.fit(X_train, y_train)
                time_metrics["training_time"].append(round((time.time() - total_time_start), 4))

                y_pred_lasso = lasso_cv.predict(X_test)
                time_metrics["test_time"].append(round((time.time() - total_time_start), 4))


                set_final_scoring_metrics(models_metric, model, phenotype, y_test, y_pred_lasso)
                y_prediction[f"{model}_y_pred"] = y_pred_lasso

                #Save model
                joblib.dump(lasso_cv, f"{MODELS_BASE_PATH}/final/{exp_number}/models/lasso_{phenotype}.joblib")

                #Save coeficientes
                coefs_dict = {
                    "features": ridge_cv.feature_names_in_,
                    "coefficients": lasso_cv.coef_,
                    "filename": [i.split('_')[0] for i in ridge_cv.feature_names_in_]
                }
           
                coefs = pd.DataFrame.from_dict(coefs_dict)
                coefs = coefs.sort_values(by="coefficients", ascending=False)
                coefs.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/csv/lasso_coef_results_{phenotype}.csv")


            if model == "RandomForestRegressor":
                params_grid_rf = {
                                    'n_estimators': [200, 300, 400, 500, 600],
                                    'max_features': ['sqrt', 'log2'],
                                    'max_depth' : [4,5,6,7,8],
                                    'criterion' :['absolute_error'],
                                }
                
                rf_model = GridSearchCV(
                                RandomForestRegressor(), 
                                param_grid=params_grid_rf, 
                                scoring='neg_mean_absolute_error', 
                                n_jobs=-1)
                
                rf_gscv = rf_model.fit(X_train, y_train)
                time_metrics["training_time"].append(round((time.time() - total_time_start), 4))

                y_pred_rf = rf_gscv.best_estimator_.predict(X_test)
                time_metrics["test_time"].append(round((time.time() - total_time_start), 4))

                set_final_scoring_metrics(models_metric, model, phenotype, y_test, y_pred_rf)
                y_prediction[f"{model}_y_pred"] = y_pred_rf

                #Save model
                joblib.dump(rf_gscv.best_estimator_, f"{MODELS_BASE_PATH}/final/{exp_number}/models/rf_{phenotype}.joblib")

                feats = {
                    "feature": [],
                    "absolute_error": [],
                    "filename": []
                }

                for feature, importance in zip(features_ohe.columns, rf_gscv.best_estimator_.feature_importances_):
                    feats["feature"].append(feature)
                    feats["absolute_error"].append(importance)
                    feats["filename"].append(feature.split("_")[0])

                importances = pd.DataFrame.from_dict(feats, orient='columns').sort_values(by="absolute_error", ascending=False).reset_index()
                importances.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/csv/rf_coef_results_{phenotype}.csv")

        #Save results
        y_prediction["y_test"] = y_test

        y_predictions_df = pd.DataFrame.from_dict(y_prediction)
        y_predictions_df.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/pred/{phenotype}_y_pred.csv")


        time_metrics["total_time"].append(round((time.time() - total_time_start), 4))


    #Save metrics
    models_metric_df = pd.DataFrame.from_dict(models_metric)
    models_metric_df.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/models_metrics_results.csv")


        

"""

def final_model_step(choosen_phenotypes=[], exp_number=0, top_n_features = 5, MODELS_BASE_PATH="", DATASET_PATH="", PHENOTYPES_PATH="", INDEX_PATH=""):
    filepaths = get_filepaths(DATASET_PATH)
    # filepaths = filepaths[:100]

    # choosen_phenotypes = choosen_phenotypes[:1]

    for phenotype in choosen_phenotypes:
        final_df = pd.DataFrame()
        
        start = time.time()
        for filepath in filepaths:
            filename = get_filename(filepath)
            df, data_length = data_preparation(filepath, PHENOTYPES_PATH)

            if data_length <= top_n_features:
                # Add all data from files
                temp_df = df.iloc[:, 0: data_length]
                temp_df.columns = [f"{filename}_"] * len(temp_df.columns)
                final_df = pd.concat([final_df, temp_df], axis=1)
            else:
                # This filters data base on filter models
                filtered_df = filter_data_based_on_index_file(df, phenotype, 
                                                            exp_number, INDEX_PATH, filename)
                filtered_df.columns = f"{filename}_" + filtered_df.columns
                final_df = pd.concat([final_df, filtered_df], axis=1)

        end = time.time()
        print(f"            All files loaded for phenotype: {phenotype} | time: {round(end-start, 2)} sec")
        
        start2 = time.time()
        test_df, _ = data_preparation(filepaths[0], PHENOTYPES_PATH)

        # final_df.columns = map(str, list(range(len(final_df.columns))))
        final_df = pd.concat([final_df, test_df.iloc[:, -12:]], axis=1)
        
        #Preparing data for train the model
        #Y labels
        y = np.array(final_df[phenotype])

        #X features
        features_ohe = pd.get_dummies(final_df.iloc[:, :-12])
        X = pd.DataFrame(np.array(features_ohe), columns=features_ohe.columns)

        #Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)


        #------- Random Forest ---------

        # # GridSearchCV
        # params_grid_rf = {
        #     "n_estimators":  np.arange(100, 500, 100),
        #     "criterion": ["squared_error", "absolute_error", "poisson"]
        # }

        # grid_RF = GridSearchCV(
        #                 estimator=RandomForestRegressor(),
        #                 param_grid=params_grid_rf,
        #                 scoring="neg_mean_absolute_error",
        #                 n_jobs=-1,
        #                 error_score='raise')
        

        # grid_RF = RandomForestRegressor(n_estimators=300, n_jobs=-1)

        # grid_RF.fit(X_train, y_train)
        # y_pred = grid_RF.best_estimator_.predict(X_test)
        # y_pred = grid_RF.predict(X_test)

        #------------------------------

        # params_grid_ridge = {
        #     "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]
        #     # "alpha": np.logspace(0.1, 25, 50)
        #     # "alpha": np.arange(0.1, 25, 1)
        # }

        # scoring = 'neg_mean_absolute_error'

        # grid_Ridge = GridSearchCV(estimator=Ridge(), 
        #                         param_grid=params_grid_ridge,
        #                         cv=5, 
        #                         n_jobs=-1, 
        #                         scoring=scoring)

        # grid_Ridge.fit(X_train, y_train)
        # y_pred = grid_Ridge.best_estimator_.predict(X_test)

        # print(f"score on training set: {round(grid_Ridge.best_estimator_.score(X_train, y_train), 4)}")
        # print(f"score on test set: {round(grid_Ridge.best_estimator_.score(X_test, y_test), 4)}")
        

        # scoring = 'neg_mean_absolute_error'

        # ridge_cv = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10],
        #                 #    cv=KFold(n_splits=5, shuffle=False, random_state=None), 
        #                    scoring=scoring,
        #                    store_cv_values=True).fit(X_train, y_train)

        # print(f"score on training set: {round(ridge_cv.score(X_train, y_train), 4)}")
        # print(f"score on test set: {round(ridge_cv.score(X_test, y_test), 4)}")
        
        # y_pred = ridge_cv.predict(X_test)


        #------ RidgeCV ------
        scoring = 'neg_mean_absolute_error'
        ridge_model = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], 
                            cv=KFold(n_splits=5, shuffle=False, random_state=None), 
                            scoring=scoring)

        lasso_model = LassoCV(alphas=[0.0001, 0.001,0.01, 0.1, 1, 10],
                            cv=KFold(n_splits=5, shuffle=False, random_state=None), 
                            n_jobs=-1)
                            
        ridge_cv = ridge_model.fit(X_train, y_train)
        lasso_cv = lasso_model.fit(X_train, y_train)
        y_pred = ridge_cv.predict(X_test)
        y_pred_lasso = lasso_cv.predict(X_test)

        print(f"                - Ridge score on training set: {round(ridge_cv.score(X_train, y_train), 4)}")
        print(f"                - Ridge score on test set: {round(ridge_cv.score(X_test, y_test), 4)}")
        print()
        print(f"                - Lasso score on training set: {round(lasso_cv.score(X_train, y_train), 4)}")
        print(f"                - Lasso score on test set: {round(lasso_cv.score(X_test, y_test), 4)}")



        # # Save the model
        if check_working_os():
            # joblib.dump(grid_RF.best_estimator_, f"{MODELS_BASE_PATH}/final/{exp_number}/rf_{phenotype}.joblib")
            joblib.dump(ridge_cv, f"{MODELS_BASE_PATH}/final/{exp_number}/ridge_{phenotype}.joblib")
            joblib.dump(lasso_cv, f"{MODELS_BASE_PATH}/final/{exp_number}/lasso_{phenotype}.joblib")

        else:
            joblib.dump(ridge_cv, f"{MODELS_BASE_PATH}\\final\\{exp_number}\\ridge_{phenotype}.joblib")
            joblib.dump(lasso_cv, f"{MODELS_BASE_PATH}\\final\\{exp_number}\\lasso_{phenotype}.joblib")


        # Save coef & names
        coefs = pd.DataFrame(ridge_cv.coef_, 
                        columns=["coefficients"], 
                        index=ridge_cv.feature_names_in_)

        coefs = coefs.sort_values(by="coefficients", ascending=False)
        coefs.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/csv/ridge_coef_results_{phenotype}.csv")
        
        coefs_lasso = pd.DataFrame(lasso_cv.coef_, 
                        columns=["coefficients"], 
                        index=lasso_cv.feature_names_in_)

        coefs_lasso = coefs_lasso.sort_values(by="coefficients", ascending=False)
        coefs_lasso.to_csv(f"{MODELS_BASE_PATH}/final/{exp_number}/csv/lasso_coef_results_{phenotype}.csv")
"""