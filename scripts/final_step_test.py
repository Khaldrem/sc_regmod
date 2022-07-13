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

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import matplotlib.pyplot as plt



def set_final_scoring_metrics(models_metric, model_name, phenotype, y_test, y_pred):
    models_metric["model_name"].append(model_name)
    models_metric["phenotype"].append(phenotype)

    models_metric["r2"].append(round(r2_score(y_test, y_pred), 4))
    models_metric["MAE"].append(round(mean_absolute_error(y_test, y_pred), 4))
    models_metric["MSE"].append(round(mean_squared_error(y_test, y_pred), 4))
    models_metric["RMSE"].append(round(mean_squared_error(y_test, y_pred, squared=True), 4))



files_data_df = pd.read_csv("/home/khaldrem/code/sc_regmod/dataset/models/exp_1_file_data.csv")
phenotypes = [
    "SM300-Efficiency", "SM300-Rate", "SM300-Lag", "SM300-AUC",
    "SM60-Efficiency",  "SM60-Rate",  "SM60-Lag",  "SM60-AUC",
    "Ratio-Efficiency", "Ratio-Rate", "Ratio-Lag", "Ratio-AUC",
]

phenotypes = phenotypes[:1]

models_metric = {
    "model_name": [],
    "phenotype": [],
    "r2": [],
    "MAE": [],
    "MSE": [],
    "RMSE": []
}


for phenotype in phenotypes:
    total_time = time.time()

    df = pd.read_csv(f"/home/khaldrem/code/sc_regmod/dataset/models_dataset/final/2/{phenotype}_final_data.csv")

    print(df)

    # features_ohe = pd.get_dummies(df.iloc[:, 1:(df.shape[1] - 12)])
    features_ohe = pd.get_dummies(df.iloc[:, 1:1000])
    features_list = list(features_ohe.columns)

    X = pd.DataFrame(np.array(features_ohe), columns=features_ohe.columns)
    
    y = np.array(df[phenotype])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

    models_to_train = ["Ridge", "Lasso", "RandomForestRegressor"]
    
    y_predictions = {
        "y_test": [],
        "Ridge_y_pred": [],
        "Lasso_y_pred": [],
        "RandomForestRegressor_y_pred": [],
    }

    for model in models_to_train:
        # if model == "Ridge":
        #     ridge_model = RidgeCV(alphas = np.linspace(0.001, 1, 25), 
        #                 cv=KFold(n_splits=5, shuffle=False, random_state=None), 
        #                 scoring='neg_mean_absolute_error')

        #     ridge_cv = ridge_model.fit(X_train, y_train)
        #     y_pred_ridge = ridge_cv.predict(X_test)

        #     set_final_scoring_metrics(models_metric, model, phenotype, y_test, y_pred_ridge)
        #     y_predictions[f"{model}_y_pred"] = y_pred_ridge

        #     coefs = pd.DataFrame(ridge_cv.coef_, 
        #                 columns=["coefficients"], 
        #                 index=ridge_cv.feature_names_in_)

        #     coefs = coefs.sort_values(by="coefficients", ascending=False)
        #     coefs.to_csv(f"ridge_coef_results_{phenotype}.csv")

        # if model == "Lasso":
        #     lasso_model = LassoCV(alphas = np.linspace(0.001, 1, 25),
        #                         cv=KFold(n_splits=5, shuffle=False, random_state=None), 
        #                         n_jobs=-1)
            
        #     lasso_cv = lasso_model.fit(X_train, y_train)
        #     y_pred_lasso = lasso_cv.predict(X_test)

        #     set_final_scoring_metrics(models_metric, model, phenotype, y_test, y_pred_lasso)
        #     y_predictions[f"{model}_y_pred"] = y_pred_lasso

        
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
            y_pred_rf = rf_gscv.best_estimator_.predict(X_test)

            set_final_scoring_metrics(models_metric, model, phenotype, y_test, y_pred_rf)
            y_predictions[f"{model}_y_pred"] = y_pred_rf

            # importances = list(rf_gscv.best_estimator_.feature_importances_)
            # coefs = pd.DataFrame(importances, 
            #         columns=["coefficients"], 
            #         index=features_list)

            # coefs = coefs.sort_values(by="coefficients", ascending=False)
            # coefs.to_csv(f"rf_coef_results_{phenotype}.csv")

            feats = {
                "feature": [],
                "absolute_error": [],
                "filename": []
            }
            for feature, importance in zip(features_list, rf_gscv.best_estimator_.feature_importances_):
                feats["feature"].append(feature)
                feats["absolute_error"].append(importance)
                feats["filename"].append(feature.split("_")[0])
                
                # feats[feature] = importance #add the name/value pair


            importances = pd.DataFrame.from_dict(feats, orient='columns').sort_values(by="absolute_error", ascending=False).reset_index()
            importances.to_csv(f"rf_coef_results_{phenotype}.csv")
            # importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)

           
            # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]

            # #sort
            # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse=True)
            # [print("Var: {:20} Importance: {}".format(*pair)) for pair in feature_importances];


    #Save results
#     y_predictions["y_test"] = y_test

#     y_predictions_df = pd.DataFrame.from_dict(y_predictions)
#     y_predictions_df.to_csv(f"{phenotype}_y_pred.csv")


# #Save metrics
# models_metric_df = pd.DataFrame.from_dict(models_metric)
# models_metric_df.to_csv("final_metrics_results.csv")

