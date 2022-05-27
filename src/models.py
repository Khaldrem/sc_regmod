
import numpy as np
import pandas as pd
from itertools import product

from sklearn import linear_model
from src.anova import order_phenotypes_by_files_id
from src.io import read_phylip_file, read_phenotypes_file, get_filepaths, write_pandas_csv
import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedKFold, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR

from src.utils import check_directory, check_working_os, get_filename

import plotly.express as px

import joblib
import random

def data_preparation(filepath, phenotypes_path):

    phenotypes_df = read_phenotypes_file(phenotypes_path)
    phenotypes_df = order_phenotypes_by_files_id(filepath, phenotypes_df)
    data = read_phylip_file(filepath)

    cols_data = {}
    for col in range(data.get_alignment_length()):
        key_name = "x" + str(col)
        cols_data[key_name] = list(data[:, col])

    df = pd.DataFrame.from_dict(cols_data, orient="columns")
   
    #Reseteo el indice de phenotypes_df
    phenotypes_df = phenotypes_df.reset_index()
    phenotypes_df.drop('index', axis = 1, inplace=True)

    #Obtengo las llaves
    df_keys = df.keys().tolist() + phenotypes_df.keys().tolist()

    df = pd.concat([df, phenotypes_df], axis='columns', ignore_index=True)
    df.columns = df_keys

    #drop columnas innecesarias
    df = df.drop(columns=["Standard", "Haploide-Diploide", "Ecological info", "Standard_num"])

    return df, data.get_alignment_length()


def train_best_model(best_models, index):
    pass


def filter_model_step(choosen_phenotypes=[], DATASET_PATH="", PHENOTYPES_PATH=""):
    import matplotlib.pyplot as plt
    
    filepaths = get_filepaths(DATASET_PATH)
    scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
                   'neg_mean_squared_error', 'neg_root_mean_squared_error','r2']

    
    filepaths = ['/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YPL085W.phylip']

    for (filepath, phenotype) in product(filepaths, choosen_phenotypes):
        filename = get_filename(filepath)
        df, data_length = data_preparation(filepath, PHENOTYPES_PATH)
        print(f"File: {filename} | alignment_length: {data_length}")


        #Y labels
        labels = np.array(df[phenotype])

        #X features
        features_ohe = pd.get_dummies(df.iloc[:, 0:data_length])
        features_list = list(features_ohe.columns)
        features = np.array(features_ohe)

        #Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20, shuffle=False)

        # Validación empleando el Out-of-Bag error
        # ==============================================================================
        train_scores = []
        oob_scores   = []

        # Valores evaluados
        estimator_range = range(1, 250, 5)

        # Bucle para entrenar un modelo con cada valor de n_estimators y extraer su error
        # de entrenamiento y de Out-of-Bag.
        for n_estimators in estimator_range:
            modelo = RandomForestRegressor(
                        n_estimators = n_estimators,
                        criterion    = 'squared_error',
                        max_depth    = None,
                        bootstrap    = True,
                        oob_score    = True,
                        n_jobs       = -1,
                        random_state = 123
                    )
            modelo.fit(X_train, y_train)
            train_scores.append(modelo.score(X_train, y_train))
            oob_scores.append(modelo.oob_score_)


        asdf = pd.DataFrame([train_scores, oob_scores], index=['train scores', 'oob scores'], columns=estimator_range)

        fig = px.line(asdf.T)
        fig.show()
        print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(oob_scores)]}")


        # Gráfico con la evolución de los errores
        # fig, ax = plt.subplots(figsize=(6, 3.84))
        # ax.plot(estimator_range, train_scores, label="train scores")
        # ax.plot(estimator_range, oob_scores, label="out-of-bag scores")
        # ax.plot(estimator_range[np.argmax(oob_scores)], max(oob_scores),
        #         marker='o', color = "red", label="max score")
        # ax.set_ylabel("R^2")
        # ax.set_xlabel("n_estimators")
        # ax.set_title("Evolución del out-of-bag-error vs número árboles")
        # # plt.legend();
        # plt.show()



        # if data_length > 10:

        #     #Y labels
        #     labels = np.array(df[phenotype])

        #     #X features
        #     features_ohe = pd.get_dummies(df.iloc[:, 0:data_length])
        #     features_list = list(features_ohe.columns)
        #     features = np.array(features_ohe)

        #     #Split Train/Test
        #     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20, shuffle=False)

        #     #Set model
        #     model = RandomForestRegressor()

        #     #Cross validation
        #     kfold = KFold(n_splits=5, shuffle=False)
        #     cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, error_score='raise', n_jobs=-1, return_train_score=True, return_estimator=True)

        #     print(cv_results)
        
        # else:
        #     print(f"        File: {filename} tiene menos de 10 features.")

        # break



def automated_model_train(choosen_phenotypes=[], DATASET_PATH="", MODELS_PATH="", PHENOTYPES_PATH=""):

    #Set CSV PATH
    CSV_PATHS = ""
    if check_working_os():
        CSV_PATHS = MODELS_PATH + "/csv/"
    else:
        CSV_PATHS = MODELS_PATH + "\\csv\\"

    check_directory(CSV_PATHS)


    # filepaths = get_filepaths(DATASET_PATH)
    filepaths = ['/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YER155C.phylip']
    #               '/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YIL169C.phylip']
    #             #  '/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YLR106C.phylip',
    #             #  '/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YBR289W.phylip',
    #             #  '/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YPL085W.phylip'] #BORRAR

    # filepaths = random.sample(filepaths, 20)
    model_freq = []
    
    start_overall = time.time()

    idx = 0
    for item in product(filepaths, choosen_phenotypes):
        filename = get_filename(item[0])
        print(f"======================================================")
        df, data_length = data_preparation(item[0], PHENOTYPES_PATH)
        # print(item[0])
        print(f"File: {filename} | alignment_length: {data_length} | {idx}/{len(filepaths)}")
        idx = idx + 1
        
        #Y labels
        labels = np.array(df[item[1]])

        #X features
        features_ohe = pd.get_dummies(df.iloc[:, 0:data_length])
        features_list = list(features_ohe.columns)
        features = np.array(features_ohe)

        #Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20, shuffle=False)
        
        #Models
        models = [
            ('LinLasso', Lasso()),
            ('LinRidge', Ridge()),
            ('RF', RandomForestRegressor()),
            ('SVR', SVR(max_iter=1000))
        ]

        dfs = []
        scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
                   'neg_mean_squared_error', 'neg_root_mean_squared_error','r2']

        CV_SPLITS = 5

        start = time.time()
        for name, model in models:
            kfold = KFold(n_splits=CV_SPLITS, shuffle=False)
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, error_score='raise', n_jobs=-1)
            this_df = pd.DataFrame(cv_results)
            this_df['model_name'] = name
            dfs.append(this_df)
        

        #Escribimos los resultados de la validacion cruzada
        final = pd.concat(dfs, ignore_index=True)
        final.to_csv(CSV_PATHS + filename + ".csv")

        models_best_values = []
        for name, model in models:
            value = final.loc[final['model_name'] == name, 'test_neg_mean_absolute_error'].max()
            print(f"    Model: {name} Best MAE: {value}")
            models_best_values.append({"model_name": name, "value": value, "model": model})


        print()
        best_index = max(range(len(models_best_values)), key=lambda index: models_best_values[index]['value'])
        print(f"Overall best model: {models_best_values[best_index]['model_name']} with MAE: {models_best_values[best_index]['value']}")
        model_freq.append(models_best_values[best_index]['model_name'])
    


    

        """

        params_grid = {}
        #Seteamos los parametros a buscar
        if models_best_values[best_index]['model_name'] == "LinRidge":
            #From docs: alpha must be a non-negative float i.e. in [0, inf).
            params_grid["alpha"] = np.arange(0, 10, 0.1)
            
        if models_best_values[best_index]['model_name'] == "RF":
            params_grid["n_estimators"] = np.arange(100, 500, 100) #Camiar!
            params_grid["criterion"] = ["squared_error", "absolute_error", "poisson"]
            params_grid["n_jobs"] = -1
                    
        if models_best_values[best_index]['model_name'] == "SVR":
            # params_grid["kernel"] = ["linear", "poly", "rbf"]
            params_grid["kernel"] = ["linear"]
            # params_grid["degree"] = [3, 4, 5, 6, 7, 8] #Solo para kernel poly
            # params_grid["gamma"] = ["scale", "auto"]
            # params_grid["C"] = np.arange(0.0001, 1, 0.002, dtype='float')
            params_grid["C"] = [0.001, 0.01, 0.1, 1]
            # params_grid["epsilon"] = np.arange(0.0001, 1, 0.002, dtype='float')
            params_grid["epsilon"] = [0.0001, 0.001, 0.01, 0.1]
        
        grid_model = GridSearchCV(estimator=models_best_values[best_index]["model"],
                                param_grid=params_grid,
                                scoring="neg_mean_absolute_error",
                                n_jobs=-1,
                                error_score='raise')

        grid_model.fit(X_train, y_train)
        y_pred = grid_model.best_estimator_.predict(X_test)

        print()
        print(f"Model: {models_best_values[best_index]['model_name']}")
        print(f"best params: {grid_model.best_params_}")
        print(f"Accuracy: {round(1-mean_absolute_percentage_error(y_test, y_pred), 4)}")

        #Save model
        if check_working_os():
            joblib.dump(grid_model.best_estimator_, f"{MODELS_PATH}/{filename}.joblib")
        else:
            joblib.dump(grid_model.best_estimator_, f"{MODELS_PATH}\\{filename}.joblib")


        #Update index file, insert best N features
        if models_best_values[best_index]['model_name'] == "LinRidge":
            max_values_indexes = (-grid_model.best_estimator_.coef_).argsort()[:50]
            for item in max_values_indexes:
                print(f"{features_list[item]} {grid_model.best_estimator_.coef_[item]}") 

        if models_best_values[best_index]['model_name'] == "SVR":
            max_values_indexes = (-grid_model.best_estimator_.coef_[0]).argsort()[:50]
            for item in max_values_indexes:
                print(f"{features_list[item]} {grid_model.best_estimator_.coef_[0][item]}")

        if models_best_values[best_index]['model_name'] == "RF":
            importances = list(grid_model.best_estimator_.feature_importances_)
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
            
            #sort
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse=True)
            [print("Var: {:20} Importance: {}".format(*pair)) for pair in feature_importances];

        """

        end = time.time()
        print(f"It took: {round((end-start)/60, 2)} min.")
    
    set_model_freq = set(model_freq)
    for key in set_model_freq:
        print(f"key: {key} | {model_freq.count(key)}")
    
    end_overall = time.time()
    print(f"All: {round((end_overall-start_overall)/60, 2)} min.")
        
    
"""
Dudas:
    # Consultar por el % de train/test (ahora esta en 75% train 25% test)
    # Consultar por la opcion shuffle y el random state del cross_validate
    # 
    Hay que definir un criterio respecto a que archivos consultar, porque hay 
    algunos que tienen pocas columnas. Ademas que pasa en los casos en que 
    por ejem. el largo de columnas es de 20 y nosotros consideramos extraer las top 50

Pasos:

    1. Cargar datos
    2. Transformar los datos a OHE
    3. Cross Validation entre 4 Modelos:
        3.1 Regresion lineal (funciona muy mal)
        3.1 Ridge Regression
        3.1 Random Forest
        3.1 SVR (es la que mejor funciona)
    
    4. Identificar el mejor modelo de la validacion cruzada
        4.1 Guardar el csv generado
    5. Realizar una busqueda de los mejores parametros para el modelo
        5.1 Aqui demora mucho mucho, debido a la cantidad practicamente infinita que se puede generar sobretodo para SVR
    6. Identificar las columnas de mayor importancia
    7. Guardar columnas de mayor importancia en el indice
    8. Guardar modelo        

"""