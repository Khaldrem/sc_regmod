#Imports
import os, sys
path_to_package = os.path.abspath(os.path.join('../'))
if path_to_package not in sys.path:
    sys.path.append(path_to_package)

import numpy as np
import pandas as pd
from src.io import get_filepaths
from src.utils import get_anova_filepaths, check_working_os, get_filename
from src.models import data_preparation, automated_model_train

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px

# PATH DEFINITIONS
if check_working_os():
    ANOVA_DATASET_PATH = "/home/khaldrem/code/sc_regmod/dataset/anova"
    INDEX_PATH = "/home/khaldrem/code/sc_regmod/dataset/index"
    PHENOTYPES_PATH = "/home/khaldrem/code/sc_regmod/dataset/phenotypes/clean_phenotypes.csv"

else:
    CLEAN_DATASET_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\clean"
    ANOVA_DATASET_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\anova"
    INDEX_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\index"
    PHENOTYPES_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\phenotypes\\clean_phenotypes.csv"



filepath = '/home/khaldrem/code/sc_regmod/dataset/anova/anova_at_least_one_phenotype/p_value_0_05/all/YDR543C.phylip'
filename = get_filename(filepath)
df, data_length = data_preparation(filepath, PHENOTYPES_PATH)

print(f"File: {filename} | alignment_length: {data_length}")

choosen_phenotype = 'SM300-Efficiency'

#Y labels
labels = np.array(df[choosen_phenotype])

#X features
features_ohe = pd.get_dummies(df.iloc[:, 0:data_length])
features_list = list(features_ohe.columns)
print(f"features_list: {len(features_list)}")
features = np.array(features_ohe)

#Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state=42, shuffle=False)


alphas = np.logspace(-10, -2, 100)

res = []
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    temp = {
        "alpha": a,
        "accuracy": round(1-mean_absolute_percentage_error(y_test, y_pred), 6)
    }
    # print(f"alpha: {a} - accuracy: {round(1-mean_absolute_percentage_error(y_test, y_pred), 4)}")
    res.append(temp)

res_sorted = sorted(res, key=lambda d: d['accuracy'], reverse=True) 

for i in range(10):
    print(f"alpha: {res_sorted[i]['alpha']} - accuracy: {res_sorted[i]['accuracy']}")



#FIGURA Features vs Coef
# fig = px.scatter(x=features_list, y=model.coef_, title=f"model: RidgeRegression | file: {filename} | length: {data_length}")
# fig.show()

#Encuentro los 50 elementos con los coef mas altos
# max_values_indexes = (-model.coef_).argsort()[:50]

# for item in max_values_indexes:
#     print(f"{features_list[item]} {model.coef_[item]}")


#Ridge info
# print(f"coef_: {model.coef_}")
# print(f"interpect_: {model.intercept_}")
# print(f"n_iter_: {model.n_iter_}")
# print(f"n_features_in_: {model.n_features_in_}")
# print(f"score: {model.score(X_test, y_test)}")
# print(f"params: {model.get_params()}")


#TODOS!
""" 
    Dudas: Que coeficientes se eligen: los mas altos + o - ?

    Encontrar coef mas importantes
    Probar diferentes alphas
    obtener las metricas 
"""