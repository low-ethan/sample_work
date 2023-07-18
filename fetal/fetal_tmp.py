import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import fetal_functions as f_fn
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
# import streamlit as st






def main():
    fetal = f_fn.getData()
    X, y = f_fn.split_f_t(fetal, 'fetal_health')
    X = f_fn.scale(X)
    X = f_fn.power_transform(X)
    X_train, X_test, y_train, y_test = f_fn.trainTest(X, y, 0.2, 43)

    # preprocessing and EDA
    # f_fn.correlation_matrix(fetal)
    # f_fn.distribution(fetal, 'fetal_health')



    # # A few Algorithms
    f_fn.kmeans_elbow(X)
    km_preds = f_fn.kmeans_model(X_train, X_test, y_test)
    knn_preds = f_fn.knn_model(X_train, X_test, y_train, y_test)
    dt_preds = f_fn.dt_model(X_train, X_test, y_train, y_test)  # TODO: PREVENT OVER-FITTING
    mlp_preds = f_fn.mlp_model(X_train, X_test, y_train, y_test)

    f_fn.confusionMatrix(y_test, km_preds, "km")
    f_fn.confusionMatrix(y_test, knn_preds, "knn")
    f_fn.confusionMatrix(y_test, dt_preds, "dt")
    f_fn.confusionMatrix(y_test, mlp_preds, "mlp")


    # Bagging, boosting
    voted_preds = f_fn.ensemble(mlp_preds, knn_preds, dt_preds)
    print("Voting accuracy_score: " + str(accuracy_score(y_test, voted_preds)))

    f_fn.confusionMatrix(y_test, voted_preds, "voted")


    return 1


if __name__ == '__main__':
    main()

