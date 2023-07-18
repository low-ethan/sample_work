import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import fetal_knn as f_knn
import fetal_dt as f_dt
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fetal_health.csv")
print(df.head(5))
# (2126, 22)
print(df.shape)
na_vec = df.isnull().sum()
# NO MISSING VALUES
print(na_vec)
# MOSTLY NORMAL
print(df["fetal_health"].value_counts())

y = df["fetal_health"]
X = df.drop(["fetal_health"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# hidden_layer_sizes=(100)
# NOT SCALED: ACCURACY = 0.876
# SCALED: ACCURACY = 0.913

# hidden_layer_size=(100,100)
# ACCURACY = .930

# hidden_layer_size=(100,10,100)
# NOT SCALED: ACCURACY = 0.812
# SCALED: ACCURACY = 0.913

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,100), random_state=1, max_iter=20000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("MLP ACCURACY: ", accuracy_score(y_test, y_pred))

# GRID SEARCH
# mlp = MLPClassifier(max_iter=13000)
# parameters = {
#     'hidden_layer_sizes': [(50,),(150,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [1e-5, 0.05],
#     'learning_rate': ['constant', 'adaptive'],
# }
#
# clf = GridSearchCV(mlp, parameters, n_jobs=-1, cv=2)
# clf.fit(X, y)
# print("Best parameters found:\n", clf.best_params_)
# y_pred = clf.predict(X_test)
#
# print(accuracy_score(y_test, y_pred))


knn_preds = f_knn.main()
dt_preds = f_dt.main()


def ensemble(pred1, pred2, pred3):
    voted_preds = [0 for _ in range(len(pred1))]
    for i in range(len(pred1)):
        lst = [pred1[i], pred2[i], pred3[i]]
        voted_preds[i] = max(lst, key=lst.count)


    return voted_preds


voted_preds = ensemble(y_pred, knn_preds, dt_preds)
print(ensemble(y_pred, knn_preds, dt_preds))
print("accuracy_score: " + str(accuracy_score(y_test, voted_preds)))

# MLP: 0.9296
# KNN: 0.8873
# DT: 0.9272

# ENSEMBLE: 0.9366


# MLP PRED SIMILARITY WITH ENSEMBLE: 0.9883

print("SIMILARITY BETWEEN MLP PRED AND ENSEMBLE PRED: ", accuracy_score(y_pred, voted_preds))

# mod_1 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,100), random_state=1, max_iter=20000)
# mod_2 = KNNClassifier()
# mod_3 = RandomForestClassifier()
#
# final_model = VotingClassifier(
#     estimators=[('mlp', model_1), ('knn', model_2), ('rf', model_3)], voting='hard')
#
# final_model.fit(X_train, y_train)
#
# # predicting the output on the test dataset
# pred_final = final_model.predict(X_test)