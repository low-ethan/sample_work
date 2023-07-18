import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import seaborn as sns



def getData():
    fetal = pd.read_csv("fetal_health.csv")
    # print(fetal.head())
    return fetal

def split_f_t(dataset, col: str):
    y = dataset[col]
    x = dataset.drop(col, axis = 1)
    return x, y

def scale(X):
    X = MinMaxScaler().fit_transform(X)
    # y = MinMaxScaler().fit_transform(np.array(y).reshape(-1,1))
    return X


def remove_outliers(data, y_name):
    z = np.abs(stats.zscore(data[y_name]))
    d_c = data[(z<3)]
    print('REMOVING OUTLIERS of nonViol, 3 std_devs')
    print(data.shape, d_c.shape)
    return d_c


def correlation_matrix(data):
    sns.set(rc={"figure.figsize": (15, 15)})
    corr_mat = data.corr().round(3)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask).figure.savefig("heat_map_1.png")
    # corr_mat = corr_mat.unstack()
    # high_corr = corr_mat[abs(corr_mat) > 0.7]
    # high_corr = high_corr[1 > high_corr]
    # print("HIGH CORRELATION")
    # print(high_corr)
    # f = open("high_corr.txt", x)
    # f.write(high_corr)
    # f.close()
    # plt.show()


def power_transform(X):
    pt = preprocessing.PowerTransformer()
    X = pt.fit_transform(X)
    # y = pt.fit_transform(np.array(y).reshape(-1,1))
    return X

def distribution(df, target):
    figure, axes = plt.subplots(nrows=6, ncols=4, figsize =(15,20))
    row = 0
    col = 0
    for feature in df.columns:
        df[feature].value_counts().plot(ax=axes[row, col], kind='bar', xlabel=feature, rot=0)
        col = col+1 if col < 3 else 0
        row = row + 1 if col == 0 else row
    plt.tight_layout()
    plt.savefig('count_chart.png')
    # plt.show()

    # show again based on correlation
    y = df[target]
    X = df.drop([target], axis=1)

    df1 = df[df[target] == 1]
    df2 = df[df[target] == 2]
    df3 = df[df[target] == 3]
    figure, axes = plt.subplots(nrows=len(df.columns)-1, ncols=3, figsize=(15,40))
    row = 0
    for feature in df.columns:
        if feature == target:
            continue
        df1[feature].value_counts().plot(ax=axes[row, 0], kind='bar', xlabel=feature+" 1", rot=0)
        df2[feature].value_counts().plot(ax=axes[row, 1], kind='bar', xlabel=feature+" 2", rot=0)
        df3[feature].value_counts().plot(ax=axes[row, 2], kind='bar', xlabel=feature+" 3", rot=0)
        row += 1
    plt.tight_layout()
    plt.savefig('distribution.png')

def kmeans_elbow(X):
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
def kmeans_model(X_train, X_test, y_test):
    km = KMeans(
        n_clusters=3, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X_train)
    y_km = km.predict(X_test)

    distr = [[0,0,0],[0,0,0],[0,0,0]]

    tmp = list(y_test)
    for i in range(len(y_km)):
        distr[y_km[i]][int(tmp[i])-1] += 1

    clusterID = [id.index(max(id)) for id in distr]

    y_km = [clusterID[i]+1 for i in y_km]
    return y_km

def trainTest(vars, respvar, test_size: float, random_state: int):
    x_train, x_test, y_train, y_test = train_test_split(vars,respvar, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def dt_model(x_train, x_test, y_train, y_test):
    decisiont = DecisionTreeClassifier()
    decisiont = decisiont.fit(x_train, y_train)
    y_preds_test = decisiont.predict(x_test)
    print("accuracy score of decision tree: " + str(accuracy_score(y_test, y_preds_test)))
    return y_preds_test

def knn_model(x_train, x_test, y_train_, y_test_):
    classifier = neighbors.KNeighborsClassifier()
    k_range = list(range(1, 35))
    param_grid = dict(n_neighbors=k_range)

    # defining parameter range
    grid = GridSearchCV(classifier, param_grid,
                        cv=10, scoring='accuracy', return_train_score=False, verbose=1)

    # fitting the model for grid search
    grid_search = grid.fit(x_train, y_train_.ravel())
    best_neighbor_number = grid_search.best_params_['n_neighbors']

    # create a new classifier with the optimized parameters
    new_classifier = neighbors.KNeighborsClassifier(n_neighbors=best_neighbor_number)
    new_classifier.fit(x_train, y_train_.ravel())

    # test to see how good it is
    gridsearch_preds = new_classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test_, gridsearch_preds)

    print(f'KNN Model Accuracy: {accuracy}')

    return gridsearch_preds

def mlp_model(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=1, max_iter=20000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("MLP ACCURACY: ", accuracy_score(y_test, y_pred))
    return y_pred


def ensemble(pred1, pred2, pred3):
    voted_preds = [0 for _ in range(len(pred1))]
    for i in range(len(pred1)):
        lst = [pred1[i], pred2[i], pred3[i]]
        voted_preds[i] = max(lst, key=lst.count)
    return voted_preds

def confusionMatrix(y_real, y_pred, title):
    # cm = confusion_matrix(y_real,y_pred, labels = decisiont.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=decisiont.classes_)
    cm = confusion_matrix(y_real, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()