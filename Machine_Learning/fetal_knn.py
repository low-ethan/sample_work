import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def get_data():
    fetal = pd.read_csv("fetal_health.csv")
    return fetal


# split into feature/target
def split_into_feature_and_target(dataset, column: str):
    target = dataset[column]#'fetal_health']
    features = dataset.drop(column, axis=1)

    # encode the target data
    le = preprocessing.LabelEncoder()
    encoded_target = pd.DataFrame(le.fit_transform(target))

    return encoded_target, features


def test_train_split(target_, features_, test_size: float, random_state: int):
    X = features_
    y = target_.values
    X_train_, X_test_, y_train_, y_test_ = model_selection.train_test_split(X, y,
                                                                        test_size=test_size,
                                                                        random_state=random_state)
    return X_train_, X_test_, y_train_, y_test_


def create_cross_validated_model(x_train, x_test, y_train_, y_test_):
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


def main():
    fetal = get_data()
    encoded_target, features = split_into_feature_and_target(fetal, 'fetal_health')
    X_train, X_test, y_train, y_test = test_train_split(encoded_target, features, 0.2, 42)
    preds = create_cross_validated_model(X_train, X_test, y_train, y_test)
    return preds


if __name__ == '__main__':
    main()
