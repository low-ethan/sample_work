
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print('test')
def getData():
    fetal = pd.read_csv("fetal_health.csv")
    # print(fetal.head())
    return fetal

def split_f_t(dataset, col: str):
    y = dataset[col]
    x = dataset.drop(col, axis = 1)

    return x, y

def trainTest(vars, respvar, test_size: float, random_state: int):
    x_train, x_test, y_train, y_test = train_test_split(vars,respvar, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

def create_model_and_preds(x_train, x_test, y_train, y_test):
    decisiont = DecisionTreeClassifier()
    decisiont = decisiont.fit(x_train, y_train)
    y_preds_test = decisiont.predict(x_test)
    print("accuracy score of decision tree: " + str(accuracy_score(y_test, y_preds_test)))
    cm = confusion_matrix(y_test,y_preds_test, labels = decisiont.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=decisiont.classes_)
    disp.plot()
    plt.show()
    return y_preds_test

def main():
    fetal = getData()
    x, y =split_f_t(fetal,'fetal_health')
    xtrain, xtest, ytrain, ytest = trainTest(x, y, .2, 42)
    preds = create_model_and_preds(xtrain, xtest, ytrain, ytest)
    
    return preds

if __name__ == '__main__':
    print('working')
    main()
    

    