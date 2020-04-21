import numpy as np
import pandas as pd
import os.path, matplotlib
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

def preprocess_data(data, center=True):
    '''
    centers and splits the data for testing and training
    returns x_train, x_test, y_train, y_test
    '''
    label = data.label
    df = data.drop(['label','filename'], axis=1)
    if center:
        scaler = StandardScaler()
        scaler.fit(df)
        df = scaler.transform(df)
    return train_test_split(df, label, test_size=0.2)

def train_linear_svm(x_train, x_test, y_train, y_test, kernel="linear"):
    '''
    fits using linear svm
    returns linear svm model, test score, train score
    '''
    # linear kernel svc
    # linsvc = SVC(kernel=kernel)
    linsvc = OneVsRestClassifier(LinearSVC(max_iter=10000)) # OVR
    start_time = time.time()
    linsvc_model = linsvc.fit(x_train, y_train)
    print("elapsed: {}".format(time.time() - start_time))
    linsvc_pred = linsvc_model.predict(x_test)
    test_score = linsvc_model.score(x_test, y_test)
    print("test accuracy: {}".format(test_score))
    linsvc_pred = linsvc_model.predict(x_train)
    train_score = linsvc_model.score(x_train, y_train)
    print("train accuracy: {}".format(train_score))
    return linsvc_model, test_score, train_score

def fit_linear_regression(data, center=True):
    '''
    one hot encodes the labels and fits the data using linear regression
    returns linear regression model
    '''
    encoded_labels = pd.get_dummies(data.label)
    data = pd.concat([data, encoded_labels],axis=1)
    data = data.drop(['label','filename'], axis=1)
    df = data.drop(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'], axis=1)
    if center:
        scaler = StandardScaler()
        scaler.fit(df)
        df = scaler.transform(df)
    x_train, x_test, y_train, y_test = train_test_split(df, encoded_labels, test_size=0.1)
    # linear model
    linreg = linear_model.LinearRegression()
    start_time = time.time()
    linreg_model = linreg.fit(x_train, y_train)
    print("elapsed: {}".format(time.time() - start_time))
    linreg_pred = linreg_model.predict(x_test)
    print("test accuracy: {}".format(linreg_model.score(x_test, y_test)))
    linreg_pred = linreg_model.predict(x_train)
    print("test accuracy: {}".format(linreg_model.score(x_train, y_train)))
    return linreg_model

def PCA_analysis(data, center=True):
    '''
    plots 3D plot of data
    '''
    label = data.label
    df = data.drop(['label','filename'], axis=1)
    if center:
        scaler = StandardScaler()
        scaler.fit(df)
        df = scaler.transform(df)
    pca = PCA(n_components=3)
    pca.fit(df)
    temp = pca.transform(df)
    _,pcalabels = np.unique(label,return_inverse=True)
    plot_3D_proj(data=temp, labels=pcalabels)

def plot_3D_proj(data=None,labels=None):
    '''
    Creates a 3D visualization of the data, returning the created figure. See
    the main() function for example usage.
    '''
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111,projection='3d')
    proj_3d_data = data
    ax.scatter(proj_3d_data[:,0],proj_3d_data[:,1],proj_3d_data[:,2],c=labels)
    fig.gca().set_title('PCA 3D transformation')
    matplotlib.pyplot.show()
    # return fig

def gridsearch(classifier, params, x_train, y_train):
    clf = GridSearchCV(classifier, params)
    return clf.fit(x_train, y_train)

def rfc(data, center=True):
    '''
    Uses GridSearchCV to tune hyperparameters for RandomForestClassification
    Saves the grid results to a pickle
    '''
    x_train, x_test, y_train, y_test = preprocess_data(data)
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_depth = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
    max_features = ['sqrt','log2', None]
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'max_features': max_features,
    }
    grid = gridsearch(RandomForestClassifier(), params, x_train, y_train)
    best_params = grid.best_params_
    best_score = grid.best_score_
    print("GridSearch \nBest params: {} \nScore: {}".format(best_params, best_score))
    
    filename = "GridSearchRFC.sav"
    pickle.dump(grid, open(filename, 'wb'))
    
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(x_train, y_train)
    print("Test Accuracy: {} Train Accuracy: {}".format(best_model.score(x_test, y_test), best_model.score(x_train, y_train)))

data = pd.read_csv('data/data.csv')

rfc(data)

# PCA_analysis(data)

# fit_linear_regression(data)

# average test and train score for linear svm
if False:
    test = 0
    train = 0
    size = 100
    for i in range(size):
        x_train, x_test, y_train, y_test = preprocess_data(data)
        _, test_score, train_score = train_linear_svm(x_train, x_test, y_train, y_test,'rbf')
        test+=test_score
        train+=train_score
    print("Avg Test Accuracy: {} Avg Train Accuracy: {}".format(test/size, train/size))