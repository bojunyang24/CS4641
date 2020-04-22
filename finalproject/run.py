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
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
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
    return train_test_split(df, label, test_size=0.5)

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

def gridsearch(classifier, params, x_train, y_train, name="Test_"):
    '''
    Uses GridSearchCV to tune hyperparameters and saves the GridSearchCV results
    Trains the classifier with the best parameters and scores the model
    '''
    start_time = time.time()
    clf = GridSearchCV(classifier, params, n_jobs=-1, cv=10)
    grid = clf.fit(x_train, y_train)
    print("GridSearchCV elapsed time: {}".format(time.time() - start_time))

    # best_params = grid.best_params_
    # best_score = grid.best_score_
    # print("{}GridSearch \nBest params: {} \nScore: {}".format(name, best_params, best_score))
    
    # Saves GridSearch Result
    filename = "{}GridSearch.sav".format(name)
    pickle.dump(grid, open(filename, 'wb'))
    return grid
    # Fits with optimal hyperparameters and scores
    # best_model = RandomForestClassifier(**best_params)
    # best_model.fit(x_train, y_train)
    # print("Test Accuracy: {} Train Accuracy: {}".format(best_model.score(x_test, y_test), best_model.score(x_train, y_train)))


def non_linear_svm(data, center=True):
    x_train, x_test, y_train, y_test = preprocess_data(data)
    C = np.logspace(-2, 4, 7)
    gamma = np.logspace(-3, 3, 7)
    kernel = ['poly', 'rbf', 'sigmoid']
    params = {
        'C': C,
        'gamma': gamma,
        'kernel': kernel,
    }
    grid = gridsearch(SVC(), params, x_train, y_train, name="NonLinearSVC0_")
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = cross_val_score(clf, x_test, y_test, cv=10)

    # 9 degrees of freedome 95% two tailed CI
    # t = 2.262
    # mean = np.mean(scores)
    # se = np.std(scores)/10
    # ci = [mean + (t*se), mean - (t*se)]
    # print("95% Confidence Interval: [{}, {}]".format(ci[0], ci[1]))

    grid = gridsearch(SVC(), params, x_test, y_test, name="NonLinearSVC1_")
    res = grid.cv_results_
    clf = OneVsRestClassifier(grid.best_estimator_)
    scores = np.concatenate((scores, cross_val_score(clf, x_train, y_train, cv=10)))

    get_ci(scores)

def rfc(data, center=True):
    '''
    Uses GridSearchCV to tune hyperparameters for RandomForestClassification
    Saves the grid results to a pickle
    '''
    x_train, x_test, y_train, y_test = preprocess_data(data)
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
    max_depth = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
    max_features = ['sqrt','log2', None]
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'max_features': max_features,
    }
    grid = gridsearch(RandomForestClassifier(), params, x_train, y_train, name="RandomForest0_")
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = cross_val_score(clf, x_test, y_test, cv=10)

    grid = gridsearch(RandomForestClassifier(), params, x_train, y_train, name="RandomForest1_")
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = np.concatenate((scores, cross_val_score(clf, x_train, y_train, cv=10)))
    
    get_ci(scores)

def neuralnet(data, center=True):
    x_train, x_test, y_train, y_test = preprocess_data(data)
    hidden_layer_sizes = [(10,10,10),(20,20,20),(10,10),(20,20)]
    activation =  ['logistic', 'relu']
    alpha = [0.0001, 0.001]
    learning_rate_init = [0.01, 0.001]
    max_iter = [2000, 3000]
    params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha': alpha,
        'learning_rate_init': learning_rate_init,
        'max_iter': max_iter,
    }
    grid = gridsearch(MLPClassifier(), params, x_train, y_train, name="MLP0_")
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = cross_val_score(clf, x_test, y_test, cv=10)

    grid = gridsearch(RandomForestClassifier(), params, x_train, y_train, name="MLP1_")
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = np.concatenate((scores, cross_val_score(clf, x_train, y_train, cv=10)))

    get_ci(scores)

def get_ci(scores):
    '''
    19 degrees of freedome 95% two tailed CI
    '''
    t = 2.093
    mean = np.mean(scores)
    se = np.std(scores)/len(scores)
    ci = [mean - (t*se), mean + (t*se)]
    print("95% Confidence Interval: [{}, {}]".format(ci[0], ci[1]))



data = pd.read_csv('data/data.csv')
neuralnet(data)
# rfc(data)
# non_linear_svm(data)


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