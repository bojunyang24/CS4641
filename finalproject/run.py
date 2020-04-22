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
from matplotlib import pyplot as plt
import pickle
import time
import sys

def saveprint(s, filename):
    print(s)
    temp = sys.stdout
    sys.stdout = open("{}_out.txt".format(filename), "a")
    print(s)
    sys.stdout.close()
    sys.stdout = temp

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
    
    saveprint("GridSearchCV elapsed time: {}".format(time.time() - start_time), name[0:len(name)-2])

    # sys.stdout = open("non_linear_svm.txt", "a")
    # print("GridSearchCV elapsed time: {}".format(time.time() - start_time))
    # sys.stdout.close()
    # print("GridSearchCV elapsed time: {}".format(time.time() - start_time))

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
    
    outfile = "NonLinearSVC"
    saveprint(params, outfile)
    get_ci(scores, outfile)

    # sys.stdout = open("non_linear_svm.txt", "a")
    # print(params)
    # get_ci(scores)
    # sys.stdout.close()

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
    
    outfile = "RandomForest"
    saveprint(params, outfile)
    get_ci(scores, outfile)

    # sys.stdout = open("non_linear_svm.txt", "a")
    # print(params)
    # get_ci(scores)
    # sys.stdout.close()

def neuralnet(data, center=True):
    outfile = "MLP"
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
    saveprint("Best Param 1: {}".format(grid.best_params_), outfile)
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = cross_val_score(clf, x_test, y_test, cv=10)

    grid = gridsearch(MLPClassifier(), params, x_train, y_train, name="MLP1_")
    saveprint("Best Param 2: {}".format(grid.best_params_), outfile)
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = np.concatenate((scores, cross_val_score(clf, x_train, y_train, cv=10)))
    
    saveprint(params, outfile)
    get_ci(scores, outfile)

def neuralnet2(data, center=True):
    outfile = "MLP"
    x_train, x_test, y_train, y_test = preprocess_data(data)
    hidden_layer_sizes = [(100,100,100)]
    activation =  ['relu']
    alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    learning_rate_init = [0.001, 0.005, 0.01, 0.05, 0.1]
    max_iter = [1500]
    params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha': alpha,
        'learning_rate_init': learning_rate_init,
        'max_iter': max_iter,
    }
    grid = gridsearch(MLPClassifier(), params, x_train, y_train, name="MLP0_")
    saveprint("Best Param 1: {}".format(grid.best_params_), outfile)
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = cross_val_score(clf, x_test, y_test, cv=10)

    grid = gridsearch(MLPClassifier(), params, x_train, y_train, name="MLP1_")
    saveprint("Best Param 2: {}".format(grid.best_params_), outfile)
    res = grid.cv_results_
    clf = grid.best_estimator_
    scores = np.concatenate((scores, cross_val_score(clf, x_train, y_train, cv=10)))
    
    saveprint(params, outfile)
    get_ci(scores, outfile)

def get_ci(scores, outfile):
    '''
    19 degrees of freedome 95% two tailed CI
    '''
    t = 2.093
    mean = np.mean(scores)
    se = np.std(scores)/len(scores)
    ci = [mean - (t*se), mean + (t*se)]
    saveprint("95% Confidence Interval: [{}, {}]".format(ci[0], ci[1]), outfile)
    # print("95% Confidence Interval: [{}, {}]".format(ci[0], ci[1]))



data = pd.read_csv('data/data.csv')

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

# fit_linear_regression(data)

# PCA_analysis(data)

# neuralnet(data)
neuralnet2(data)

# rfc(data)

# non_linear_svm(data)

### Graph code below ###

def rfc_graphs():
    grid0 = pickle.load(open('RandomForest0_GridSearch.sav', 'rb'))
    res0 = grid0.cv_results_
    grid1 = pickle.load(open('RandomForest0_GridSearch.sav', 'rb'))
    res1 = grid1.cv_results_
    x = [param['max_depth'] for param in res0['params']]
    y = res0['mean_test_score']
    get_rfc_graphs(x,y, res0['params'], None, 'sqrt', 'log2', "plots/RF_max_depth", "max_depth")
    x = [param['n_estimators'] for param in res0['params']]
    get_rfc_graphs(x,y, res0['params'], None, 'sqrt', 'log2', "plots/RF_n_estimators", "n_estimators")

def get_rfc_graphs(x, y, params, c1, c2, c3, name, xname, yname='mean_test_score'):
    redx = []
    redy = []
    greenx = []
    greeny = []
    bluex = []
    bluey = []
    i = 0
    for param in params:
        if param['max_features'] == c1:
            redx.append(x[i])
            redy.append(y[i])
        if param['max_features'] == c2:
            greenx.append(x[i])
            greeny.append(y[i])
        if param['max_features'] == c3:
            bluex.append(x[i])
            bluey.append(y[i])
        i+=1
    plt.scatter(redx,redy,color='red',label='None')
    plt.scatter(greenx,greeny,color='green',label='sqrt')
    plt.scatter(bluex,bluey,color='blue',label='log2')
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('{}0.png'.format(name))
    plt.clf()
    
    plt.scatter(redx,redy,color='red',label='None')
    plt.scatter(greenx,greeny,color='green',label='sqrt')
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('{}1.png'.format(name))
    plt.clf()

    plt.scatter(greenx,greeny,color='green',label='sqrt')
    plt.scatter(bluex,bluey,color='blue',label='log2')
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('{}2.png'.format(name))
    plt.clf()

    plt.scatter(redx,redy,color='red',label='None')
    plt.scatter(bluex,bluey,color='blue',label='log2')
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('{}3.png'.format(name))
    plt.clf()

    uniquex = np.unique(np.array(x))

    redline = make_line(redx, redy, uniquex)
    greenline = make_line(greenx, greeny, uniquex)
    blueline = make_line(bluex, bluey, uniquex)
    plt.semilogx(redline[:,0], redline[:,1], label=str(c1), color="red")
    plt.semilogx(greenline[:,0], greenline[:,1], label=str(c2), color="green")
    plt.semilogx(blueline[:,0], blueline[:,1], label=str(c3), color="blue")
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('{}_line.png'.format(name))
    plt.clf()


def svm_graphs():
    grid0 = pickle.load(open('NonLinearSVC0_GridSearch.sav', 'rb'))
    res0 = grid0.cv_results_
    grid1 = pickle.load(open('NonLinearSVC1_GridSearch.sav', 'rb'))
    res1 = grid1.cv_results_
    x = [param['C'] for param in res0['params']]
    y = res0['mean_test_score']
    get_svm_graphs(x,y, res0['params'], 'poly', 'rbf', 'sigmoid', "plots/SVM_C", "C")
    x = [param['gamma'] for param in res0['params']]
    get_svm_graphs(x,y, res0['params'], 'poly', 'rbf', 'sigmoid', "plots/SVM_gamma", "gamma")

def get_svm_graphs(x, y, params, c1, c2, c3, name, xname, yname='mean_test_score'):
    redx = []
    redy = []
    greenx = []
    greeny = []
    bluex = []
    bluey = []
    i = 0
    for param in params:
        if param['kernel'] == c1:
            redx.append(x[i])
            redy.append(y[i])
        if param['kernel'] == c2:
            greenx.append(x[i])
            greeny.append(y[i])
        if param['kernel'] == c3:
            bluex.append(x[i])
            bluey.append(y[i])
        i+=1
    plt.scatter(redx,redy,color='red',label=c1)
    plt.scatter(greenx,greeny,color='green',label=c2)
    plt.scatter(bluex,bluey,color='blue',label=c3)
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    if xname=='gamma':
        plt.xlim(0.0001, 10000)
    plt.xscale('log')
    plt.savefig('{}0.png'.format(name))
    plt.clf()
    
    plt.scatter(redx,redy,color='red',label=c1)
    plt.scatter(greenx,greeny,color='green',label=c2)
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    if xname=='gamma':
        plt.xlim(0.0001, 10000)
    plt.xscale('log')
    plt.savefig('{}1.png'.format(name))
    plt.clf()

    plt.scatter(greenx,greeny,color='green',label=c2)
    plt.scatter(bluex,bluey,color='blue',label=c3)
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    if xname=='gamma':
        plt.xlim(0.0001, 10000)
    plt.xscale('log')
    plt.savefig('{}2.png'.format(name))
    plt.clf()

    plt.scatter(redx,redy,color='red',label=c1)
    plt.scatter(bluex,bluey,color='blue',label=c3)
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    if xname=='gamma':
        plt.xlim(0.0001, 10000)
    plt.xscale('log')
    plt.savefig('{}3.png'.format(name))
    plt.clf()

    uniquex = np.unique(np.array(x))

    redline = make_line(redx, redy, uniquex)
    greenline = make_line(greenx, greeny, uniquex)
    blueline = make_line(bluex, bluey, uniquex)
    plt.semilogx(redline[:,0], redline[:,1], label=str(c1), color="red")
    plt.semilogx(greenline[:,0], greenline[:,1], label=str(c2), color="green")
    plt.semilogx(blueline[:,0], blueline[:,1], label=str(c3), color="blue")
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('{}_line.png'.format(name))
    plt.clf()

def make_line(x,y,xvals):
    xy = np.concatenate((np.reshape(x, (len(x),1)), np.reshape(y, (len(y),1))), axis=1)
    line = []
    for gam in xvals:
        targets = np.where(xy[:,0] == gam)
        line.append([gam, np.average(xy[targets,1])])
    return np.array(line)

def nn_graphs():
    grid0 = pickle.load(open('MLP0_GridSearch.sav', 'rb'))
    res0 = grid0.cv_results_
    grid1 = pickle.load(open('MLP1_GridSearch.sav', 'rb'))
    res1 = grid1.cv_results_
    y = res0['mean_test_score']
    x = [param['alpha'] for param in res0['params']]
    get_nn_graphs(x,y, res0['params'], 'logistic', 'relu', "plots/MLP_alpha", "alpha")
    x = [param['learning_rate_init'] for param in res0['params']]
    get_nn_graphs(x,y, res0['params'], 'logistic', 'relu', "plots/MLP_learning_rate_init", "learning_rate_init")
    x = [param['max_iter'] for param in res0['params']]
    get_nn_graphs(x,y, res0['params'], 'logistic', 'relu', "plots/MLP_max_iter", "max_iter")

def get_nn_graphs(x, y, params, c1, c2, name, xname, yname='mean_test_score', weirdx=False):
    redx = []
    redy = []
    greenx = []
    greeny = []
    bluex = []
    bluey = []
    i = 0
    for param in params:
        if param['activation'] == c1:
            if weirdx:
                redx.append(x[i][0])
            else:
                redx.append(x[i])
            redy.append(y[i])
        if param['activation'] == c2:
            if weirdx:
                greenx.append(x[i][0])
            else:
                greenx.append(x[i])
            greeny.append(y[i])
        i+=1

    uniquex = np.unique(np.array(x))

    redline = make_line(redx, redy, uniquex)
    greenline = make_line(greenx, greeny, uniquex)
    plt.semilogx(redline[:,0], redline[:,1], label=str(c1), color="red")
    plt.semilogx(greenline[:,0], greenline[:,1], label=str(c2), color="green")
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    # plt.show()
    plt.savefig('{}_line.png'.format(name))
    plt.clf()

# rfc_graphs()

# svm_graphs()

# nn_graphs()