import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from dtaidistance import dtw
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def load_dataset(path:str, squeeze=False):
    with open(path, 'rb') as f:
        d = np.load(f)
        X_train = d['X_train']
        y_train = d['y_train']
        X_test = d['X_test']
        y_test = d['y_test']
        enc_vect = d['enc_dict']
        enc_dict = {i:class_ for class_ ,i in zip(enc_vect, range(enc_vect.shape[0]))}

    if squeeze:
        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)

    return (X_train,y_train,X_test,y_test,enc_dict)

def amplitude_scaling(data, labels):
    X = np.squeeze(data)
    X = np.asarray(pd.DataFrame(X).apply(zscore, axis=1))

    for i in range(X.shape[0]):
        if np.isnan(X[i,:]).any():
            nan_index = i
            print(f'NaN index: {i}')
            X = np.delete(X,nan_index,0)
            labels = np.delete(labels,nan_index,0)
    return X,labels


def save_distance_matrix(X, path:str):
    
    euclidean = pairwise_distances(X, metric='euclidean')
    print('euclidean calcolata')

    ds = dtw.distance_matrix_fast(X, use_pruning=True)
    ds[ds==np.inf] = 0
    ds += ds.T

    np.savez(path, euclidean=euclidean, dtw=ds)


def dtw_dist(x,y):
    return dtw.distance_fast(x,y,use_pruning=True)

def decode_y(y_enc, enc_dict):
    return [enc_dict[y] for y in y_enc]



def top_flop_f1(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    l = [(key, val['f1-score']) for key,val in report.items() if (isinstance(val, dict) and key not in ['macro avg', 'weighted avg'])]
    l.sort(key= lambda x: x[1], reverse=True)
    top5 = l[:5]
    l.sort(key= lambda x: x[1], reverse=False)
    flop5 = l[:5]

    print('TOP 5 F1-SCORE:')
    for top in top5:
        print(f'{top[0]}: {top[1]}')

    print('\n\nFLOP 5 F1-SCORE:')
    for flop in flop5:
        print(f'{flop[0]}: {flop[1]}')

def load_csv_data(path:str):
    X_train = pd.read_csv(f'{path}/X_train.csv')
    X_test = pd.read_csv(f'{path}/X_test.csv')
    y_train = np.squeeze(np.array(pd.read_csv(f'{path}/y_train.csv')))
    y_test = np.squeeze(np.array(pd.read_csv(f'{path}/y_test.csv')))

    return (X_train,X_test,y_train,y_test)


def plot_pca(X_pca, y_train):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.prism, edgecolor='k', alpha=0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()
