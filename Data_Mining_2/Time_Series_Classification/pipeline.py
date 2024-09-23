import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE


X = np.random.rand(200,5)
y = np.array([random.randint(0,2) for i in range(200)])
order = [0,2,1]

class Pipeline:
    def __init__(self,order) -> None:
        self.X = None
        self.y = None
        self.models = []
        self.order = order
        self.predictions = None

    def fit(self,X,y, resampling_ratio=1):
        self.X = X
        self.y = y

        for class_ in self.order[:-1]:

            sm = SMOTE(random_state=8, sampling_strategy=resampling_ratio)
            class_X = self.X
            class_y = np.array([1 if label==class_ else 0 for label in self.y])
            try:
                train_X, train_y = sm.fit_resample(class_X, class_y)
            except ValueError:
                print(class_)
                print(np.unique(class_y, return_counts=True))
                resampling_ratio = min(1,resampling_ratio*2)
                sm = SMOTE(random_state=8, sampling_strategy=resampling_ratio)
                train_X, train_y = sm.fit_resample(class_X, class_y)

            #print(np.unique(train_y, return_counts=True))
            #print(class_X.shape, class_y.shape)
            model = KNeighborsClassifier(metric='euclidean', n_neighbors=8)
            model.fit(train_X, train_y)
            #print(class_X.shape,class_y.shape)

            self.models.append(model)
            self.X = self.X[class_y == 0]
            self.y = self.y[class_y == 0]
            #print(self.X.shape,self.y.shape, '\n')

            resampling_ratio += 1/20
    
    def predict(self,X_test):
        self.predictions = np.array([-1 for i in range(X_test.shape[0])])
        index = np.zeros((X_test.shape[0]))
        for i in range(len(self.order) - 1):
            model = self.models[i]
            try:
                raw_pred = model.predict(X_test[index == 0])
            except ValueError:
                print(f'fermati alla classe {self.order[i]}')
                break

            raw_pred[raw_pred == 0] = -1
            raw_pred[raw_pred == 1] = self.order[i]

            self.predictions[index == 0] = raw_pred

            index[self.predictions != -1] = 1 # già fatti
            #print(self.predictions)

        
        self.predictions[self.predictions == -1] = self.order[i+1]
        return self.predictions

        

"""
print(y)
pip = Pipeline(order)
pip.fit(X,y)
print(pip.predict(X))
"""