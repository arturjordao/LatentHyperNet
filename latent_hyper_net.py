import numpy as np
import sklearn
import copy

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Model
from keras.layers import *


class LatentHyperNet(BaseEstimator, ClassifierMixin):
    __name__ = 'Latent Hyper Net'

    def __init__(self, n_comp=2,
                 dm_method=None, oaa=False, as_network=True,
                 model=None, layers=None, batch_size=32):
        self.n_comp = n_comp
        self.dm_layer = []
        self.dm_method = dm_method
        self.oaa = oaa
        self.as_network = as_network
        self.model = self.custom_model(model=model, layers=layers)
        self.layers = layers
        self.batch_size = batch_size

    def custom_model(self, model, layers):

        for i in range(0, len(model.layers)):
            if i > layers[-1]:
                model.layers.pop()

        outputs = []
        for i in layers:
            layer = model.get_layer(index=i)
            outputs.append(Flatten()(layer.output))

        model = Model(model.input, outputs)
        return model

    def pls_as_network(self):
        pls_into_fc = []
        for layer_idx in range(0, len(self.layers)):
            for i in range(0, len(self.dm_layer[layer_idx])):
                dm = self.dm_layer[layer_idx][i]
                id = '{}_{}'.format(layer_idx, i)

                w = dm.x_rotations_
                x_mean = dm.x_mean_
                x_std = dm.x_std_

                H = self.model.get_layer(index=self.layers[layer_idx]).output
                H = Flatten(name='flatten_pls_' + id)(H)

                H = Lambda(lambda x: (x - x_mean) / x_std)(H)
                H = Dense(self.n_comp, weights=[w],
                          use_bias=False,
                          trainable=False,
                          name='pls_model_' + id)(H)
                pls_into_fc.append(H)
                H = None

        H = concatenate(pls_into_fc)
        self.model = Model(self.model.input, H)

        for i in range(0, len(self.model.layers)):
            self.model.layers[i].trainable = False

        return self

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError()

        if (y.shape[1]==1):
            self.oaa = False

        self.dm_layer = [[] for i in range(0, len(self.layers))]
        dm_template = PLSRegression(n_components=self.n_comp)

        X = self.model.predict(X)
        y_tmp = np.argmax(y, axis=1)
        num_classes = np.unique(y_tmp)

        for layer_idx in range(0, len(self.layers)):
            if self.oaa:
                for c in num_classes:
                    target = np.ones(y_tmp.shape)
                    target[y_tmp != c] = -1
                    dm = copy.copy(dm_template)
                    dm.fit(X[layer_idx], target)
                    self.dm_layer[layer_idx].append(dm)
                    del dm
            else:
                dm = copy.copy(dm_template)
                dm.fit(X[layer_idx], y)
                self.dm_layer[layer_idx].append(dm)
                del dm

        if self.as_network:
            self.pls_as_network()

        return self

    def transform(self, X):
        proj_x = None

        X = self.model.predict(X, batch_size=256)

        for layer_idx in range(0, len(self.layers)):
            for pls_model in self.dm_layer[layer_idx]:
                if proj_x is None:
                    proj_x = pls_model.transform(X[layer_idx])
                else:
                    proj_tmp = pls_model.transform(X[layer_idx])
                    proj_x = np.column_stack((proj_x, proj_tmp))

        return proj_x

class LatentHyperNetSingleProjection(BaseEstimator, ClassifierMixin):
    __name__ = 'Latent Hyper Net'

    def __init__(self, n_comp=2,
                 dm_method=None, oaa=False, as_network=True,
                 model=None, layers=None, batch_size=32):
        self.n_comp = n_comp
        self.dm_layer = []
        self.dm_method = dm_method
        self.oaa = oaa
        self.as_network = as_network
        self.model = self.custom_model(model=model, layers=layers)
        self.layers = layers
        self.batch_size = batch_size

    def custom_model(self, model, layers):

        for i in range(0, len(model.layers)):
            if i > layers[-1]:
                model.layers.pop()

        outputs = []
        for i in layers:
            layer = model.get_layer(index=i)
            outputs.append(Flatten()(layer.output))

        outputs = concatenate(outputs)
        model = Model(model.input, outputs)
        return model

    def pls_as_network(self):
        pls_into_fc = []

        id = 0
        for i in range(0, len(self.dm_layer)):
            dm = self.dm_layer[i]
            id = id + 1

            w = dm.x_rotations_
            x_mean = dm.x_mean_
            x_std = dm.x_std_

            H = self.model.get_layer(index=-1).output
            H = Lambda(lambda x: (x - x_mean) / x_std)(H)
            H = Dense(self.n_comp, weights=[w],
                      use_bias=False,
                      trainable=False,
                      name='pls_model_{}'.format(id))(H)
            pls_into_fc.append(H)
            H = None

        H = concatenate(pls_into_fc)
        self.model = Model(self.model.input, H)

        for i in range(0, len(self.model.layers)):
            self.model.layers[i].trainable = False

        return self

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError()

        # if np.unique(y).shape[0] == 2:
        #     self.oaa = False

        self.dm_layer = []
        dm_template = PLSRegression(n_components=self.n_comp)

        X = self.model.predict(X)
        y_tmp = np.argmax(y, axis=1)
        num_classes = np.unique(y_tmp)

        if self.oaa:
            for c in num_classes:
                target = np.ones(y_tmp.shape)
                target[y_tmp != c] = -1
                dm = copy.copy(dm_template)
                dm.fit(X, target)
                self.dm_layer.append(dm)
                del dm
        else:
            dm = copy.copy(dm_template)
            dm.fit(X, y)
            self.dm_layer.append(dm)
            del dm

        if self.as_network:
            self.pls_as_network()

        return self

    def transform(self, X):
        proj_x = None

        X = self.model.predict(X)

        for pls_model in self.dm_layer:
            if proj_x is None:
                proj_x = pls_model.transform(X)
            else:
                proj_tmp = pls_model.transform(X)
                proj_x = np.column_stack((proj_x, proj_tmp))

        return proj_x