import numpy as np
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


class MatrixFactorizationALS:
    def __init__(self,
                 ratings,
                 n_factors=40,
                 item_reg=0.0,
                 user_reg=0.0,
                 verbose=False):
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose

    def als_step_user(self,
                      latent_vectors,
                      fixed_vecs,
                      ratings,
                      _lambda):
        YTY = fixed_vecs.T.dot(fixed_vecs)
        lambdaI = np.eye(YTY.shape[0]) * _lambda

        for u in range(latent_vectors.shape[0]):
            latent_vectors[u, :] = solve((YTY + lambdaI), ratings[u, :].dot(fixed_vecs))
        return latent_vectors

    def als_step_item(self,
                      latent_vectors,
                      fixed_vecs,
                      ratings,
                      _lambda):
        XTX = fixed_vecs.T.dot(fixed_vecs)
        lambdaI = np.eye(XTX.shape[0]) * _lambda

        for i in range(latent_vectors.shape[0]):
            latent_vectors[i, :] = solve((XTX + lambdaI), ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_items, self.n_factors))
        self.partial_train(n_iter)

    def partial_train(self, n_iter):
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print(f'\tcurrent iteration: {ctr}')
            self.user_vecs = self.als_step_user(self.user_vecs,
                                                self.item_vecs,
                                                self.ratings,
                                                self.user_reg)
            self.item_vecs = self.als_step_item(self.item_vecs,
                                                self.user_vecs,
                                                self.ratings,
                                                self.item_reg)

            ctr += 1

    def predict(self, u, i):
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)

    def predict_all(self):
        predictions = np.zeros((self.user_vecs.shape[0],
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        return predictions

    def calculate_learning_curve(self, iter_array, test):
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print(f'Iteration: {n_iter}')
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            if self._v:
                print(f'Train mse: {str(self.train_mse[-1])}')
                print(f'Test mse: {str(self.test_mse[-1])}')
            iter_diff = n_iter
