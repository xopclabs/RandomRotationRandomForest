from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_random_state


class BaseRandomRotation(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    '''Base class for random rotations.
    Warning: This class should not be used directly.
    Use derived classes instead.
    '''

    @abstractmethod
    def __init__(self, n_features='auto', *,
                 dense_output=False,
                 random_state=None):
        self.n_features = n_features
        self.dense_output = dense_output
        self.random_state = random_state

    def _make_random_rotation(self):
        '''Generate the random rotation matrix.
        Parameters
        ----------
        n_features : int,
            Dimensionality of the feature space.
        Returns
        -------
        rotation_matrix : ndarray of shape \
                (n_features, n_features)
            The generated random rotation matrix.
        '''
        r = self.random.normal(size=(self.n_features, self.n_features))
        Q, R = np.linalg.qr(r)
        rotation_matrix = np.dot(Q, np.diag(np.sign(np.diag(R))))
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] = -rotation_matrix[:, 0]
        rotation_matrix = rotation_matrix.astype(np.float32)
        return rotation_matrix

    def fit(self, X, y=None):
        '''Generate a random rotation matrix.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y
            Ignored
        Returns
        -------
        self
        '''
        X = self._validate_data(X, accept_sparse=['csr', 'csc'])

        if self.n_features == 'auto':
            _, self.n_features = X.shape
        else:
            if self.n_features <= 0:
                raise ValueError(f'n_features must be greater than 0, got {self.n_features}')
            elif self.n_features != X.shape[1]:
                raise ValueError(f'n_features must match 2nd dimension in X, \
                                got {self.n_features} for shape of {X.shape}')

        # Get new random generator
        self.random = check_random_state(self.random_state)

        # Generate a rotation matrix
        self.rotation_matrix_ = self._make_random_rotation()

        return self

    def transform(self, X):
        '''Transform the data by using matrix product with the random rotation matrix
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to randomly rotate.
        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Rotated array.
        '''
        X = check_array(X, accept_sparse=['csr', 'csc'])

        check_is_fitted(self)

        X_new = safe_sparse_dot(X, self.rotation_matrix_,
                                dense_output=self.dense_output)
        return X_new


class RandomRotation(BaseRandomRotation):
    '''Improve ensemble diversity using Random Rotations.
    Rotation matrix is drawn using Householder QR decomposition.
    Parameters
    ----------
    n_features : int or 'auto', default='auto'
        Dimensionality of the target feature space.
        n_features can be inferred from fitted data when 'auto' is passed.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        rotation matrix at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    rotation_matrix_ : ndarray of shape (n_featuers, n_features)
        Random rotation matrix.
    Examples
    --------
    >>> import numpy as np
    >>> from rrsklearn import RandomRotation
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(100, 10000)
    >>> transformer = RandomRotation(random_state=rng)
    >>> X_new = transformer.fit_transform(X)
    '''

    def __init__(self, n_features='auto', *, random_state=None):
        super().__init__(
            n_features=n_features,
            dense_output=True,
            random_state=random_state)
