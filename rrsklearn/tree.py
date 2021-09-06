from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import BaseDecisionTree
from sklearn.utils.extmath import safe_sparse_dot
from .rotation import RandomRotation

class RRBaseDecisionTree(BaseDecisionTree):
    '''Base class for Random Rotation Decision Trees

    Warning: This class should not be used directly.
    Use derived classes instead.
    '''

    def rotate(self, X):
        return self.rotater.transform(X)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated"):
        self.rotater = RandomRotation(random_state=self.random_state).fit(X)
        return super().fit(self.rotate(X), y,
                           sample_weight=sample_weight,
                           check_input=check_input,
                           X_idx_sorted=X_idx_sorted)

    def predict(self, X, check_input=True):
        return super().predict(self.rotate(X), check_input=check_input)

    def predict_proba(self, X, check_input=True):
        return super().predict_proba(self.rotate(X), check_input=check_input)

    def apply(self, X, check_input=True):
        return super().apply(self.rotate(X), check_input=check_input)

    def decision_path(self, X, check_input=True):
        return super().decision_path(self.rotate(X), check_input=check_input)

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        return super().cost_complexity_pruning_path(self.rotate(X),
                                                    y,
                                                    sample_weight=sample_weight)


class RRDecisionTreeClassifier(RRBaseDecisionTree, DecisionTreeClassifier):
    '''Mixes Random Rotation BaseDecisionTree with DecisionTreeClassifier'''
    pass


class RRDecisionTreeRegressor(RRBaseDecisionTree, DecisionTreeRegressor):
    '''Mixes Random Rotation BaseDecisionTree with DecisionTreeRegressor'''
    pass
