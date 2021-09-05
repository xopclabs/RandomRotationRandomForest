from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.extmath import safe_sparse_dot
from .rotation import RandomRotation


class RRDecisionTreeClassifier(DecisionTreeClassifier):

    def __init__(self, *,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)

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


class RRDecisionTreeRegressor(DecisionTreeRegressor):

    def __init__(self, *,
                    criterion="mse",
                    splitter="best",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.,
                    max_features=None,
                    random_state=None,
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.,
                    min_impurity_split=None,
                    ccp_alpha=0.0):
            super().__init__(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                ccp_alpha=ccp_alpha
            )

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

    def apply(self, X, check_input=True):
        return super().apply(self.rotate(X), check_input=check_input)

    def decision_path(self, X, check_input=True):
        return super().decision_path(self.rotate(X), check_input=check_input)
