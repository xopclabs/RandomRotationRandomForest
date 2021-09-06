import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from .tree import RRDecisionTreeRegressor


class RRBaseGradientBoosting(BaseGradientBoosting):
    """Abstract base class for Random Rotation Gradient Boosting."""

    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
                   random_state, X_csc=None, X_csr=None):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self.loss_
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)

            rs = random_state.randint(0, 2**32 - 1)
            # induce regression tree on residuals
            tree = RRDecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=rs,
                ccp_alpha=self.ccp_alpha)

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight,
                     check_input=False)

            # update tree leaves
            loss.update_terminal_regions(
                tree.tree_, X, y, residual, raw_predictions, sample_weight,
                sample_mask, learning_rate=self.learning_rate, k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions


class RRGradientBoostingClassifier(RRBaseGradientBoosting, GradientBoostingClassifier):
    '''Mixes Random Rotation BaseGradientBoosting with GradientBoostingClassifier'''
    pass


class RRGradientBoostingRegressor(RRBaseGradientBoosting, GradientBoostingRegressor):
    '''Mixes Random Rotation BaseGradientBoosting with GradientBoostingRegressor'''
    pass
