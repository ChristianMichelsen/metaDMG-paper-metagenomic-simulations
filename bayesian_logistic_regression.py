#%%

import warnings

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

#%%

_custom_data_name = "predictions_constant_data"


#%%


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, do_scale=True):
        self.do_scale = do_scale

    def _get_X(self, X):
        return X[self.X_columns_].copy()

    def _get_Xy(self, X, y):
        Xy = self._get_X(X)
        Xy["y"] = y
        return Xy

    def _get_formula(self):
        formula = f"y['1'] ~ "
        formula += " + ".join([f"{col}" for col in self.X_columns_])
        return formula

    def _scale_X(self, X):
        Xt = (X.copy() - self.X_mean_) / self.X_std_
        return Xt

    def _get_Xy_scaled(self, X, y):
        Xt = self._scale_X(X)
        return self._get_Xy(Xt, y)

    def fit(self, X, y, draws=1000, chains=4, cores=1):

        assert isinstance(X, pd.DataFrame)
        check_X_y(X, y)

        self.X_columns_ = list(X.columns)
        self.X_mean_ = X.mean()
        self.X_std_ = X.std()

        if self.do_scale:
            Xy = self._get_Xy_scaled(X, y)
        else:
            Xy = self._get_Xy(X, y)

        self.formula_ = self._get_formula()
        model = bmb.Model(self.formula_, Xy, family="bernoulli")

        self.N_variables_ = len(model.terms)

        results = model.fit(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=42,
        )

        self._add_results(results)

    def _compute_summary_stats(self, results):
        summary = az.summary(results)
        coefficients = summary["mean"]
        t_values = (summary["mean"] / summary["sd"]).abs().sort_values(ascending=False)
        return summary, coefficients, t_values

    def _add_results(self, results):
        p = self._compute_summary_stats(results)
        self.summary_, self.coefficients_, self.t_values_ = p
        self.results_ = results

    def get_summary(self):
        return self.summary_

    def plot_trace(self, variables=None, figsize=None):

        check_is_fitted(self, "results_")

        if figsize is None:
            figsize = (10, self.N_variables_ * 5)

        if variables is None:
            variables = ["Intercept"] + self.X_columns_

        # Use ArviZ to plot the results_
        az.plot_trace(
            self.results_,
            var_names=variables,
            compact=False,
            figsize=figsize,
            show=True,
        )

    def _get_tmp_model(self):

        mock_data = {"y": [0, 1]}
        for col in self.X_columns_:
            mock_data[col] = [0, 1]

        model = bmb.Model(
            self.formula_,
            pd.DataFrame(mock_data),
            family="bernoulli",
        )
        return model

    def _compute_idata(self, X):

        check_is_fitted(self, ["X_mean_", "X_std_"])

        if self.do_scale:
            X = self._scale_X(X)

        idata = self._get_tmp_model().predict(
            self.results_,
            kind="mean",  # pps, mean
            data=X,
            inplace=False,
        )

        return idata

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self, "coefficients_")

        X = self._get_X(X)

        # Input validation
        check_array(X)

        idata = self._compute_idata(X)
        ys = idata.posterior["y_mean"].stack(sample=("chain", "draw")).values

        df_predict = pd.DataFrame(
            {
                "mean": np.mean(ys, axis=1),
                "std": np.std(ys, axis=1),
                "CI_16": np.percentile(ys, 16, axis=1),
                "CI_84": np.percentile(ys, 84, axis=1),
                "median": np.mean(ys, axis=1),
            }
        )

        return df_predict

    def plot_prediction(self, Xy):

        if len(Xy) > 10:
            warnings.warn(
                "Too many samples to plot, consider using a subset "
                "or gml.plot_all_predictions(Xy)"
            )

        idata = self._compute_idata(Xy)

        return az.plot_trace(
            idata.posterior["y_mean"],
            compact=False,
            figsize=(10, 5 * len(Xy)),
            show=True,
        )

    def save(self, path):

        check_is_fitted(self, "coefficients_")

        results = self.results_.copy()

        del results.log_likelihood
        del results.sample_stats
        del results.observed_data
        if "y_mean" in results.posterior:
            del results.posterior["y_mean"]

        dataset = az.convert_to_inference_data(
            {
                "X_mean": self.X_mean_,
                "X_std": self.X_std_,
            },
            group=_custom_data_name,
        )
        dataset[_custom_data_name].attrs["do_scale"] = str(self.do_scale)
        dataset[_custom_data_name].attrs["formula"] = self.formula_
        dataset[_custom_data_name].attrs["X_columns"] = self.X_columns_
        dataset[_custom_data_name].attrs["N_variables"] = self.N_variables_

        results.add_groups(dataset)
        results.to_netcdf(path)

    def _load_results(self, results):

        self.formula_ = results[_custom_data_name].attrs["formula"]
        self.X_columns_ = results[_custom_data_name].attrs["X_columns"]
        self.N_variables_ = results[_custom_data_name].attrs["N_variables"]

        X_mean = results[_custom_data_name].X_mean.values[0]
        self.X_mean_ = pd.Series(X_mean, index=self.X_columns_)

        X_std = results[_custom_data_name].X_std.values[0]
        self.X_std_ = pd.Series(X_std, index=self.X_columns_)

        results.__delattr__(_custom_data_name)
        self._add_results(results)

    @classmethod
    def load(cls, path):
        results = az.from_netcdf(path)
        do_scale = bool(results[_custom_data_name].attrs["do_scale"])

        blr = cls(do_scale=do_scale)
        blr._load_results(results)
        return blr
