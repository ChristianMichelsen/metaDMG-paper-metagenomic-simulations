#%%


import arviz as az
import bambi as bmb

#%%


class GLM:
    def __init__(
        self,
        cols_to_use,
        family="bernoulli",
        seed=42,
        do_scale=True,
        y_col="y",
    ):
        self.cols_to_use = cols_to_use
        self.family = family
        self.seed = seed
        self.do_scale = do_scale
        self.y_col = y_col
        self.is_fitted = False

        self.formula = self.get_formula()

    def get_formula(self):
        prefix = "scale(" if self.do_scale else ""
        suffix = ")" if self.do_scale else ""

        formula = f"{self.y_col}['1'] ~ "
        formula += " + ".join([f"{prefix}{col}{suffix}" for col in self.cols_to_use])
        return formula

    def init(self, Xy):
        self.model = bmb.Model(self.formula, Xy, family=self.family)
        self.model.build()
        return self

    def fit(self, Xy=None, draws=1000, chains=4, cores=1):

        if Xy is not None:
            self.init(Xy)

        self.results = self.model.fit(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=self.seed,
        )

        self.summary = az.summary(self.results)

        self.t_values = (
            (self.summary["mean"] / self.summary["sd"])
            .abs()
            .sort_values(ascending=False)
        )

        self.is_fitted = True
        return self

    def plot_trace(self, figsize=(10, 20)):

        # Use ArviZ to plot the results
        az.plot_trace(
            self.results,
            compact=False,
            figsize=(10, 20),
            show=True,
        )

    def predict(self, Xy):

        y_predict = (
            self.model.predict(
                self.results,
                kind="mean",  # pps, mean
                data=Xy[self.cols_to_use],
                inplace=False,
            )
            .posterior["y_mean"]
            .mean(dim=["chain", "draw"])
            .values
        )

        return y_predict


#%%


def make_classification_comparison(cols, Xy_scaled, SEED=42):

    cols_to_use = [col for col in cols[:-1]]

    d_models = {}
    d_results = {}
    d_cols = {}

    while len(cols_to_use) > 0:

        i = len(cols_to_use)
        print(i)

        formula = "y['1'] ~ " + " + ".join(cols_to_use)

        model_scaled = bmb.Model(
            formula,
            Xy_scaled,
            family="bernoulli",
        )

        results_scaled = model_scaled.fit(
            draws=1000,
            chains=4,
            cores=1,
            random_seed=SEED,
        )

        # Key summary and diagnostic info on the model parameters
        summary_scaled = az.summary(results_scaled)

        t_values = (
            (summary_scaled["mean"] / summary_scaled["sd"])
            .abs()
            .sort_values(ascending=True)
        )

        d_models[i] = model_scaled
        d_results[i] = results_scaled
        d_cols[i] = [col for col in cols_to_use]

        col_to_remove = (
            t_values.index[0] if t_values.index[0] != "Intercept" else t_values.index[1]
        )
        cols_to_use.remove(col_to_remove)

    models_dict = d_results
    df_compare = az.compare(models_dict)
    df_compare

    df_compare["d_loo"] / df_compare["dse"]

    az.plot_compare(df_compare, insample_dev=False)
    az.plot_compare(df_compare.iloc[:-3], insample_dev=False)

    d_cols[5]
    d_cols[6]
    d_cols[7]
    d_cols[8]
    d_cols[9]
