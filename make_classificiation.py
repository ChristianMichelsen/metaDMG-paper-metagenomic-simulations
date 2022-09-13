#%%

from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import utils

#%%


parser_results_path = utils.parse.compile(
    "{sample}__{sim_name:SimName}__{N_reads:Int}.results",
    dict(Int=int, SimName=lambda s: utils.fix_sim_name(s, utils.D_SIM_NAME_TRANSLATE)),
)


def parse_path(path):
    d = parser_results_path.parse(path.stem).named
    return d["sample"], d["N_reads"], d["sim_name"]


reload(utils)


def load_df_results():

    paths = sorted((Path("data") / "results").glob("*.parquet"))

    dfs = []
    for path in tqdm(paths):
        # break

        sample, N_reads, simulation_method = parse_path(path)

        df_in = (
            pd.read_parquet(path)
            .rename(columns={"sample": "sample_name"})
            .astype({"tax_id": int})
            # .query("tax_rank == 'species'")
        )

        df = utils.add_simulation_information_to_df_comparison(
            df_in,
            sample,
            N_reads,
            simulation_method,
            N_reads_col="N_reads_simulated",
        )

        dfs.append(df)

    df_results = pd.concat(dfs)

    df_results["D_max_significance"] = df_results["D_max"] / df_results["D_max_std"]
    df_results["rho_Ac_abs"] = np.abs(df_results["rho_Ac"])

    df_results["Bayesian_D_max_significance"] = (
        df_results["Bayesian_D_max"] / df_results["Bayesian_D_max_std"]
    )
    df_results["Bayesian_rho_Ac_abs"] = np.abs(df_results["Bayesian_rho_Ac"])

    return df_results


df_results = load_df_results()


#%%

df_only_ancient = df_results.query("simulated_only_ancient == True")
df_only_ancient_frag = df_only_ancient.query("simulation_method == 'frag'")
df_only_ancient_art = df_only_ancient.query("simulation_method == 'art'")



#%%

x=x


#%%

# fig, ax = plt.subplots(figsize=(10, 6))

# sns.set(font_scale = 2)
sns.set_style("ticks")


g = sns.relplot(
    data=df_only_ancient,
    x="Bayesian_z",
    y="Bayesian_D_max",
    col="simulation_method",
    hue="sample",
    size="N_reads_simulated",
    kind="scatter",
    alpha=0.5,
    height=7,
)
g.tight_layout()
g.savefig("figures/facet_plot_bayesian.pdf")

# %%

g = sns.relplot(
    data=df_only_ancient,
    x="lambda_LR",
    y="D_max",
    col="simulation_method",
    hue="sample",
    size="N_reads_simulated",
    kind="scatter",
    alpha=0.5,
    height=7,
)
g.tight_layout()
g.savefig("figures/facet_plot_MAP.pdf")

# %%

#%%

import statsmodels.formula.api as smf
# import statsmodels
# statsmodels.__version__


#%%


def ols_formula(df, dependent_var, *excluded_cols):
    """
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    """
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + " ~ " + " + ".join(df_columns)


#%%

from scipy.stats import norm as sp_norm


def get_n_sigma_probability(n_sigma):
    return sp_norm.cdf(n_sigma) - sp_norm.cdf(-n_sigma)


CONF_1_SIGMA = get_n_sigma_probability(1)

#%%


df_only_ancient_frag_1M = df_only_ancient_frag.query("N_reads_simulated == 1_000_000")

good_samples = [
    # "Lake-9",
    # "Lake-7",
    "Cave-22",
    # "Cave-100",
    "Cave-102",
    "Pitch-6",
]
df_only_ancient_art_1M_cut = df_only_ancient_art.query(
    "N_reads_simulated == 1_000_000 & sample in @good_samples & k_sum_total > 10"
)

cols = [
    "Bayesian_z",
    "Bayesian_D_max",
    # "Bayesian_D_max_std",
    "Bayesian_D_max_significance",
    # "Bayesian_q",
    # "Bayesian_phi",
    # "Bayesian_rho_Ac",
    "Bayesian_rho_Ac_abs",
    # "N_reads",
    # "N_alignments",
    # "mean_L",
    # "mean_GC",
    "simulation_method",
]

X = pd.concat([df_only_ancient_frag_1M[cols], df_only_ancient_art_1M_cut[cols]])
X.loc[:, "simulation_method"] = pd.get_dummies(X["simulation_method"])["art"]


formula = ols_formula(X, "simulation_method") +  "-1"


log_reg = smf.logit(formula, data=X).fit()
log_reg.summary(alpha=1 - CONF_1_SIGMA)

# print(log_reg.params)
# log_reg.pred_table()


#%%


log_reg.summary(alpha=1 - CONF_1_SIGMA)
log_reg.summary2(alpha=1 - CONF_1_SIGMA)



#%%


exog = log_reg.model.exog
predicted_mean = log_reg.model.predict(log_reg.params, exog)

covb = log_reg.cov_params()
var_pred_mean = (exog * np.dot(covb, exog.T).T).sum(1)
np.sqrt(var_pred_mean)

#%%
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families


res = GLM(
    X["simulation_method"],
    X.drop(columns="simulation_method"),
    family=families.Binomial(),
).fit(attach_wls=True, atol=1e-10)
print(res.summary())


res.results_wls

res.get_prediction(X.drop(columns="simulation_method")).predicted_mean
# res.get_prediction(X.drop(columns="simulation_method")).se
np.sqrt(res.get_prediction(X.drop(columns="simulation_method")).var_pred_mean)
res.get_prediction(X.drop(columns="simulation_method")).conf_int()




#%%

print(log_reg.summary())

#%%

_df_art = df_only_ancient_art.query(
    "N_reads_simulated != 1_000_000 & sample in @good_samples & k_sum_total > 10"
)[cols]
log_reg.predict(_df_art).min()

#%%


# conf_int',
#  'cov_kwds',
#  'cov_params',
#  'cov_type

# log_reg.get_prediction()

log_reg.cov_kwds
log_reg.cov_params()
log_reg.cov_type


#%%

log_reg.pred_table(threshold=.1)
log_reg.pred_table(threshold=.5)
log_reg.pred_table(threshold=.9)

fittedvalues = log_reg.model.cdf(log_reg.fittedvalues)


#%%

_df_frag = df_only_ancient_frag.query("N_reads_simulated != 1_000_000")[cols]

log_reg.predict(_df_frag).max()
log_reg.predict(_df_frag).argmax()


log_reg.predict(_df_frag.iloc[756:758])


#%%


#%%

model_odds = pd.DataFrame(np.exp(log_reg.params), columns=["OR"])
model_odds["z-value"] = log_reg.pvalues
model_odds[["2.5%", "97.5%"]] = np.exp(log_reg.conf_int())

model_odds
