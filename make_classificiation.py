#%%

from importlib import reload
from os import X_OK
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import utils
import utils_classification

#%%


parser_results_path = utils.parse.compile(
    "{sample}__{sim_name:SimName}__{N_reads:Int}.results",
    dict(Int=int, SimName=lambda s: utils.fix_sim_name(s, utils.D_SIM_NAME_TRANSLATE)),
)


def parse_path(path):
    d = parser_results_path.parse(path.stem).named
    return d["sample"], d["N_reads"], d["sim_name"]


# reload(utils)


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


good_samples = [
    # "Lake-9",
    # "Lake-7",
    "Lake-7-forward",
    "Cave-22",
    # "Cave-100",
    "Cave-100-forward",
    "Cave-102",
    "Pitch-6",
]


df_only_ancient_frag_1M = df_only_ancient_frag.query("N_reads_simulated == 1_000_000")

df_only_ancient_art_1M = df_only_ancient_art.query(
    "N_reads_simulated == 1_000_000 & sample in @good_samples & k_sum_total > 10"
)


#%%


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    df_only_ancient_art_1M["Bayesian_z"],
    df_only_ancient_art_1M["Bayesian_D_max"],
    "o",
    color="green",
    alpha=0.5,
    label="Ancient",
)

ax.plot(
    df_only_ancient_frag_1M["Bayesian_z"],
    df_only_ancient_frag_1M["Bayesian_D_max"],
    "o",
    color="red",
    alpha=0.5,
    label="Modern",
)


ax.set(xlabel="Bayesian_z", ylabel="Bayesian_D_max")
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
ax.legend()
fig

#%%

cols = [
    "Bayesian_z",
    "Bayesian_D_max",
    "Bayesian_D_max_std",
    "Bayesian_D_max_significance",
    "Bayesian_D_max_confidence_interval_1_sigma_low",
    "Bayesian_q",
    "Bayesian_phi",
    "Bayesian_rho_Ac",
    "Bayesian_rho_Ac_abs",
    "N_reads",
    "N_alignments",
    "mean_L",
    "mean_GC",
    "simulation_method",
]


Xy = pd.concat(
    [df_only_ancient_frag_1M[cols], df_only_ancient_art_1M[cols]]
).reset_index(drop=True)
Xy.loc[:, "simulation_method"] = pd.get_dummies(Xy["simulation_method"])["art"]
Xy = Xy.rename(columns={"simulation_method": "y"})


#%%

cols_to_use = [
    # "Bayesian_z",
    # "Bayesian_D_max",
    # "Bayesian_D_max_significance",
    "Bayesian_phi",
    # "Bayesian_rho_Ac",
]

reload(utils_classification)
glm = utils_classification.GLM(cols_to_use=cols_to_use, do_scale=True)
glm.init(Xy)
glm.model

glm.fit(Xy)

glm.summary
glm.plot_trace()
glm.model.graph()


# A Laplace prior with mean of 0 and scale of 10

# Set the prior when adding a term to the model; more details on this below.


priors = {}
for col in cols_to_use:
    priors[col] = bmb.Prior("Normal", mu=Xy[col].mean(), sigma=2.5 / Xy[col].std())


model = bmb.Model(
    " y['1'] ~ Bayesian_z + Bayesian_D_max + Bayesian_D_max_significance + Bayesian_phi + Bayesian_rho_Ac",
    Xy,
    priors=priors,
    family="bernoulli",
)
model.build()
model.plot_priors()


results = model.fit(
    draws=1000,
    chains=4,
    cores=1,
    random_seed=42,
)
az.summary(results)

az.plot_trace(
            results,
            compact=False,
            figsize=(10, 20),
            show=True,
        )


#%%

x = x

#%%


df_only_ancient_frag_not_1M = df_only_ancient_frag.query(
    "N_reads_simulated != 1_000_000"
)


df_only_ancient_art_not_1M = df_only_ancient_art.query(
    "N_reads_simulated != 1_000_000"
).reset_index(drop=True)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    df_only_ancient_art_not_1M["Bayesian_z"],
    df_only_ancient_art_not_1M["Bayesian_D_max"],
    "o",
    color="green",
    alpha=0.5,
    label="Ancient",
)

ax.plot(
    df_only_ancient_frag_not_1M["Bayesian_z"],
    df_only_ancient_frag_not_1M["Bayesian_D_max"],
    "o",
    color="red",
    alpha=0.5,
    label="Modern",
)


ax.set(xlabel="Bayesian_z", ylabel="Bayesian_D_max")
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
ax.legend()
fig

#%%


y_pred_frag = glm.predict(df_only_ancient_frag_not_1M)
y_pred_art = glm.predict(df_only_ancient_art_not_1M)


#%%


glm_simple = utils_classification.GLM(
    cols_to_use=["Bayesian_z", "Bayesian_D_max"], do_scale=True
)
glm_simple.fit(Xy)

glm_simple.summary
glm_simple.plot_trace()


y_pred_frag_simple = glm_simple.predict(df_only_ancient_frag_not_1M)
y_pred_art_simple = glm_simple.predict(df_only_ancient_art_not_1M)


#%%

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_frag_simple, 100, range=(0, 1), label="Modern", histtype="step")
ax.hist(y_pred_art_simple, 100, range=(0, 1), label="Ancient", histtype="step")
# ax.set_yscale("log")
ax.legend()
ax.set(ylim=(0, 10))


#%%


fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_frag, 100, range=(0, 1), label="Modern", histtype="step")
ax.hist(y_pred_art, 100, range=(0, 1), label="Ancient", histtype="step")
# ax.set_yscale("log")
ax.set(ylim=(0, 10))
ax.legend()

#%%

y_true = np.concatenate([np.zeros(len(y_pred_frag)), np.ones(len(y_pred_art))])


y_pred = np.concatenate([y_pred_frag, y_pred_art])
y_pred_simple = np.concatenate([y_pred_frag_simple, y_pred_art_simple])


#%%

from sklearn.metrics import auc, roc_curve

fpr, tpr, _ = roc_curve(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_pred_simple)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()


#%%

# # sns.set(font_scale = 2)
# sns.set_style("ticks")


# g_bayesian = sns.relplot(
#     data=df_only_ancient,
#     x="Bayesian_z",
#     y="Bayesian_D_max",
#     col="simulation_method",
#     hue="sample",
#     size="N_reads_simulated",
#     kind="scatter",
#     alpha=0.5,
#     height=7,
# )
# g_bayesian.tight_layout()
# g_bayesian.savefig("figures/facet_plot_bayesian.pdf")
# g_bayesian
# plt.show()


# # %%

# g_MAP = sns.relplot(
#     data=df_only_ancient,
#     x="lambda_LR",
#     y="D_max",
#     col="simulation_method",
#     hue="sample",
#     size="N_reads_simulated",
#     kind="scatter",
#     alpha=0.5,
#     height=7,
# )
# g_MAP.tight_layout()
# g_MAP.savefig("figures/facet_plot_MAP.pdf")

# g_MAP
# plt.show()


# #%%


# good_samples = [
#     # "Lake-9",
#     # "Lake-7",
#     "Lake-7-forward",
#     "Cave-22",
#     # "Cave-100",
#     "Cave-100-forward",
#     "Cave-102",
#     "Pitch-6",
# ]

# cols = [
#     "Bayesian_z",
#     "Bayesian_D_max",
#     "Bayesian_D_max_std",
#     "Bayesian_D_max_significance",
#     "Bayesian_D_max_confidence_interval_1_sigma_low",
#     "Bayesian_q",
#     "Bayesian_phi",
#     "Bayesian_rho_Ac",
#     "Bayesian_rho_Ac_abs",
#     "N_reads",
#     "N_alignments",
#     "mean_L",
#     "mean_GC",
#     "simulation_method",
# ]


# #%%


# #%%

# from math import ceil

# from matplotlib.backends.backend_pdf import PdfPages


# def foo():

#     fig_name = Path("figures") / "overview_bayesian_all.pdf"
#     fig_name.parent.mkdir(exist_ok=True)

#     N_reads_simulated_all = sorted(df_only_ancient.N_reads_simulated.unique())
#     log_cols = ["N_reads", "N_alignments", "Bayesian_phi"]

#     with PdfPages(fig_name) as pdf:

#         for N_reads_simulated in N_reads_simulated_all:

#             _df_only_ancient_frag = df_only_ancient_frag.query(
#                 f"N_reads_simulated == {N_reads_simulated}"
#             )
#             _df_only_ancient_art = df_only_ancient_art.query(
#                 f"N_reads_simulated == {N_reads_simulated} & sample in @good_samples & k_sum_total > 10"
#             )

#             X = pd.concat(
#                 [_df_only_ancient_frag[cols], _df_only_ancient_art[cols]]
#             ).reset_index(drop=True)
#             X.loc[:, "simulation_method"] = pd.get_dummies(X["simulation_method"])[
#                 "art"
#             ]

#             fig, axs = plt.subplots(2, ceil(len(cols[:-1]) / 2), figsize=(24, 9))
#             for col, ax in zip(cols[:-1], axs.flatten()):

#                 sns.histplot(
#                     X,
#                     x=col,
#                     hue="simulation_method",
#                     element="step",
#                     stat="density",
#                     common_norm=False,
#                     ax=ax,
#                     fill=False,
#                     # kde=True,
#                     legend=False,
#                     log_scale=True if col in log_cols else False,
#                 )
#                 ax.set(xlim=df_only_ancient[col].quantile([0.0, 1.0]))

#             sns.histplot(
#                 X,
#                 x="simulation_method",
#                 hue="simulation_method",
#                 element="step",
#                 # stat="density",
#                 # common_norm=False,
#                 ax=axs.flatten()[-1],
#                 fill=False,
#                 legend=True,
#             )

#             fig.suptitle("N_reads_simulated = " + str(N_reads_simulated))

#             fig.tight_layout()
#             pdf.savefig(fig)
#             plt.close()


# foo()

# #%%

# import statsmodels.formula.api as smf

# # import statsmodels
# # statsmodels.__version__


# #%%


# def ols_formula(df, dependent_var, *excluded_cols):
#     """
#     Generates the R style formula for statsmodels (patsy) given
#     the dataframe, dependent variable and optional excluded columns
#     as strings
#     """
#     df_columns = list(df.columns.values)
#     df_columns.remove(dependent_var)
#     for col in excluded_cols:
#         df_columns.remove(col)
#     return dependent_var + " ~ " + " + ".join(df_columns)


# #%%

# from scipy.stats import norm as sp_norm


# def get_n_sigma_probability(n_sigma):
#     return sp_norm.cdf(n_sigma) - sp_norm.cdf(-n_sigma)


# CONF_1_SIGMA = get_n_sigma_probability(1)

# #%%


# #%%

# #%%


# import arviz as az

# az.style.use("arviz-darkgrid")

# import pymc as pm

# print(f"Running on PyMC v{pm.__version__}")

# #%%

# logistic_model = pm.Model()

# with logistic_model:

#     # # Priors for unknown model parameters
#     # alpha = pm.Normal("alpha", mu=0, sigma=10)
#     # beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
#     # sigma = pm.HalfNormal("sigma", sigma=1)

#     # # Expected value of outcome
#     # mu = alpha + beta[0] * X1 + beta[1] * X2

#     # # Likelihood (sampling distribution) of observations
#     # Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

#     # α = pm.Normal("α", mu=0, sd=10)
#     # β = pm.Normal("β", mu=0, sd=10)

#     # μ = α + pm.math.dot(x_c, β)
#     # θ = pm.Deterministic("θ", pm.math.sigmoid(μ))
#     # bd = pm.Deterministic("bd", -α / β)

#     # y_1 = pm.Bernoulli("y_1", p=θ, observed=y_simple)

#     # trace_simple = pm.sample(1000, tune=1000)
#     formula = "simulation_method ~ Bayesian_z + Bayesian_D_max"
#     pm.glm.GLM.from_formula(formula, Xy_scaled, family=pm.glm.families.Binomial())


# pm.model_to_graphviz(logistic_model)


# with logistic_model:

#     trace = pm.sample(
#         tune=1000,
#         draws=1000,
#         chains=4,
#         init="adapt_diag",
#         cores=4,
#     )


# az.summary(trace)
# pm.plot_trace(trace)


# ppc = pm.sample_ppc(trace_simple, model=model_simple, samples=500)
# preds = np.rint(ppc["y_1"].mean(axis=0)).astype("int")
# print("Accuracy of the simplest model:", accuracy_score(preds, data["outcome"]))
# print("f1 score of the simplest model:", f1_score(preds, data["outcome"]))


# #%%


# #%%

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV

# clf_scaled = LogisticRegression(
#     random_state=0,
#     penalty="l1",
#     solver="liblinear",
#     C=10,
# ).fit(X_train_scaled, y_train)

# mask = (clf_scaled.coef_ != 0)[0]
# X_train_cut = X_train.loc[:, mask]

# clf = LogisticRegression(random_state=0, penalty="l2").fit(X_train_cut, y_train)
# clf.coef_

# # sigmoid( dot([val1, val2], lr.coef_) + lr.intercept_ )

# #%%

# import statsmodels.api as sm

# logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())
# print(logm1.fit().summary())


# bla = logm1.fit_regularized(alpha=1, L1_wt=1)
# bla.params
# bla.fittedvalues


# #%%

# formula = ols_formula(Xy, "simulation_method")  # + "-1"

# log_reg = smf.logit(formula, data=Xy).fit()
# print(log_reg.summary(alpha=1 - CONF_1_SIGMA))

# # print(log_reg.params)
# # log_reg.pred_table()
# #%%


# log_reg = smf.logit(formula, data=Xy_scaled)
# blabla = log_reg.fit_regularized(alpha=0.5, L1_wt=1)
# print(blabla.summary())

# #%%

# import seaborn as sns

# sns.clustermap(X.drop(columns="simulation_method").corr())
# plt.show()


# #%%

# # from firthlogist import FirthLogisticRegression


# #%%

# log_reg.summary(alpha=1 - CONF_1_SIGMA)
# log_reg.summary2(alpha=1 - CONF_1_SIGMA)


# #%%


# exog = log_reg.model.exog
# predicted_mean = log_reg.model.predict(log_reg.params, exog)

# covb = log_reg.cov_params()
# var_pred_mean = (exog * np.dot(covb, exog.T).T).sum(1)
# np.sqrt(var_pred_mean)

# from statsmodels.genmod import families

# #%%
# from statsmodels.genmod.generalized_linear_model import GLM

# res = GLM(
#     X["simulation_method"],
#     X.drop(columns="simulation_method"),
#     family=families.Binomial(),
# ).fit(attach_wls=True, atol=1e-10)
# print(res.summary())


# res.results_wls

# res.get_prediction(X.drop(columns="simulation_method")).predicted_mean
# # res.get_prediction(X.drop(columns="simulation_method")).se
# np.sqrt(res.get_prediction(X.drop(columns="simulation_method")).var_pred_mean)
# res.get_prediction(X.drop(columns="simulation_method")).conf_int()


# #%%

# print(log_reg.summary())

# #%%

# _df_art = df_only_ancient_art.query(
#     "N_reads_simulated != 1_000_000 & sample in @good_samples & k_sum_total > 10"
# )[cols]
# log_reg.predict(_df_art).min()

# #%%


# # conf_int',
# #  'cov_kwds',
# #  'cov_params',
# #  'cov_type

# # log_reg.get_prediction()

# log_reg.cov_kwds
# log_reg.cov_params()
# log_reg.cov_type


# #%%

# log_reg.pred_table(threshold=0.1)
# log_reg.pred_table(threshold=0.5)
# log_reg.pred_table(threshold=0.9)

# fittedvalues = log_reg.model.cdf(log_reg.fittedvalues)


# #%%

# _df_frag = df_only_ancient_frag.query("N_reads_simulated != 1_000_000")[cols]

# log_reg.predict(_df_frag).max()
# log_reg.predict(_df_frag).argmax()


# log_reg.predict(_df_frag.iloc[756:758])


# #%%


# #%%

# model_odds = pd.DataFrame(np.exp(log_reg.params), columns=["OR"])
# model_odds["z-value"] = log_reg.pvalues
# model_odds[["2.5%", "97.5%"]] = np.exp(log_reg.conf_int())

# model_odds

# %%
