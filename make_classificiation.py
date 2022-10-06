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

reload(utils_classification)

#%%

path_alignment_files = Path("input-data") / "data-pre-mapping"


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
            path_alignment_files,
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


df_results_org = load_df_results()
df_results = df_results_org.loc[:, utils_classification.relevant_columns]


#%%


good_samples = [
    "Lake-9",
    # "Lake-7",
    "Lake-7-forward",
    "Cave-22",
    # "Cave-100",
    "Cave-100-forward",
    "Cave-102",
    "Pitch-6",
]


query = (
    "tax_rank == 'species' and simulation_method == 'art' and sample in @good_samples"
)
df_ancient = df_results.query(
    # query + " and simulated_only_ancient == True and k_sum_total > 10 and N_reads > 10"
    query
    + " and simulated_only_ancient == True"
)
df_modern = df_results.query(
    # query + " and simulated_only_ancient != True and N_reads > 10"
    query
    + " and simulated_only_ancient != True"
)


#%%

df_modern_1M = df_modern.query("N_reads_simulated == 1_000_000")
df_ancient_1M = df_ancient.query("N_reads_simulated == 1_000_000")


#%%


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    df_ancient["Bayesian_z"],
    df_ancient["Bayesian_D_max"],
    "o",
    color="green",
    alpha=0.5,
    label="Ancient",
)

ax.plot(
    df_modern["Bayesian_z"],
    df_modern["Bayesian_D_max"],
    "o",
    color="red",
    alpha=0.5,
    label="Modern",
)


ax.set(xlabel="Bayesian_z", ylabel="Bayesian_D_max")
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
ax.legend()

#%%

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    df_ancient_1M["Bayesian_z"],
    df_ancient_1M["Bayesian_D_max"],
    "o",
    color="green",
    alpha=0.5,
    label="Ancient",
)

ax.plot(
    df_modern_1M["Bayesian_z"],
    df_modern_1M["Bayesian_D_max"],
    "o",
    color="red",
    alpha=0.5,
    label="Modern",
)


ax.set(xlabel="Bayesian_z", ylabel="Bayesian_D_max")
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
ax.legend()

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


def get_Xy(df_modern, df_ancient):

    Xy = pd.concat([df_modern[cols], df_ancient[cols]]).reset_index(drop=True)
    Xy["y"] = np.hstack(
        [
            np.zeros(len(df_modern)),
            np.ones(len(df_ancient)),
        ]
    ).astype(int)

    return Xy


#%%

Xy_1M = get_Xy(df_modern_1M, df_ancient_1M)
Xy_all = get_Xy(df_modern, df_ancient)
# Xy_not_1M = get_Xy(df_modern, df_ancient)


#%%

X_columns = [
    "Bayesian_z",
    "Bayesian_D_max",
    "Bayesian_D_max_significance",
    "Bayesian_phi",
    "Bayesian_rho_Ac",
]


X_1M = Xy_1M[X_columns]
y_1M = Xy_1M["y"]

#%%

from bayesian_logistic_regression import BayesianLogisticRegression

path = "BLR_1M.nc"

blr_1M = BayesianLogisticRegression()
blr_1M.fit(X_1M, y_1M, draws=1000)
blr_1M.save(path)

blr_1M.plot_trace()
blr_1M.summary_
blr_1M.t_values_

df_predict_1M = blr_1M.predict(X_1M)
blr_1M.predict(Xy_1M)
blr_1M.predict(df_modern_1M)

# blr_1M.plot_trace()


#%%

blr2 = BayesianLogisticRegression.load(path)
blr2.predict(X_1M)
blr2.predict(Xy_1M)
blr2.predict(df_modern_1M)

blr2.plot_trace()

blr_1M.results_
blr2.results_

blr2.predict(df_modern_1M.iloc[5:10])
blr2.plot_prediction(df_modern_1M.iloc[5:10])


#%%

# y_pred = np.full(len(Xy), -1)
# for i, row in df_predict.iterrows():
#     if row["CI_16"] > 0.5:
#         y_pred[i] = 1
#     elif row["CI_84"] < 0.5:
#         y_pred[i] = 0

# y_true = Xy["y"]

# from sklearn.metrics import classification_report, confusion_matrix

# target_names = ["Unclear", "Clear Modern", "Ancient"]
# print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
# confusion_matrix(y_true, y_pred)


#%%


# import pretty_confusion_matrix

# reload(pretty_confusion_matrix)

# df_cm = pd.DataFrame(
#     confusion_matrix(y_true, y_pred), index=target_names, columns=target_names
# )
# pretty_confusion_matrix.pp_matrix(
#     df_cm,
#     rotation=0,
#     cmap="binary",
#     vmin=-200,
#     vmax=700,
#     # cmap="binary_r",
#     # vmin=-200,
#     # vmax=1000,
# )

#%%

#%%


df_modern_not_1M = df_modern.query("N_reads_simulated != 1_000_000")
df_ancient_not_1M = df_ancient.query("N_reads_simulated != 1_000_000")
Xy_not_1M = get_Xy(df_modern_not_1M, df_ancient_not_1M)
df_predict_not_1M = blr_1M.predict(Xy_not_1M)


#%%

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    df_ancient_not_1M["Bayesian_z"],
    df_ancient_not_1M["Bayesian_D_max"],
    "o",
    color="green",
    alpha=0.5,
    label="Ancient",
)

ax.plot(
    df_modern_not_1M["Bayesian_z"],
    df_modern_not_1M["Bayesian_D_max"],
    "o",
    color="red",
    alpha=0.5,
    label="Modern",
)


ax.set(xlabel="Bayesian_z", ylabel="Bayesian_D_max")
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
ax.legend()

#%%


#%%

y_true_not_1M = Xy_not_1M["y"]

from sklearn.metrics import auc, roc_curve

df_predict_not_1M["CI_16"]


def compute_roc_stats(y_true, y_pred):
    FPR, TPR, threshold = roc_curve(y_true, y_pred)
    AUC = auc(FPR, TPR)
    return {"FPR": FPR, "TPR": TPR, "threshold": threshold, "AUC": AUC}


d_rocs = {
    "mean": compute_roc_stats(y_true_not_1M, df_predict_not_1M["mean"]),
    "CI_16": compute_roc_stats(y_true_not_1M, df_predict_not_1M["CI_16"]),
    # "mean_simple": compute_roc_stats(y_true_not_1M, df_predict_not_1M_simple["mean"]),
    # "CI16_simple": compute_roc_stats(y_true_not_1M, df_predict_not_1M_simple["CI_16"]),
}

#%%

import plotly.express as px
import plotly.graph_objects as go

#%%


# The histogram of scores compared to true labels
fig_hist = px.histogram(
    x=df_predict_not_1M["mean"],
    color=y_true_not_1M,
    nbins=50,
    labels=dict(color="True Labels", x="Score"),
    log_y=True,
    range_x=(0, 1),
)

fig_hist.show()


#%%


# The histogram of scores compared to true labels
fig_hist = px.histogram(
    x=df_predict_not_1M["CI_16"],
    color=y_true_not_1M,
    nbins=50,
    labels=dict(color="True Labels", x="Score"),
    log_y=True,
    range_x=(0, 1),
)

fig_hist.show()


#%%


# Evaluating model performance at various thresholds
df = pd.DataFrame(
    {
        "False Positive Rate": d_rocs["mean"]["FPR"],
        "True Positive Rate": d_rocs["mean"]["TPR"],
    },
    index=d_rocs["mean"]["threshold"],
)
df.index.name = "Thresholds"
df.columns.name = "Rate"

fig_thresh = px.line(df, title="TPR and FPR at every threshold", width=700, height=500)

fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
fig_thresh.update_xaxes(range=[0, 1], constrain="domain")
fig_thresh.show()

#%%


# Evaluating model performance at various thresholds
df = pd.DataFrame(
    {
        "False Positive Rate": d_rocs["CI_16"]["FPR"],
        "True Positive Rate": d_rocs["CI_16"]["TPR"],
    },
    index=d_rocs["CI_16"]["threshold"],
)
df.index.name = "Thresholds"
df.columns.name = "Rate"

fig_thresh = px.line(df, title="TPR and FPR at every threshold", width=700, height=500)

fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
fig_thresh.update_xaxes(range=[0, 1], constrain="domain")
fig_thresh.show()

#%%


# Create an empty figure, and iteratively add new lines
# every time we compute a new class
fig = go.Figure()
fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

for name, d_roc in d_rocs.items():
    AUC = d_roc["AUC"]
    name = f"{name} (AUC={AUC:.2%})"
    fig.add_trace(
        go.Scatter(
            x=d_roc["FPR"],
            y=d_roc["TPR"],
            name=name,
            mode="lines",
            customdata=d_roc["threshold"],
            hovertemplate="FPR = %{x:.3%}<br>TPR = %{y:.3%}<br>Threshold =%{customdata:.3%}",
        )
    )

fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    yaxis=dict(constrain="domain"),
    xaxis=dict(constrain="domain"),
    width=700,
    height=500,
)

# fig.update_layout(hovermode="x unified")

fig.show()
