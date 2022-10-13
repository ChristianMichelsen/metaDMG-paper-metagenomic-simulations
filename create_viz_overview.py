#%%

from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import seaborn as sns
from tqdm import tqdm

import utils

#%%


# columns =

#%%


parser_template = "{sample}__{simulation_method:SimName}__{simulated_N_reads:Int}"
parser = parse.compile(
    parser_template,
    dict(
        Int=int,
        SimName=lambda s: utils.fix_sim_name(s, utils.D_SIM_NAME_TRANSLATE),
    ),
)


sim_columns = ["sample", "simulation_method", "simulated_N_reads"]


def split_simulation_name(simulation_name):
    result = parser.parse(simulation_name).named
    return result["sample"], result["simulation_method"], result["simulated_N_reads"]


def split_name_pd(name):
    return pd.Series(split_simulation_name(name), index=sim_columns)


def load_results():

    df = (
        pd.read_parquet("data/results/")
        .reset_index(drop=True)
        .rename(columns={"sample": "simulation_name"})
        .astype({"tax_id": int})
    ).copy()

    df["Bayesian_significance"] = df["Bayesian_D_max"] / df["Bayesian_D_max_std"]
    df["Bayesian_prob_not_zero_damage"] = 1 - df["Bayesian_prob_zero_damage"]
    df["Bayesian_prob_gt_1p_damage"] = 1 - df["Bayesian_prob_lt_1p_damage"]
    df[sim_columns] = df["simulation_name"].apply(split_name_pd)

    columns = [
        "simulation_name",
        "sample",
        "simulation_method",
        "simulated_N_reads",
        "tax_id",
        "tax_name",
        "tax_rank",
        "N_reads",
        "D_max",
        "D_max_std",
        "Bayesian_D_max",
        "Bayesian_D_max_std",
        "significance",
        "Bayesian_significance",
        "Bayesian_prob_not_zero_damage",
        "Bayesian_prob_gt_1p_damage",
        "mean_L",
        "mean_GC",
        "q",
        "A",
        "c",
        "phi",
        "rho_Ac",
        "valid",
        "asymmetry",
        "std_L",
        "std_GC",
        "lambda_LR",
        "q_std",
        "phi_std",
        "A_std",
        "c_std",
        "N_x=1_forward",
        # "N_x=1_reverse",
        "N_sum_total",
        "N_sum_forward",
        # "N_sum_reverse",
        "N_min",
        "k_sum_total",
        "k_sum_forward",
        # "k_sum_reverse",
        "Bayesian_D_max_median",
        "Bayesian_D_max_confidence_interval_1_sigma_low",
        "Bayesian_D_max_confidence_interval_1_sigma_high",
        "Bayesian_D_max_confidence_interval_2_sigma_low",
        "Bayesian_D_max_confidence_interval_2_sigma_high",
        "Bayesian_D_max_confidence_interval_3_sigma_low",
        "Bayesian_D_max_confidence_interval_3_sigma_high",
        "Bayesian_D_max_confidence_interval_95_low",
        "Bayesian_D_max_confidence_interval_95_high",
        "Bayesian_A",
        "Bayesian_A_std",
        "Bayesian_A_median",
        "Bayesian_A_confidence_interval_1_sigma_low",
        "Bayesian_A_confidence_interval_1_sigma_high",
        "Bayesian_q",
        "Bayesian_q_std",
        "Bayesian_q_median",
        "Bayesian_q_confidence_interval_1_sigma_low",
        "Bayesian_q_confidence_interval_1_sigma_high",
        "Bayesian_c",
        "Bayesian_c_std",
        "Bayesian_c_median",
        "Bayesian_c_confidence_interval_1_sigma_low",
        "Bayesian_c_confidence_interval_1_sigma_high",
        "Bayesian_phi",
        "Bayesian_phi_std",
        "Bayesian_phi_median",
        "Bayesian_phi_confidence_interval_1_sigma_low",
        "Bayesian_phi_confidence_interval_1_sigma_high",
        "Bayesian_rho_Ac",
        # "Bayesian_significance",
        "var_L",
        "var_GC",
        # "f+1",
        # "f+15",
        # "f-1",
        # "f-15",
    ]

    df = df.loc[:, columns]

    return df


#%%

df_results = load_results()
df_results_species = df_results.query("tax_rank == 'species'")


#%%

reload(utils)

directory = Path("input-data") / "data-pre-mapping"
df_simulation = utils.get_simulation_details(directory)


#%%

# x = x


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


df_good = df_results_species.query("sample in @good_samples")


dfs_ancient = []
dfs_modern = []
dfs_non_simulation = []

for (sample, simulated_N_reads), group in df_good.groupby(
    ["sample", "simulated_N_reads"]
):
    # break

    if "forward" in sample:
        sample = sample.replace("-forward", "")

    query = f"sample == '{sample}' and simulated_N_reads == {simulated_N_reads}"
    df_simulation_group = df_simulation.query(query)
    drop_cols = ["sample", "simulated_N_reads"]

    # ancient

    query = "simulated_only_ancient == 'True'"
    df_simulation_group_ancient = df_simulation_group.query(query).copy()
    df_simulation_group_ancient["type"] = "Ancient"

    dfs_ancient.append(
        group.merge(
            df_simulation_group_ancient.drop(columns=drop_cols),
            on="tax_id",
        )
    )

    # modern

    query = "simulated_only_ancient != 'True'"
    df_simulation_group_modern = df_simulation_group.query(query).copy()
    df_simulation_group_modern["type"] = "Non-ancient"

    dfs_modern.append(
        group.merge(
            df_simulation_group_modern.drop(columns=drop_cols),
            on="tax_id",
        ),
    )

    # non-simulation

    tax_ids_non_simulation = set(group.tax_id) - set(df_simulation_group.tax_id)
    df_non_simulation = group.query(f"tax_id in @tax_ids_non_simulation").copy()
    df_non_simulation["type"] = "Non-simulated"
    dfs_non_simulation.append(df_non_simulation)


df_ancient = pd.concat(dfs_ancient).reset_index(drop=True)
df_modern = pd.concat(dfs_modern).reset_index(drop=True)
df_non_simulation = pd.concat(dfs_non_simulation).reset_index(drop=True)

#%%

df_all = pd.concat([df_ancient, df_modern, df_non_simulation], axis=0)

#%%

import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages


def plot_overview(
    df_in,
    title="",
    xlims=None,
    ylims=None,
):

    fig, axes = plt.subplots(figsize=(18, 5), ncols=3, sharey=True)

    types = df_all["type"].unique()
    colors = [f"C{i}" for i in range(10)]
    samples = df_all["sample"].unique()

    for i_type, (type_, ax) in enumerate(zip(types, axes)):
        # break

        df_damage_type = df_in.query("type == @type_")

        for i_sample, sample in enumerate(samples):

            group = df_damage_type.query("sample == @sample")

            ax.scatter(
                group["Bayesian_significance"],
                group["Bayesian_D_max"],
                s=2 + np.sqrt(group["N_reads"]) / 5,
                alpha=0.5,
                color=colors[i_sample],
                label=sample,
            )

        ax.set(
            title=type_,
            xlabel="Bayesian significance",
            ylabel="Bayesian D_max",
            xlim=xlims[i_type] if xlims is not None else None,
            ylim=ylims[i_type] if ylims is not None else None,
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if i_type > 0:
            ax.yaxis.set_tick_params(labelbottom=True)
            leg = ax.legend(markerscale=5)
            for handle in leg.legendHandles:
                handle.set_sizes([30.0])
                handle.set_alpha(1)

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.85)

    return fig


#%%

simulation_methods = ["art", "deam", "frag"]

filename = f"figures/overview_bayesian.pdf"
with PdfPages(filename) as pdf:

    for simulation_method in simulation_methods:

        fig = plot_overview(
            df_all.query(f"simulation_method == '{simulation_method}'"),
            title="Simulation method: " + simulation_method,
            xlims=[(-0.1, 26), (0, 2), (0, 4.1)],
            ylims=[(-0.01, 0.7), (-0.01, 0.7), (-0.01, 0.7)],
        )

        pdf.savefig(fig)
        plt.close()

#%%

# df_modern.sort_values("Bayesian_significance", ascending=False).head(10)
# df_non_simulation.sort_values("Bayesian_significance", ascending=False).head(10)
# df_results.query("tax_id == 134927")
