#%%
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

import plot_utils
import utils

# reload(plot_utils)


#%%

parser_comparison = parse.compile(
    "{sample}.{N_reads:Int}.{sim_name}.comparison",
    dict(Int=int),
)


def get_sample_N_reads_simulation_method(path):
    d = parser_comparison.parse(path.stem).named
    return d["sample"], d["N_reads"], d["sim_name"]


def load_df_comparisons():

    paths = sorted(Path("data/analysis_comparison").glob("*.csv"))

    dfs = []
    for path in paths:
        # break

        sample, N_reads, simulation_method = get_sample_N_reads_simulation_method(path)

        df_comparison = pd.read_csv(path).rename(
            columns={"N_reads": "N_reads_simulated"}
        )

        try:

            df_metaDMG_results = (
                utils.load_df_metaDMG_results_all(
                    sample,
                    N_reads,
                    simulation_methods=[simulation_method],
                )[simulation_method]
                .rename(columns={"sample": "sample_name"})
                .astype({"tax_id": int})
            )

        except FileNotFoundError:
            continue

        if not (
            "Bayesian_D_max_confidence_interval_1_sigma_low"
            in df_metaDMG_results.columns
        ):
            raise AssertionError("Expected this file to be found")

        drop_cols = [
            "tax_name",
            "D_max",
            "Bayesian_D_max",
            "lambda_LR",
            "Bayesian_z",
        ]

        if "f-15" in df_metaDMG_results.columns:
            drop_cols.extend(df_metaDMG_results.loc[:, "k+1":"f-15"].columns)
        else:
            drop_cols.extend(df_metaDMG_results.loc[:, "k+1":"f+15"].columns)

        tmp = df_metaDMG_results.drop(columns=drop_cols)
        df_comparison = pd.merge(df_comparison, tmp, on="tax_id")

        dfs.append(df_comparison)

    dfs = pd.concat(dfs).sort_values(
        by=["sample", "N_reads_simulated", "simulation_method", "|A|"],
        ascending=[True, True, False, False],
    )

    return dfs


# x=x


#%%


df_comparisons = load_df_comparisons()


#%%


#%%


reload(plot_utils)

plot_utils.plot_comparison_across_N_reads_simulated_and_sim_method(
    df_comparisons,
    use_bayesian=True,
)
plot_utils.plot_comparison_across_N_reads_simulated_and_sim_method(
    df_comparisons,
    use_bayesian=False,
)


#%%

# reload(plot_utils)

groups = df_comparisons.groupby(by=["sample", "N_reads_simulated", "simulation_method"])
for (sample, N_reads_simulated, simulation_method), df_comparison in tqdm(groups):
    plot_utils.plot_df_comparison_plt(
        df_comparison, sample, N_reads_simulated, simulation_method
    )


#%%

#%%

# cols = [
#     "N_reads",
#     "simulation_method",
#     "|A|",
#     "simulated_D_max",
#     "Bayesian_D_max",
#     "Bayesian_D_max_std",
#     "D_max",
#     "D_max_std",
# ]


x = x

#%%

#%%


#%%

fig_A_B = px.scatter(
    df_comparison,
    x="tax_id",
    y="|B|/|A|",
    hover_data=["tax_name", "tax_id", "|A|", "|B|"],
    title=f"{sample}, {N_reads} reads: |B| / |A|",
)
fig_A_B.show()

#%%

fig_B_C = px.scatter(
    df_comparison,
    x="tax_id",
    y="|C|/|B|",
    hover_data=["tax_name", "tax_id", "|B|", "|C|"],
    title=f"{sample}, {N_reads} reads: |C| / |B|",
)
fig_B_C.show()


#%%


fig_A_C = px.scatter(
    df_comparison,
    x="tax_id",
    y="|C|/|A|",
    hover_data=["tax_name", "tax_id", "|A|", "|C|"],
    title=f"{sample}, {N_reads} reads: |C| / |A|",
)
fig_A_C.show()


#%%


fig_CT_mismatch_vs_C = px.scatter(
    df_comparison,
    x="C) f_CT (x=1)",
    y="D) f_CT (x=1)",
    hover_data=[
        "tax_name",
        "tax_id",
        "C) k_CT (x=1)",
        "C) N_C (x=1)",
        "D) k_CT (x=1)",
        "D) N_C (x=1)",
    ],
    title="C->T: D vs C",
)
fig_CT_mismatch_vs_C.add_trace(px.line(y=[0, 1]).data[0])


fig_CT_mismatch_vs_C.update_xaxes(
    range=[0, 1],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
)

fig_CT_mismatch_vs_C.update_yaxes(
    range=[0, 1],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
)

fig_CT_mismatch_vs_C.show()


#%%


# Build figure
fig_comparison_CT = go.Figure()

kwargs = dict(
    mode="markers",
    # marker_size=10,
    # marker_line_color="black",
    marker_line_color="lightgrey",
    marker_line_width=0.5,
    showlegend=True,
)


_ = [
    ("A", "A) f_CT (x=1)", "#C00000 ", "circle"),
    ("B", "B) f_CT (x=1)", "#0070C0", "square"),
    ("C", "C) f_CT (x=1)", "#00B050", "diamond"),
    ("A\B", "A\B) f_CT (x=1)", "#802540", "hexagon"),
    ("B\C", "B\C) f_CT (x=1)", "#00859B", "star-square"),
    ("D", "D) f_CT (x=1)", "#FFC000", "star"),
]

for (name, variable, color, symbol) in _:

    customdata = np.stack(
        (
            df_comparison["tax_id"],
            df_comparison["tax_name"],
            df_comparison[f"|{name}|"],
            df_comparison[f"{name}) k_CT (x=1)"],
            df_comparison[f"{name}) N_C (x=1)"],
        )
    ).T

    # Add scatter trace with medium sized markers
    fig_comparison_CT.add_trace(
        go.Scatter(
            x=df_comparison["|A|"],
            y=100 * df_comparison[variable],
            name=name,
            marker_color=color,
            marker_symbol=symbol,
            marker_size=1 + 2 * np.log(1 + df_comparison[f"|{name}|"]),
            customdata=customdata,
            hovertemplate=""
            + "<br>"
            + "Tax ID = %{customdata[0]}"
            + "<br>"
            + "Tax name = %{customdata[1]}"
            + "<br>"
            + "<br>"
            + f"|{name}| "
            + "= %{customdata[2]}"
            + "<br>"
            + f"{name}) "
            + "k_CT (x=1) = %{customdata[3]}"
            + "<br>"
            + f"{name}) "
            + "N_C (x=1) = %{customdata[4]}"
            + "<br>"
            + f"{name}) "
            + "f_CT (x=1) = %{y:.1f}%"
            + "<br>",
            # + "<extra></extra>",
            **kwargs,
        )
    )

fig_comparison_CT.update_xaxes(type="log")

fig_comparison_CT.update_layout(
    title=f"{sample}, {N_reads} reads: f_CT (x=1)",
    xaxis_title="|A|",
    yaxis_title="f_CT (x=1)",
    autosize=True,
    width=1600 / 1.5,
    height=900 / 1.5,
)


fig_comparison_CT.show()

fig_name = (
    Path("figures") / f"{sample}.{N_reads}.{simulation_method}.comparison-CT.html"
)
fig_comparison_CT.write_html(fig_name)

# %%


# Build figure
fig_comparison_GA = go.Figure()

kwargs = dict(
    mode="markers",
    # marker_size=10,
    # marker_line_color="black",
    marker_line_color="lightgrey",
    marker_line_width=0.5,
    showlegend=True,
)


_ = [
    ("A", "A) f_GA (x=-1)", "#C00000 ", "circle"),
    ("B", "B) f_GA (x=-1)", "#0070C0", "square"),
    ("C", "C) f_GA (x=-1)", "#00B050", "diamond"),
    ("A\B", "A\B) f_GA (x=-1)", "#802540", "hexagon"),
    ("B\C", "B\C) f_GA (x=-1)", "#00859B", "star-square"),
    ("D", "D) f_GA (x=-1)", "#FFC000", "star"),
]

for (name, variable, color, symbol) in _:

    customdata = np.stack(
        (
            df_comparison["tax_id"],
            df_comparison["tax_name"],
            df_comparison[f"|{name}|"],
            df_comparison[f"{name}) k_GA (x=-1)"],
            df_comparison[f"{name}) N_G (x=-1)"],
        )
    ).T

    # Add scatter trace with medium sized markers
    fig_comparison_GA.add_trace(
        go.Scatter(
            x=df_comparison["|A|"],
            y=100 * df_comparison[variable],
            name=name,
            marker_color=color,
            marker_symbol=symbol,
            marker_size=1 + 2 * np.log(1 + df_comparison[f"|{name}|"]),
            customdata=customdata,
            hovertemplate=""
            + "<br>"
            + "Tax ID = %{customdata[0]}"
            + "<br>"
            + "Tax name = %{customdata[1]}"
            + "<br>"
            + "<br>"
            + f"|{name}| "
            + "= %{customdata[2]}"
            + "<br>"
            + f"{name}) "
            + "k_GA (x=-1) = %{customdata[3]}"
            + "<br>"
            + f"{name}) "
            + "N_G (x=-1) = %{customdata[4]}"
            + "<br>"
            + f"{name}) "
            + "f_GA (x=-1) = %{y:.1f}%"
            + "<br>",
            # + "<extra></extra>",
            **kwargs,
        )
    )

fig_comparison_GA.update_xaxes(type="log")

fig_comparison_GA.update_layout(
    title=f"{sample}, {N_reads} reads: f_GA (x=-1)",
    xaxis_title="|A|",
    yaxis_title="f_GA (x=-1)",
    autosize=True,
    width=1600 / 1.5,
    height=900 / 1.5,
)


fig_comparison_GA.show()

fig_name = (
    Path("figures") / f"{sample}.{N_reads}.{simulation_method}.comparison-GA.html"
)
fig_comparison_GA.write_html(fig_name)

# %%
