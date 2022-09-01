#%%

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import plotly.express as px
import plotly.graph_objects as go
from engineering_notation import EngNumber
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

#%%

plt.rcParams["font.size"] = "16"


#%%


def plot_single_group(group, tax_id, sample, N_reads, simulation_method):

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))

    delta = 0.15

    markersize = 12

    ax1.plot(
        1 - delta,
        group["A) f_CT (x=1)"],
        color="#C00000",
        marker="o",
        markersize=5 * np.log10(1 + group["A) N_C (x=1)"]),
        # label="A) f_CT",
        linestyle="None",
    )
    ax1.plot(
        1 + delta,
        group["A) f_GA (x=-1)"],
        color="#C00000",
        marker="^",
        markersize=5 * np.log10(1 + group["A) N_G (x=-1)"]),
        # label="A) f_GA",
        linestyle="None",
    )

    ax1.plot(
        2 - delta,
        group["A\B) f_CT (x=1)"],
        color="#802540",
        marker="o",
        markersize=5 * np.log10(1 + group["A\B) N_C (x=1)"]),
        # label="A\B) f_CT",
        linestyle="None",
    )
    ax1.plot(
        2 + delta,
        group["A\B) f_GA (x=-1)"],
        color="#802540",
        marker="^",
        markersize=5 * np.log10(1 + group["A\B) N_G (x=-1)"]),
        # label="A\B) f_GA",
        linestyle="None",
    ),

    ax1.plot(
        3 - delta,
        group["B) f_CT (x=1)"],
        color="#0070C0",
        marker="o",
        markersize=5 * np.log10(1 + group["B) N_C (x=1)"]),
        # label="B) f_CT",
        linestyle="None",
    )
    ax1.plot(
        3 + delta,
        group["B) f_GA (x=-1)"],
        color="#0070C0",
        marker="^",
        markersize=5 * np.log10(1 + group["B) N_G (x=-1)"]),
        # label="B) f_GA",
        linestyle="None",
    )

    ax1.plot(
        4 - delta,
        group["B\C) f_CT (x=1)"],
        color="#00859B",
        marker="o",
        markersize=5 * np.log10(1 + group["B\C) N_C (x=1)"]),
        # label="B\C) f_CT",
        linestyle="None",
    )
    ax1.plot(
        4 + delta,
        group["B\C) f_GA (x=-1)"],
        color="#00859B",
        marker="^",
        markersize=5 * np.log10(1 + group["B\C) N_G (x=-1)"]),
        # label="B\C) f_GA",
        linestyle="None",
    )

    ax1.plot(
        5 - delta,
        group["C) f_CT (x=1)"],
        color="#00B050",
        marker="o",
        markersize=5 * np.log10(1 + group["C) N_C (x=1)"]),
        # label="C) f_CT",
        linestyle="None",
    )
    ax1.plot(
        5 + delta,
        group["C) f_GA (x=-1)"],
        color="#00B050",
        marker="^",
        markersize=5 * np.log10(1 + group["C) N_G (x=-1)"]),
        # label="C) f_GA",
        linestyle="None",
    )

    ax1.plot(
        6 - delta,
        group["D) f_CT (x=1)"],
        color="#FFC000",
        marker="o",
        markersize=5 * np.log10(1 + group["D) N_C (x=1)"]),
        # label="D) f_CT",
        linestyle="None",
    )
    ax1.plot(
        6 + delta,
        group["D) f_GA (x=-1)"],
        color="#FFC000",
        marker="^",
        markersize=5 * np.log10(1 + group["D) N_G (x=-1)"]),
        # label="D) f_GA",
        linestyle="None",
    )

    ax1.plot(
        0,
        0,
        color="k",
        marker="o",
        markersize=10,
        linestyle="None",
        label="C → T",
    )

    ax1.plot(
        0,
        0,
        color="k",
        marker="^",
        markersize=10,
        linestyle="None",
        label="G → A",
    )

    ax1.set(
        ylabel="Fraction",
        ylim=(0, ax1.get_ylim()[1] * 1.1),
        xlim=(0.5, 6.5),
    )
    ax1.set_xticks([1, 2, 3, 4, 5, 6], labels=["A", "A\B", "B", "B\C", "C", "D"])

    width = 0.15
    ax2.bar(
        1 - delta,
        height=group["A) N_C (x=1)"],
        color="#C00000",
        width=width,
    )
    ax2.bar(
        1 + delta,
        height=group["A) N_G (x=-1)"],
        color="#C00000",
        width=width,
    )

    ax2.bar(
        2 - delta,
        height=group["A\B) N_C (x=1)"],
        color="#802540",
        width=width,
    )
    ax2.bar(
        2 + delta,
        height=group["A\B) N_G (x=-1)"],
        color="#802540",
        width=width,
    )

    ax2.bar(
        3 - delta,
        height=group["B) N_C (x=1)"],
        color="#0070C0",
        width=width,
    )
    ax2.bar(
        3 + delta,
        height=group["B) N_G (x=-1)"],
        color="#0070C0",
        width=width,
    )

    ax2.bar(
        4 - delta,
        height=group["B\C) N_C (x=1)"],
        color="#00859B",
        width=width,
    )
    ax2.bar(
        4 + delta,
        height=group["B\C) N_G (x=-1)"],
        color="#00859B",
        width=width,
    )

    ax2.bar(
        5 - delta,
        height=group["C) N_C (x=1)"],
        color="#00B050",
        width=width,
    )
    ax2.bar(
        5 + delta,
        height=group["C) N_G (x=-1)"],
        color="#00B050",
        width=width,
    )

    ax2.bar(
        6 - delta,
        height=group["D) N_C (x=1)"],
        color="#FFC000",
        width=width,
    )
    ax2.bar(
        6 + delta,
        height=group["D) N_G (x=-1)"],
        color="#FFC000",
        width=width,
    )

    ax2.set(
        ylabel="Counts (N)",
        ylim=(0, ax2.get_ylim()[1] * 1.1),
        xlim=(0.5, 6.5),
    )

    ax2.set_xticks([1, 2, 3, 4, 5, 6], labels=["A", "A\B", "B", "B\C", "C", "D"])

    ax1.legend(
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.1, 1.5),
    )

    N_ancient = EngNumber(int(group.simulated_seq_depth_ancient))
    N_modern = EngNumber(int(group.simulated_seq_depth_modern))

    s = (
        f"Only ancient: {group.simulated_only_ancient}"
        "\n"
        f"Ancient:Modern = {N_ancient} : {N_modern}"
    )

    ax1.text(
        -0.1,
        1.26,
        s,
        horizontalalignment="left",
        transform=ax1.transAxes,
        fontsize=10,
    )

    title = (
        f"{sample}, {N_reads}, {simulation_method}"
        "\n"
        f"tax_id={tax_id}"
        "\n"
        f"tax_name={group['tax_name']}"
    )
    fig.suptitle(
        title,
        fontsize=16,
    )
    fig.subplots_adjust(
        top=0.83,
    )

    return fig


# fig = plot_single_group(group, tax_id, sample, N_reads, simulation_method)


#%%


def plot_df_comparison_plt(
    df_comparison,
    sample,
    N_reads,
    simulation_method,
    use_tqdm=False,
):

    fig_name = (
        Path("figures")
        / f"{sample}.{N_reads}.{simulation_method}.comparison-individual.pdf"
    )

    groupby = df_comparison.groupby("tax_id", sort=False)
    if use_tqdm:
        groupby = tqdm(groupby)

    with PdfPages(fig_name) as pdf:

        for tax_id, group in groupby:
            # break

            if len(group) != 1:
                raise ValueError(f"{tax_id} has {len(group)} rows")

            group = group.iloc[0]

            fig = plot_single_group(group, tax_id, sample, N_reads, simulation_method)

            pdf.savefig(fig)
            plt.close()

            # break
