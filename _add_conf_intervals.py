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

        path_comparison = (
            Path("data")
            / "analysis_comparison"
            / f"{sample}.{N_reads}.{simulation_method}.comparison.csv"
        )
        df_comparison = pd.read_csv(path_comparison)

        dfs.append(df)

    df_results = pd.concat(dfs)

    df_results["D_max_significance"] = df_results["D_max"] / df_results["D_max_std"]
    df_results["rho_Ac_abs"] = np.abs(df_results["rho_Ac"])

    df_results["Bayesian_D_max_significance"] = (
        df_results["Bayesian_D_max"] / df_results["Bayesian_D_max_std"]
    )
    df_results["Bayesian_rho_Ac_abs"] = np.abs(df_results["Bayesian_rho_Ac"])

    return df_results
