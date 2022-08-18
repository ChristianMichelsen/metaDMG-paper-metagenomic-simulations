from pathlib import Path

import numpy as np
import pandas as pd

path = Path("data") / "results"

df_results = pd.read_parquet(path)


cols = [
    "tax_id",
    "tax_name",
    "tax_rank",
    "sample",
    "N_reads",
    "N_alignments",
    "lambda_LR",
    "D_max",
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
    "tax_path",
    "D_max_std",
    "q_std",
    "phi_std",
    "A_std",
    "c_std",
    "N_x=1_forward",
    "N_x=1_reverse",
    "N_sum_total",
    "N_sum_forward",
    "N_sum_reverse",
    "N_min",
    "k_sum_total",
    "k_sum_forward",
    "k_sum_reverse",
    "Bayesian_z",
    "Bayesian_D_max",
    "Bayesian_D_max_std",
    "Bayesian_A",
    "Bayesian_A_std",
    "Bayesian_q",
    "Bayesian_q_std",
    "Bayesian_c",
    "Bayesian_c_std",
    "Bayesian_phi",
    "Bayesian_phi_std",
    "Bayesian_rho_Ac",
]

df_results = df_results.loc[:, cols]


df_results["D_max_significance"] = df_results["D_max"] / df_results["D_max_std"]
df_results["rho_Ac_abs"] = np.abs(df_results["rho_Ac"])


df_results_Pitch_art = df_results.loc[
    df_results["sample"].str.contains("Pitch-6__art__")
]

for tax_id, group in df_results_Pitch_art.groupby("tax_id"):
    break
