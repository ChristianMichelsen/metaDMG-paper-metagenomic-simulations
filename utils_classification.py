#%%


import warnings

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

#%%

relevant_columns = [
    "tax_id",
    "tax_name",
    "tax_rank",
    "sample_name",
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
    "lambda_LR_P",
    "lambda_LR_z",
    "LR_All",
    "chi2_all",
    "N_x=1_forward",
    "N_sum_total",
    "N_sum_forward",
    "N_min",
    "k_sum_total",
    "k_sum_forward",
    "Bayesian_z",
    "Bayesian_D_max",
    "Bayesian_D_max_std",
    "Bayesian_D_max_median",
    "Bayesian_D_max_confidence_interval_1_sigma_low",
    "Bayesian_D_max_confidence_interval_1_sigma_high",
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
    "D_max_significance",
    "rho_Ac_abs",
    "Bayesian_D_max_significance",
    "Bayesian_rho_Ac_abs",
    "sample",
    "N_reads_simulated",
    "simulation_method",
    "simulated_seq_depth_ancient",
    "simulated_seq_depth_modern",
    "simulated_only_ancient",
    "simulated_D_max",
]


#%%



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
