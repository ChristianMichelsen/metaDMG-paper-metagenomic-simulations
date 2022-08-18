#%%
import importlib
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from metaDMG.fit import serial
from metaDMG.utils import make_configs

#%%
config_file = Path("config_small.yaml")
configs = make_configs(config_file)
config = configs.get_nth(0)


config["cores"] = 1
# config["bayesian"] = True
# config["bayesian"] = False

# force = True
force = False

#%%

x = x  # type: ignore

#%%

serial.run_cpp(config, force=force)
df_mismatches = serial.get_df_mismatches(config, force=force)
df_fit_results = serial.get_df_fit_results(config, df_mismatches, force=force)
df_results = serial.get_df_results(config, df_mismatches, df_fit_results, force=force)


#%%

cols = [
    "tax_id",
    "tax_name",
    "tax_rank",
    "sample",
    "N_reads",
    "N_alignments",
    "D_max",
    "Bayesian_D_max",
    "lambda_LR",
    "Bayesian_z",
    "Bayesian_phi",
    # "D_max_significance",
    # "Bayesian_D_max_significance",
]

df_mismatches.query("tax_id == '131028'")
df_results.loc[:, cols].query("tax_id == '131028'").T
# mammals = df_results.query("tax_id == '282'")

#%%

df_results["D_max_significance"] = df_results["D_max"] / df_results["D_max_std"]
df_results["Bayesian_D_max_significance"] = (
    df_results["Bayesian_D_max"] / df_results["Bayesian_D_max_std"]
)


df_results.sort_values("D_max", ascending=False).loc[:, cols]


# %%

for tax_id, group in serial.fits.get_groupby(df_mismatches):
    if tax_id == "113490":
        break


#%%

from metaDMG.fit import bayesian, fits

# Do not initialise MCMC if config["bayesian"] is False
mcmc_PMD, mcmc_null = bayesian.init_mcmcs(config)


d_fit_results = {}
d_fit_results[tax_id] = fits.fit_single_group(
    config,
    group,
    mcmc_PMD,
    mcmc_null,
)


#%%

from metaDMG.fit.fits import group_to_numpyro_data

fit_result = {}
data = group_to_numpyro_data(config, group)  # type: ignore
sample = config["sample"]

#%%

from metaDMG.fit import frequentist

fit_all = frequentist.Frequentist(data, sample, tax_id, method="posterior")

# reload(frequentist)

PMD = frequentist.FrequentistPMD(
    data,
    sample,
    tax_id,
    method="posterior",
    # p0={"q": 0.0, "A": 0.0, "c": 0.01, "phi": 100},
    verbose=True,
)
PMD.p0
PMD.fit()
PMD


#%%

fit_result = {}
frequentist.make_forward_reverse_fits(fit_result, data, sample, tax_id)


np.random.seed(42)


fit_all = frequentist.Frequentist(data, sample, tax_id, method="posterior")


data_forward = {key: val[data["x"] > 0] for key, val in data.items()}
data_reverse = {key: val[data["x"] < 0] for key, val in data.items()}


# fit_forward = frequentist.Frequentist(
#     data_forward,
#     sample,
#     tax_id,
#     method="posterior",
#     p0=fit_all.PMD_values,
#     # p0 = {"q": 0.1, "A": 0.1, "c": 0.01, "phi": 1000},
#     verbose = True,
# )


fit_reverse = frequentist.Frequentist(
    data_reverse,
    sample,
    tax_id,
    method="posterior",
    p0=fit_all.PMD_values,
)

#%%

from metaDMG.fit import bayesian

config["bayesian"] = True
mcmc_PMD, mcmc_null = bayesian.init_mcmcs(config)

bayesian.fit_mcmc(mcmc_PMD, data)
bayesian.fit_mcmc(mcmc_null, data)


# mcmc = mcmc_PMD
# mcmc = mcmc_null

d_results_PMD = bayesian.get_lppd_and_waic(mcmc_PMD, data)
d_results_null = bayesian.get_lppd_and_waic(mcmc_null, data)
z = bayesian.compute_z(d_results_PMD, d_results_null)

bayesian.compute_D_max(mcmc_PMD, data)

# %%


np.random.seed(42)

fit_all = frequentist.Frequentist(data, sample, tax_id, method="posterior")

data_forward = {key: val[data["x"] > 0] for key, val in data.items()}
data_reverse = {key: val[data["x"] < 0] for key, val in data.items()}

fit_forward = frequentist.Frequentist(
    data_forward,
    sample,
    tax_id,
    method="posterior",
    p0=fit_all.PMD_values,
)


PMD = frequentist.FrequentistPMD(
    data_forward,
    sample,
    tax_id,
    method="posterior",
    p0=fit_all.PMD_values,
    # p0 = {"q": 0.1, "A": 0.1, "c": 0.01, "phi": 1000},
    verbose=True,
)
PMD.p0
PMD.fit()
PMD


PMD.m.migrad()
PMD.is_fitted = True

# First try to refit it
if not PMD.m.valid:
    if PMD.verbose:
        print("refitting A")
    for i in range(10):
        PMD.m.migrad()
        if PMD.m.valid:
            if PMD.verbose:
                print(f"Got out, A {i}")
            break

# Then try with a totally flat guess
if not PMD.m.valid:
    if PMD.verbose:
        print("refitting B")
    p0_flat = {"q": 0.0, "A": 0.0, "c": 0.01, "phi": 100}
    PMD._setup_p0(p0_flat)
    for key, val in p0_flat.items():
        PMD.m.values[key] = val
    PMD.m.migrad()
    if PMD.verbose:
        print(f"Got out, B")

# Also try with the default guess
if not PMD.m.valid:
    if PMD.verbose:
        print("refitting B2")
    p0_flat = {"q": 0.1, "A": 0.1, "c": 0.01, "phi": 1000}
    PMD._setup_p0(p0_flat)
    PMD.m.migrad()
    if PMD.verbose:
        print(f"Got out, B2")
