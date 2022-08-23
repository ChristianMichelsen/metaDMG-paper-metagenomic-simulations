#%%

import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils
from human_reference import reference

#%%

name = "Pitch-6"
N_reads = 1_000_000

_ = utils.get_alignment_files(name=name, N_reads=str(N_reads))
path_alignment_frag, path_alignment_deam, path_alignment_art = _

path_fit = Path("data")
path_fit_analysis = path_fit / "analysis"
path_fit_analysis.mkdir(exist_ok=True)

#%%


path_alignment = path_alignment_frag

df_alignment_frag, df_mismatch_frag = utils.load_alignment_file(
    path_alignment_frag,
    feather_file=path_fit_analysis / f"{name}.{N_reads}.frag.feather",
    xarray_file=path_fit_analysis / f"{name}.{N_reads}.frag.mismatches.nc",
)
df_alignment_deam, df_mismatch_deam = utils.load_alignment_file(
    path_alignment_deam,
    feather_file=path_fit_analysis / f"{name}.{N_reads}.deam.feather",
    xarray_file=path_fit_analysis / f"{name}.{N_reads}.deam.mismatches.nc",
)
df_alignment_art, df_mismatch_art = utils.load_alignment_file(
    path_alignment_art,
    feather_file=path_fit_analysis / f"{name}.{N_reads}.art.feather",
    xarray_file=path_fit_analysis / f"{name}.{N_reads}.art.mismatches.nc",
)

#%%

df_fit_mismatch_frag = pd.read_parquet(
    path_fit / "mismatches" / f"{name}__fragSim__{N_reads}.mismatches.parquet"
)

df_fit_mismatch_deam = pd.read_parquet(
    path_fit / "mismatches" / f"{name}__deamSim__{N_reads}.mismatches.parquet"
)
df_fit_mismatch_art = pd.read_parquet(
    path_fit / "mismatches" / f"{name}__art__{N_reads}.mismatches.parquet"
)


#%%

df_fit_results_frag = pd.read_parquet(
    path_fit / "results" / f"{name}__fragSim__{N_reads}.results.parquet"
)
df_fit_results_deam = pd.read_parquet(
    path_fit / "results" / f"{name}__deamSim__{N_reads}.results.parquet"
)
df_fit_results_art = pd.read_parquet(
    path_fit / "results" / f"{name}__art__{N_reads}.results.parquet"
)


#%%

tax_id_homo = 134313
taxon_accession = "Homo_sapiens----NC_012920.1"

# df_alignment_frag.query(f"taxon_accession == '{taxon_accession}'")
df_homo_frag = df_alignment_frag.query(f"tax_id == {tax_id_homo}")
df_homo_deam = df_alignment_deam.query(f"tax_id == {tax_id_homo}")
# df_homo_art = df_alignment_art.query(f"tax_id == {tax_id_homo}")

# df_homo_deam["fragment_length"].min()
# df_homo_deam["fragment_length"].max()
# df_homo_deam["fragment_length"].mean()

# df_homo_deam.iloc[0]["seq"]

# df_homo_deam["accession"].unique()
# df_homo_deam["contig_num"].unique()
# df_homo_deam["ancient_modern"].unique()
# df_homo_deam["strand"].value_counts()

df_mismatch_deam.sel(tax_id=tax_id_homo).loc["k_CT"]
df_mismatch_deam.sel(tax_id=tax_id_homo).loc["N_C"]


df_fit_mismatch_deam.query(f"tax_id == '{tax_id_homo}' & position > 0")["CT"]
df_fit_mismatch_deam.query(f"tax_id == '{tax_id_homo}' & position > 0")["C"]


N_reads_alignment_deam = len(df_alignment_deam.query(f"tax_id == {tax_id_homo}"))
N_reads_lca_deam = df_fit_results_deam.query(f"tax_id == '{tax_id_homo}'")["N_reads"]
df_fit_results_deam.query(f"tax_id == '{tax_id_homo}'")["N_alignments"]


#%%

if False:
    lca_file = path_fit / "lca" / f"{name}__deamSim__{N_reads}.lca.txt.gz"
    df_lca_tax_id = pd.DataFrame(utils.read_lca_file(lca_file)[tax_id])

    len(df_lca_tax_id)
    df_lca_tax_id["N_alignments"].sum()

    df_lca_tax_id.lca.iloc[0]
    df_lca_tax_id.lca.str.contains("134313:s__Homo sapiens").sum()

    df_lca_tax_id.loc[
        ~df_lca_tax_id.lca.str.contains("134313:s__Homo sapiens")
    ].lca.unique()


#%%

df_fit_mismatch_deam.query(f"tax_id == '{tax_id_homo}'").loc[:, "AA":"TT"].sum(axis=1)


#%%

# LCA filtering analysis

if True:

    bam_file = (
        Path("input-data") / "data" / f"{name}__deamSim__{N_reads}.dedup.filtered.bam"
    )

    lca_csv_file = Path("data") / "analysis_lca" / f"{name}__deamSim__{N_reads}.lca.csv"
    lca_csv_file.parent.mkdir(exist_ok=True, parents=True)

    if lca_csv_file.exists():
        df_lca_stats_min_similarity_score = pd.read_csv(lca_csv_file)

    else:
        df_lca_stats_min_similarity_score = (
            utils.compute_lca_stats_min_similarity_score(
                bam_file=bam_file,
                df_alignment=df_alignment_deam,
            )
        )
        df_lca_stats_min_similarity_score.to_csv(lca_csv_file, index=False)

    fig, ax = plt.subplots(figsize=(16, 10))
    for tax_id, df_ in df_lca_stats_min_similarity_score.groupby("tax_id"):
        ax.plot(df_["fraction_lca_stat"].values)
    ax.set_xticks(
        np.arange(len(utils.MIN_SIMILARITY_SCORES)),
        labels=utils.MIN_SIMILARITY_SCORES,
    )
    ax.set(
        xlabel="min_similarity_score",
        ylabel="fraction of reads after LCA (stat)",
        title=f"{name}__deamSim__{N_reads} (stat)",
    )
    fig.savefig(
        f"figures/{name}__deamSim__{N_reads}__fraction_reads_after_lca_stats.pdf"
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    for tax_id, df_ in df_lca_stats_min_similarity_score.groupby("tax_id"):
        ax.plot(df_["fraction_lca_full"].values)
    ax.set_xticks(
        np.arange(len(utils.MIN_SIMILARITY_SCORES)),
        labels=utils.MIN_SIMILARITY_SCORES,
    )
    ax.set(
        xlabel="min_similarity_score",
        ylabel="fraction of reads after LCA (full)",
        title=f"{name}__deamSim__{N_reads} (full)",
    )
    fig.savefig(
        f"figures/{name}__deamSim__{N_reads}__fraction_reads_after_lca_full.pdf"
    )


#%%

# damage analysis

#%%

df_simulation = utils.extract_simulation_parameters(name, N_reads)


#%%

df_fit_results_cols = [
    "tax_id",
    "tax_name",
    "tax_rank",
    "sample",
    #
    "N_reads",
    "N_alignments",
    "mean_L",
    "mean_GC",
    "std_L",
    "std_GC",
    #
    "D_max",
    "D_max_std",
    #
    "Bayesian_D_max",
    "Bayesian_D_max_std",
    #
    "lambda_LR",
    "Bayesian_z",
    #
    "A",
    "A_std",
    "q",
    "q_std",
    "c",
    "c_std",
    "phi",
    "phi_std",
    #
    "rho_Ac",
    "valid",
    "asymmetry",
    #
    "N_x=1_forward",
    "N_x=1_reverse",
    "N_sum_total",
    "N_sum_forward",
    "N_sum_reverse",
    "N_min",
    "k_sum_total",
    "k_sum_forward",
    "k_sum_reverse",
    #
    "Bayesian_z",
    "Bayesian_D_max",
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


df_fit_results_frag.query(f"tax_id == '{tax_id_homo}'")
df_fit_results_deam.query(f"tax_id == '{tax_id_homo}'")
df_fit_results_art.query(f"tax_id == '{tax_id_homo}'")


#%%

## Pitch-6.communities_read-abundances.tsv
# comm	    taxon	                        frag_type	seq_depth	seq_depth_rounded
# Pitch-6	Homo_sapiens----NC_012920.1	    ancient	    20844	    108872.0
# Pitch-6	Homo_sapiens----NC_012920.1	    modern	    0	        0.0


## Pitch-6.genome-compositions.tsv
# Taxon	                        Community	Coverage	        Read_length	    Read_length_std	    Read_length_min	    Read_length_max	onlyAncient	    D_max
# Homo_sapiens----NC_012920.1	Pitch-6	    72.96559633027523	81	            16.263479571	    30	                81	            True	        0.14912868


## Pitch-6.communities.tsv
# Community	    Taxon	                        Rank	Read_length	    Read_length_std	    Read_length_min	    Read_length_max	    Perc_rel_abund
# Pitch-6	    Homo_sapiens----NC_012920.1	    1	    81	            16.2634795716	    30	                81	                23.169172


## Pitch-6.filepaths.tsv
# Taxon	                        TaxId	Accession	    Fasta
# Homo_sapiens----NC_012920.1	134313	NC_012920.1	    /maps/projects/lundbeck/scratch/eDNA/DBs/gtdb/r202/flavours/vanilla-organelles-virus/pkg/genomes/fasta/NC_012920.1_genomic.fna.gz


## Pitch-6.communities.json
# "comm": "Pitch-6",
# "taxon": "Homo_sapiens----NC_012920.1",
# "rel_abund": 23.169172561638344,
# "genome_size": 16569,
# "accession": "NC_012920.1",
# "onlyAncient": true,
# "fragments_ancient": {
#     "fragments": {
#         "length": [
#             30,
#             31,
#             76,
#             81
#         ],
#         "freq": [
#             0.005098572399728076,
#             0.007672137515781295,
#             0.008594736330970186,
#             0.24084684859667865
#         ]
#     },
#     "dist_params": {
#         "scale": null,
#         "sigma": null,
#         "rnd_seed": null
#     },
#     "avg_len": 58.695882295814314,
#     "seq_depth": 20844,
#     "seq_depth_original": 20844,
#     "fold": 72.96469310157524,
#     "fold_original": 72.96559633027523
# },
# "fragments_modern": null,
# "coverage_enforced": true,
# "seq_depth": 108872

#%%

# %%
