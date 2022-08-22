#%%

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

tax_id = 134313
taxon_accession = "Homo_sapiens----NC_012920.1"

# df_alignment_frag.query(f"taxon_accession == '{taxon_accession}'")
df_homo_frag = df_alignment_frag.query(f"tax_id == {tax_id}")
df_homo_deam = df_alignment_deam.query(f"tax_id == {tax_id}")
df_homo_art = df_alignment_art.query(f"tax_id == {tax_id}")

# df_homo_deam["fragment_length"].min()
# df_homo_deam["fragment_length"].max()
# df_homo_deam["fragment_length"].mean()

# df_homo_deam.iloc[0]["seq"]

# df_homo_deam["accession"].unique()
# df_homo_deam["contig_num"].unique()
# df_homo_deam["ancient_modern"].unique()
# df_homo_deam["strand"].value_counts()


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
