#%%

import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from parse import compile
from tqdm import tqdm

from human_reference import reference

# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# from matplotlib.ticker import MaxNLocator, StrMethodFormatter
# from matplotlib.colors import LogNorm
# import matplotlib as mpl
# from matplotlib.backends.backend_pdf import PdfPages


#%%


def nth_repl(s, old, new, n):
    """helper function to find the nth occurrence of a substring `old` in the
    original string `s` and replace it with `new`."""
    find = s.find(old)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop until we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(old, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n and i <= len(s.split(old)) - 1:
        return s[:find] + new + s[find + len(old) :]
    return s


# "Abiotrophia_sp001815865----GCF_001815865.1"

parser_template = (
    "{sample}___"
    "{taxon}____"
    "{accession:Accession}__"
    "{contig_num}---"
    "{read_num:Int}:"
    "{ancient_modern}:"
    "{strand}:"
    "{reference_start:Int}:"
    "{reference_end:Int}:"
    "{fragment_length:Int}:"
    "{damaged_positions_in_fragment:Fragment}"
)


def fix_accession(accession):
    accession_fixed = nth_repl(accession, "_", ".", 2)
    return accession_fixed


def fragment_parser(damaged_positions):
    if damaged_positions == "None":
        return []
    elif "," not in damaged_positions:
        # return str([int(damaged_positions)])
        return [int(damaged_positions)]
    else:
        # return str([int(string_pos) for string_pos in damaged_positions.split(",")])
        return [int(string_pos) for string_pos in damaged_positions.split(",")]


parser = compile(
    parser_template,
    dict(Accession=fix_accession, Int=int, Fragment=fragment_parser),
)


#%%


def strip(str_: str) -> str:
    """
    :param str_: a string
    """
    return str_.strip()


def load_names(names_file: str | Path) -> pd.DataFrame:
    """
    load names.dmp and convert it into a pandas.DataFrame.
    Taken from https://github.com/zyxue/ncbitax2lin/blob/master/ncbitax2lin/data_io.py
    """

    df_data = pd.read_csv(
        names_file,
        sep="|",
        header=None,
        index_col=False,
        names=["tax_id", "name_txt", "unique_name", "name_class"],
    )

    return (
        df_data.assign(
            name_txt=lambda df: df["name_txt"].apply(strip),
            unique_name=lambda df: df["unique_name"].apply(strip),
            name_class=lambda df: df["name_class"].apply(strip),
        )
        .loc[lambda df: df["name_class"] == "scientific name"]
        .reset_index(drop=True)
    )


names_file = Path("names.dmp")
df_names = load_names(names_file)
df_names


def load_nodes(nodes_file: str | Path) -> pd.DataFrame:
    """
    load nodes.dmp and convert it into a pandas.DataFrame
    """

    df_data = pd.read_csv(
        nodes_file,
        sep="|",
        header=None,
        index_col=False,
        names=["tax_id", "parent_tax_id", "rank"],
    )

    return df_data.assign(rank=lambda df: df["rank"].apply(strip))


nodes_file = Path("nodes.dmp")
df_nodes = load_nodes(nodes_file)


def load_acc2taxid(acc2taxid_file: str | Path) -> pd.DataFrame:
    """
    load acc2taxid.map and convert it into a pandas.DataFrame
    """
    df_data = pd.read_csv(acc2taxid_file, sep="\t")
    return df_data


acc2taxid_file = Path("acc2taxid.map.gz")
df_acc2taxid = load_acc2taxid(acc2taxid_file)


def get_key2val_dict(df_acc2taxid, key_col, val_col):
    return df_acc2taxid[[key_col, val_col]].set_index(key_col).to_dict()[val_col]


def propername(name):
    return "_".join(name.split(" "))


d_acc2taxid = get_key2val_dict(df_acc2taxid, "accession", "taxid")
d_taxid2name = get_key2val_dict(df_names, "tax_id", "name_txt")


def acc2name(accesion):
    taxid = d_acc2taxid[accesion]
    name = d_taxid2name[taxid]
    return propername(name)


#%%


path_pitch6 = Path("input-data") / "data-pre-mapping" / "Pitch-6" / "single"
path_pitch6_1M = path_pitch6 / "1000000"

path_pitch6_1M_fragSim = path_pitch6_1M / "reads" / "Pitch-6_fragSim.fa.gz"
path_pitch6_1M_deamSim = path_pitch6_1M / "reads" / "Pitch-6_deamSim.fa.gz"
path_pitch6_1M_art = path_pitch6_1M / "reads" / "Pitch-6_art.fq.gz"


def load_alignment_file(alignment_file, feather_file):

    try:
        return pd.read_feather(feather_file)
    except FileNotFoundError:
        pass

    if ".fq" in alignment_file.name:
        alignment_type = "fastq"
    elif ".fa" in alignment_file.name:
        alignment_type = "fasta"
    else:
        raise AssertionError(f"Unknown alignment type: {alignment_file.suffix}")

    results = []
    with gzip.open(alignment_file, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, alignment_type)):
            # break
            name = record.name
            seq = record.seq
            result = parser.parse(name).named
            result["seq"] = str(seq)
            result["tax_id"] = d_acc2taxid[result["accession"]]
            result["taxon_accession"] = f"{result['taxon']}----{result['accession']}"
            results.append(result)

    df = pd.DataFrame(results)

    categories = [
        "sample",
        "taxon",
        "accession",
        "contig_num",
        "ancient_modern",
        "strand",
        "tax_id",
        "taxon_accession",
    ]
    for category in categories:
        df[category] = df[category].astype("category")

    df.to_feather(feather_file)

    return df


#%%

df_pitch6_1M_fragSim = load_alignment_file(
    path_pitch6_1M_fragSim, Path("data") / "pitch6_1M_fragSim.feather"
)
df_pitch6_1M_deamSim = load_alignment_file(
    path_pitch6_1M_deamSim, Path("data") / "pitch6_1M_deamSim.feather"
)
df_pitch6_1M_art = load_alignment_file(
    path_pitch6_1M_art, Path("data") / "pitch6_1M_art.feather"
)

#%%

tax_id = 134313
taxon_accession = "Homo_sapiens----NC_012920.1"

# df_pitch6_1M_fragSim.query(f"taxon_accession == '{taxon_accession}'")
df_homo_frag = df_pitch6_1M_fragSim.query(f"tax_id == {tax_id}")
df_homo_deam = df_pitch6_1M_deamSim.query(f"tax_id == {tax_id}")
df_homo_art = df_pitch6_1M_art.query(f"tax_id == {tax_id}")


df_homo_deam["fragment_length"].min()
df_homo_deam["fragment_length"].max()
df_homo_deam["fragment_length"].mean()

df_homo_deam.iloc[0]["seq"]

# df_homo_deam["accession"].unique()
# df_homo_deam["contig_num"].unique()
# df_homo_deam["ancient_modern"].unique()
# df_homo_deam["strand"].value_counts()


df_homo_deam_damaged = df_homo_deam[
    df_homo_deam["damaged_positions_in_fragment"].astype(bool)
]

df_homo_deam_damaged_pos = df_homo_deam_damaged.query("strand == '+'")

series = df_homo_deam_damaged_pos.iloc[3]
series

k_CT = np.zeros(15, dtype=int)
N_C = np.zeros(15, dtype=int)

k_GA = np.zeros(15, dtype=int)
N_G = np.zeros(15, dtype=int)


seq = Seq(series.seq)
ref = Seq(reference[series.reference_start : series.reference_end])

print(seq)
print(ref)

if series.strand == "+":

    for pos in series.damaged_positions_in_fragment:

        if abs(pos) >= 15:
            continue

        if pos > 0:
            k_CT[pos - 1] += 1
            N_C[pos - 1] += 1
        elif pos < 0:
            pos = abs(pos)
            k_GA[pos - 1] += 1
            N_G[pos - 1] += 1
        else:
            raise AssertionError("pos == 0")

    for i, base in enumerate(seq):
        if i >= 15:
            break

        if base == "C":
            N_C[i] += 1

    for i, base in enumerate(reversed(seq)):
        if i >= 15:
            break

        if base == "G":
            N_G[i] += 1


# if series.strand == "-":
#     seq = seq.reverse_complement()


#%%


# comm	    taxon	                        frag_type	seq_depth	seq_depth_rounded
# Pitch-6	Homo_sapiens----NC_012920.1	    ancient	    20844	    108872.0
# Pitch-6	Homo_sapiens----NC_012920.1	    modern	    0	        0.0

# Taxon	                        Community	Coverage	        Read_length	    Read_length_std	    Read_length_min	    Read_length_max	onlyAncient	    D_max
# Homo_sapiens----NC_012920.1	Pitch-6	    72.96559633027523	81	            16.263479571	    30	                81	            True	        0.14912868


# Community	    Taxon	                        Rank	Read_length	    Read_length_std	    Read_length_min	    Read_length_max	    Perc_rel_abund
# Pitch-6	    Homo_sapiens----NC_012920.1	    1	    81	            16.2634795716	    30	                81	                23.169172

# Taxon	                        TaxId	Accession	    Fasta
# Homo_sapiens----NC_012920.1	134313	NC_012920.1	    /maps/projects/lundbeck/scratch/eDNA/DBs/gtdb/r202/flavours/vanilla-organelles-virus/pkg/genomes/fasta/NC_012920.1_genomic.fna.gz

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
