#%%

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from parse import compile
from tqdm import tqdm

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
    "{sample}---"
    "{taxon1}-___"
    "{taxon2}____"
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
        return ""
    elif "," not in damaged_positions:
        return str([int(damaged_positions)])
    else:
        return str([int(string_pos) for string_pos in damaged_positions.split(",")])


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

import gzip

from Bio import SeqIO

# Abiotrophia_sp001815865----GCF_001815865.1


def load_alignment_file(bam_file, feather_file):

    try:
        return pd.read_feather(feather_file)
    except:
        pass

    if ".fq" in bam_file.name:
        alignment_type = "fastq"
    elif ".fa" in bam_file.name:
        alignment_type = "fasta"
    else:
        raise AssertionError(f"Unknown alignment type: {bam_file.suffix}")

    results = []
    with gzip.open(bam_file, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, alignment_type)):
            # break
            name = record.name
            seq = record.seq
            result = parser.parse(name).named
            result["seq"] = str(seq)
            result["tax_id"] = d_acc2taxid[result["accession"]]
            result["taxon_accession"] = f"{result['taxon1']}----{result['accession']}"
            results.append(result)

    df = pd.DataFrame(results)

    categories = [
        "sample",
        "taxon1",
        "taxon2",
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

# 83381

taxon_accession = "Acetobacterium_sp003260995----GCF_003260995.1"
df_83381_frag = df_pitch6_1M_fragSim.query(f"taxon_accession == '{taxon_accession}'")
df_83381_deam = df_pitch6_1M_deamSim.query(f"taxon_accession == '{taxon_accession}'")
df_83381_art = df_pitch6_1M_art.query(f"taxon_accession == '{taxon_accession}'")

df_83381_deam["ancient_modern"].value_counts()

df_83381_deam["fragment_length"].min()
df_83381_deam["fragment_length"].max()
df_83381_deam["fragment_length"].mean()
df_83381_deam["fragment_length"].median()

df_83381_frag.iloc[0]["seq"]
df_83381_deam.iloc[0]["seq"]



# comm	    taxon	                                        frag_type	seq_depth	seq_depth_rounded
# Pitch-6	Acetobacterium_sp003260995----GCF_003260995.1	ancient	    3318	    15974.0
# Pitch-6	Acetobacterium_sp003260995----GCF_003260995.1	modern	    0	        0.0

# Taxon	                                        Community	Coverage	        Read_length	Read_length_std	    Read_length_min	Read_length_max	onlyAncient
# Acetobacterium_sp003260995----GCF_003260995.1	Pitch-6	    1.89790335295038	81	        16.54811034608462	30	            81	            True


# Community	Taxon	                                        Rank	Read_length	Read_length_std	    Read_length_min	Read_length_max	Perc_rel_abund
# Pitch-6	Acetobacterium_sp003260995----GCF_003260995.1	13	    81	        16.54811034608462	30	            81	            0.5144226637497638


# Taxon	                                        TaxId	Accession	        Fasta
# Acetobacterium_sp003260995----GCF_003260995.1	83381	GCF_003260995.1	    /vol/cloud/geogenetics/DBs/gtdb/r202/data/vanilla-organelles-virus/pkg/genomes/fasta/GCF_003260995.1_genomic.fna.gz


# "comm": "Pitch-6",
# "taxon": "Acetobacterium_sp003260995----GCF_003260995.1",
# "rel_abund": 0.5144226637497638,
# "genome_size": 3988368,
# "accession": "GCF_003260995.1",
# "onlyAncient": true,
# "fragments_ancient": {
#     "fragments": {
#         "length": [
#             30,
#             81,
#         ],
#         "freq": [
#             0.004524662304682653,
#             0.34546251644015785,
#         ],
#     },
#     "dist_params": {"scale": null, "sigma": null, "rnd_seed": null},
#     "avg_len": 62.609414937175856,
#     "seq_depth": 3318,
#     "seq_depth_original": 3318,
#     "fold": 0.05157899170788653,
#     "fold_original": 0.05159002378917893,
# },
# "fragments_modern": null,
# "coverage_enforced": true,
# "seq_depth": 15974,
