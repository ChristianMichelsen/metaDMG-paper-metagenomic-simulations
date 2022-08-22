import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import xarray as xr
from Bio import SeqIO
from Bio.Seq import Seq
from parse import compile
from tqdm import tqdm


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


def get_alignment_files(name, N_reads):
    path = (
        Path("input-data")
        / "data-pre-mapping"
        / name
        / "single"
        / str(N_reads)
        / "reads"
    )
    path_frag = path / f"{name}_fragSim.fa.gz"
    path_deam = path / f"{name}_deamSim.fa.gz"
    path_art = path / f"{name}_art.fq.gz"
    return path_frag, path_deam, path_art


#%%


def update_counts_bang(
    seq,
    strand,
    damaged_positions_in_fragment,
    k_CT,
    N_C,
    k_GA,
    N_G,
    max_position=15,
):

    if strand == "-":
        seq = seq.reverse_complement()

    if strand == "+":
        multiplier = 1
    else:
        multiplier = -1

    for pos in damaged_positions_in_fragment:

        # reverse strand, opposite direction
        pos *= multiplier

        if abs(pos) >= max_position:
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
        if i >= max_position:
            break

        if base == "C":
            N_C[i] += 1

    for i, base in enumerate(reversed(seq)):
        if i >= max_position:
            break

        if base == "G":
            N_G[i] += 1


#%%


def load_alignment_file(path_alignment, feather_file, xarray_file, max_position=15):

    try:
        df = pd.read_feather(feather_file)
        with xr.open_dataarray(xarray_file) as ds:
            da = ds
        return df, da

    except FileNotFoundError:
        pass

    if ".fq" in path_alignment.name:
        alignment_type = "fastq"
    elif ".fa" in path_alignment.name:
        alignment_type = "fasta"
    else:
        raise AssertionError(f"Unknown alignment type: {path_alignment.suffix}")

    k_CT = defaultdict(lambda: np.zeros(max_position, dtype=int))
    N_C = defaultdict(lambda: np.zeros(max_position, dtype=int))
    k_GA = defaultdict(lambda: np.zeros(max_position, dtype=int))
    N_G = defaultdict(lambda: np.zeros(max_position, dtype=int))

    results = []
    with gzip.open(path_alignment, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, alignment_type)):
            # break
            name = record.name
            seq = record.seq
            result = parser.parse(name).named
            result["seq"] = str(seq)
            result["tax_id"] = d_acc2taxid[result["accession"]]
            result["taxon_accession"] = f"{result['taxon']}----{result['accession']}"
            results.append(result)

            update_counts_bang(
                seq=seq,
                strand=result["strand"],
                damaged_positions_in_fragment=result["damaged_positions_in_fragment"],
                k_CT=k_CT[result["tax_id"]],
                N_C=N_C[result["tax_id"]],
                k_GA=k_GA[result["tax_id"]],
                N_G=N_G[result["tax_id"]],
                max_position=max_position,
            )

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

    x = np.zeros((4, len(k_CT.keys()), 15), dtype=int)
    for i, tax_id in enumerate(k_CT.keys()):
        x[0, i, :] = k_CT[tax_id]
        x[1, i, :] = N_C[tax_id]
        x[2, i, :] = k_GA[tax_id]
        x[3, i, :] = N_G[tax_id]

    da = xr.DataArray(
        x,
        dims=["variable", "tax_id", "pos"],
        coords={
            "variable": ["k_CT", "N_C", "k_GA", "N_G"],
            "tax_id": list(k_CT.keys()),
            "pos": 1 + np.arange(max_position),
        },
    )
    da.attrs["long_name"] = path_alignment.name
    da.to_netcdf(xarray_file)

    return df, da

    # da[0, 0, :]
    # da.loc["k_CT"]
    # da.loc["k_CT"][0]
    # da.loc["k_CT"].isel(tax_id=0)
