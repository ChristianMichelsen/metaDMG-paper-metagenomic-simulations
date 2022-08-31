#%%

from pathlib import Path

import pysam

#%%
bamfile = Path("input-data/data/Pitch-6__deamSim__1000000.dedup.filtered.bam")


#%%

s1 = "Pitch-6___Neisseria_cinerea____GCF_900475315_1__seq-0"
s2 = "GCF_900475315.1"


#%%

save = pysam.set_verbosity(0)
samfile = pysam.AlignmentFile(bamfile, "rb")
pysam.set_verbosity(save)


print_max = 100
counter = 0
reads = []
for read in samfile.fetch(until_eof=True):
    if s1 in read.query_name and read.reference_name == s2:
        reads.append(read)
        if counter < print_max:
            print(
                read.query_name, read.is_forward, read.cigarstring, read.get_tag("MD")
            )
            counter += 1


len(reads)


print(reads[0])


reads[0].get_forward_sequence()
reads[0].get_reference_sequence()


# a = read

# a.query_name
# a.query_sequence
# a.flag

# a.reference_start
# a.mapping_quality
# a.cigar
# a.tags

# %%
