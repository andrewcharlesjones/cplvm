import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import socket

if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_reads.gct"
    METADATA_PATH_SAMPLE = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    METADATA_PATH_SUBJECT = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_expected_count.gct"
    METADATA_PATH_SAMPLE = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    METADATA_PATH_SUBJECT = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"

NUM_GENES = 500


# ---------------- Load data ----------------

# Sample-level metadata
v8_metadata_sample = pd.read_table(METADATA_PATH_SAMPLE)
v8_metadata_sample["sample_id"] = [
    "-".join(x.split("-")[:3]) for x in v8_metadata_sample.SAMPID.values
]

# Subject-level metadata
v8_metadata_subject = pd.read_table(METADATA_PATH_SUBJECT)

# get lung tissue IDs
heart_sample_ids = v8_metadata_sample.sample_id.values[
    v8_metadata_sample.SMTSD == "Artery - Coronary"
]

# Get subject IDs who were or weren't on a ventilator
heartdisease_ids = v8_metadata_subject.SUBJID.values[v8_metadata_subject.MHHRTDIS == 1]
noheartdisease_ids = v8_metadata_subject.SUBJID.values[
    v8_metadata_subject.MHHRTDIS == 0
]


# Get sample names of expression data
expression_ids = pd.read_table(
    EXPRESSION_PATH, skiprows=2, index_col=0, nrows=1
).columns.values[1:]

expression_subject_ids = np.array(["-".join(x.split("-")[:2]) for x in expression_ids])
expression_sample_ids = np.array(["-".join(x.split("-")[:3]) for x in expression_ids])

heartdisease_idx = np.where(
    np.logical_and(
        np.isin(expression_sample_ids, heart_sample_ids),
        np.isin(expression_subject_ids, heartdisease_ids),
    )
    == True
)[0]
noheartdisease_idx = np.where(
    np.logical_and(
        np.isin(expression_sample_ids, heart_sample_ids),
        np.isin(expression_subject_ids, noheartdisease_ids),
    )
    == True
)[0]
assert len(np.intersect1d(heartdisease_idx, noheartdisease_idx)) == 0

# Load expression
expression_heartdisease = pd.read_table(
    EXPRESSION_PATH, skiprows=2, index_col=0, usecols=np.insert(heartdisease_idx, 0, 0)
)
expression_noheartdisease = pd.read_table(
    EXPRESSION_PATH,
    skiprows=2,
    index_col=0,
    usecols=np.insert(noheartdisease_idx, 0, 0),
)


expression_heartdisease = expression_heartdisease.transpose()
expression_noheartdisease = expression_noheartdisease.transpose()

expression_heartdisease = expression_heartdisease.fillna(0.0)
expression_noheartdisease = expression_noheartdisease.fillna(0.0)


# Get variable genes only
all_data = pd.concat([expression_heartdisease, expression_noheartdisease], axis=0)
all_data = np.log(all_data + 1)
all_data = all_data - np.mean(all_data, axis=0)
gene_vars = np.var(all_data, axis=0).values
top_gene_idx = np.argsort(-gene_vars)[:NUM_GENES]


expression_heartdisease = expression_heartdisease.iloc[:, top_gene_idx]
expression_noheartdisease = expression_noheartdisease.iloc[:, top_gene_idx]


# Save data
expression_heartdisease.to_csv("../data/gtex_expression_artery_heartdisease.csv")
expression_noheartdisease.to_csv("../data/gtex_expression_artery_noheartdisease.csv")


import ipdb

ipdb.set_trace()
