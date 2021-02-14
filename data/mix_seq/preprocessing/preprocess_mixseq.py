import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import mmread
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os

DATA_DIR = "/Users/andrewjones/Documents/beehive/differential_covariance/clvm/mixseq_experiments/data/"
NUM_GENES = 500

# DMSO
data_dmso = mmread(pjoin(DATA_DIR, "DMSO_24hr_expt1/matrix.mtx"))
barcodes_dmso = pd.read_table(pjoin(DATA_DIR, "DMSO_24hr_expt1/barcodes.tsv"), header=None)
classifications_dmso = pd.read_csv(pjoin(DATA_DIR, "DMSO_24hr_expt1/classifications.csv"))
classifications_dmso['cell_line'] = np.array([x.split("_")[0] for x in classifications_dmso.singlet_ID.values])
gene_names_dmso = pd.read_table(pjoin(DATA_DIR, "DMSO_24hr_expt1/genes.tsv"), header=None)
data_dense_dmso = pd.DataFrame(data_dmso.toarray() , columns=barcodes_dmso.iloc[:, 0].values, index=gene_names_dmso.iloc[:, 0].values)


# Nutlin
data_nutlin = mmread(pjoin(DATA_DIR, "Idasanutlin_24hr_expt1/matrix.mtx"))
barcodes_nutlin = pd.read_table(pjoin(DATA_DIR, "Idasanutlin_24hr_expt1/barcodes.tsv"), header=None)
classifications_nutlin = pd.read_csv(pjoin(DATA_DIR, "Idasanutlin_24hr_expt1/classifications.csv"))
classifications_nutlin['cell_line'] = np.array([x.split("_")[0] for x in classifications_nutlin.singlet_ID.values])
gene_names_nutlin = pd.read_table(pjoin(DATA_DIR, "Idasanutlin_24hr_expt1/genes.tsv"), header=None)
data_dense_nutlin = pd.DataFrame(data_nutlin.toarray() , columns=barcodes_nutlin.iloc[:, 0].values, index=gene_names_nutlin.iloc[:, 0].values)

print("Loaded {} DMSO cells and {} nutlin cells".format(data_dense_dmso.shape[1], data_dense_nutlin.shape[1]))

## Use scry (poisson deviance) to compute most variable genes

# Function for computing size factors
def compute_size_factors(m):
    #given matrix m with samples in the columns
    #compute size factors
    
    sz = np.sum(m.values, axis=0) # column sums (sum of counts in each cell)
    lsz = np.log(sz)
    
    #make geometric mean of sz be 1 for poisson
    sz_poisson = np.exp(lsz - np.mean(lsz))
    return sz_poisson

def poisson_deviance(X, sz):
    
    LP = X / sz #recycling
    LP[LP > 0] = np.log(LP[LP > 0]) #log transform nonzero elements only

    # Transpose to make features in cols, observations in rows
    X = X.T
    ll_sat = np.sum(np.multiply(X, LP.T), axis=0)
    feature_sums = np.sum(X, axis=0)
    ll_null = feature_sums * np.log(feature_sums / np.sum(sz))
    return 2 * (ll_sat - ll_null)

def deviance_feature_selection(X):
    
    # Remove cells without any counts
    X = X[np.sum(X, axis=1) > 0]
    
    # Compute size factors
    sz = compute_size_factors(X)
    
    # Compute deviances
    devs = poisson_deviance(X, sz)
    
    # Get associated gene names
    gene_names = X.index.values
    
    assert gene_names.shape[0] == devs.values.shape[0]
    
    return devs.values, gene_names


assert np.array_equal(data_dense_dmso.index.values, data_dense_nutlin.index.values)
all_data_concat = pd.concat([data_dense_dmso, data_dense_nutlin], axis=1)

devs, gene_names = deviance_feature_selection(all_data_concat)

genes_sorted = gene_names[np.argsort(-devs)[:NUM_GENES]]

data_dense_dmso_variable = data_dense_dmso.transpose()[genes_sorted]
data_dense_nutlin_variable = data_dense_nutlin.transpose()[genes_sorted]

### Get TP53 mutation data ###
mutation_data = pd.read_csv("/Users/andrewjones/Downloads/TP53 Gene Effect (CERES) CRISPR (Avana) Public 20Q3.csv")


## Nutlin
p53_mutations = []
for curr_cl in classifications_nutlin.cell_line.values:
    if curr_cl not in mutation_data['Cell Line Name'].unique():
        p53_mutations.append("NotAvailable")
    else:
        p53_mutations.append(mutation_data[mutation_data['Cell Line Name'] == curr_cl]['TP53 Mutations'].values[0])
        
p53_mutations = np.array(p53_mutations)

assert p53_mutations.shape[0] == data_dense_nutlin_variable.shape[0]

p53_mutations_df = pd.DataFrame(p53_mutations, index=data_dense_nutlin_variable.index.values)
p53_mutations_df.columns = ['tp53_mutation']
p53_mutations_df.to_csv("../data/nutlin/p53_mutations_nutlin.csv")

## DMSO
p53_mutations = []
for curr_cl in classifications_dmso.cell_line.values:
    if curr_cl not in mutation_data['Cell Line Name'].unique():
        p53_mutations.append("NotAvailable")
    else:
        p53_mutations.append(mutation_data[mutation_data['Cell Line Name'] == curr_cl]['TP53 Mutations'].values[0])
        
p53_mutations = np.array(p53_mutations)

assert p53_mutations.shape[0] == data_dense_dmso_variable.shape[0]

p53_mutations_df = pd.DataFrame(p53_mutations, index=data_dense_dmso_variable.index.values)
p53_mutations_df.columns = ['tp53_mutation']
p53_mutations_df.to_csv("../data/nutlin/p53_mutations_dmso.csv")


### Save data ###

data_dense_dmso_variable.to_csv("../data/nutlin/dmso_expt1.csv")
data_dense_nutlin_variable.to_csv("../data/nutlin/nutlin_expt1.csv")
