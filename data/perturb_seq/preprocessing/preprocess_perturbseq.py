import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import mmread
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os



### Load control data ###

data = mmread("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396857_dc_0hr.mtx.txt")

gene_names = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396857_dc_0hr_genenames.csv", index_col=0).iloc[:, 0].values
cell_names = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396857_dc_0hr_cellnames.csv", index_col=0).iloc[:, 0].values

data_dense = pd.DataFrame(data.toarray(), columns=cell_names, index=gene_names)

print("Loaded {} cells and {} genes".format(data_dense.shape[1], data_dense.shape[0]))

## Read in the metadata about which guides infected which genes

metadata = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396857_dc_0hr_cbc_gbc_dict.csv", header = None, names=['guide', 'cells'])

metadata['targeted_gene'] = [x.split("_")[1] for x in metadata.guide.values]

## Pull out the cell barcodes for each guide
# barcode_dict maps guide names (keys) to cell barcodes (values)
cells_split = [y.split(", ") for y in metadata.cells.values]
barcode_dict = {}
for ii, guide in enumerate(metadata.guide.values):
    barcode_dict[guide] = np.array(cells_split[ii])

# Get cells with only one guide
cells_unique, cells_counts = np.unique(np.concatenate([x.split(", ") for x in metadata.cells.values]), return_counts=True)
cells_with_one_guide = cells_unique[cells_counts == 1]

cells_with_one_guide = np.intersect1d(cells_with_one_guide, data_dense.columns.values)

data_dense = data_dense[cells_with_one_guide]

### Load treatment data ###

data = mmread("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396856_dc_3hr.mtx.txt")

# gene and cell names
gene_names = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396856_dc_3hr_genenames.csv", index_col=0).iloc[:, 0].values
cell_names = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396856_dc_3hr_cellnames.csv", index_col=0).iloc[:, 0].values

# format into dataframe
data_dense_3hr = pd.DataFrame(data.toarray(), columns=cell_names, index=gene_names)

print("Loaded {} cells and {} genes".format(data_dense_3hr.shape[1], data_dense_3hr.shape[0]))

## Get the guide data for the same guide as above

metadata_3hr = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/GSM2396856_dc_3hr_cbc_gbc_dict_strict.csv", header = None, names=['guide', 'cells'])

## Pull out the cell barcodes for each guide
cells_split = [y.split(", ") for y in metadata_3hr.cells.values]
barcode_dict_3hr = {}
for ii, guide in enumerate(metadata_3hr.guide.values):
    barcode_dict_3hr[guide] = np.array(cells_split[ii])


# Get cells with only one guide
cells_unique, cells_counts = np.unique(np.concatenate([x.split(", ") for x in metadata_3hr.cells.values]), return_counts=True)
cells_with_one_guide = cells_unique[cells_counts == 1]

cells_with_one_guide = np.intersect1d(cells_with_one_guide, data_dense_3hr.columns.values)

data_dense_3hr = data_dense_3hr[cells_with_one_guide]

# Only take guides that have data for both timepoints
guides_with_both_timepoints = np.array([x for x in metadata.guide.values if (x in barcode_dict.keys()) and (x in barcode_dict_3hr.keys())])
metadata_both_timepoints = metadata[metadata.guide.isin(guides_with_both_timepoints)]

# Get targeted genes with multiple guides
targeted_gene_counts = metadata_both_timepoints.targeted_gene.value_counts()
genes_with_multiple_guides = targeted_gene_counts.index.values[targeted_gene_counts > 1]


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



devs, gene_names = deviance_feature_selection(data_dense)




# Save data for all targeted genes and guides

NUM_GENES = 500

for one_gene in genes_with_multiple_guides:
    
    save_dir = pjoin("../data/targeted_genes", one_gene)
        
    print("Gene: {}".format(one_gene))
    
#     top_genes = gene_names[np.argsort(-devs)[:NUM_GENES]]
            
    # ------ Save data for this targeted gene (only genes with highest variance) -----------
    one_gene_guides = metadata_both_timepoints[metadata_both_timepoints.targeted_gene == one_gene].guide.unique()
    
    # loop over guides
    all_data_0hr = []
    all_data_3hr = []
    for ii, one_guide in enumerate(one_gene_guides):

        ## 0hr data
        corresponding_cells = barcode_dict[one_guide]
        corresponding_cells = np.intersect1d(corresponding_cells, data_dense.columns.values)
        data_one_guide = data_dense[corresponding_cells]

        ## 3hr data
        corresponding_cells = barcode_dict_3hr[one_guide]
        corresponding_cells_complete = np.intersect1d(corresponding_cells, data_dense_3hr.columns.values)
        data_one_guide_3hr = data_dense_3hr[corresponding_cells_complete]
            
        curr_shared_genes = np.intersect1d(data_one_guide.index.values, data_one_guide_3hr.index.values)
        if ii == 0:
            shared_genes = curr_shared_genes
        else:
            shared_genes = np.intersect1d(shared_genes, curr_shared_genes)
    
    

    all_data_0hr = []
    all_data_3hr = []

    # loop over guides
    for ii, one_guide in enumerate(one_gene_guides):

        ## 0hr data
        corresponding_cells = barcode_dict[one_guide]
        corresponding_cells = np.intersect1d(corresponding_cells, data_dense.columns.values)
        data_one_guide = data_dense[corresponding_cells]

        ## 3hr data
        corresponding_cells = barcode_dict_3hr[one_guide]
        corresponding_cells_complete = np.intersect1d(corresponding_cells, data_dense_3hr.columns.values)
        data_one_guide_3hr = data_dense_3hr[corresponding_cells_complete]
        
        # Only take genes that exist across guides.
        data_one_guide_0hr_aligned = data_one_guide.transpose()[shared_genes]
        data_one_guide_3hr_aligned = data_one_guide_3hr.transpose()[shared_genes]

        ## Get only the most variable genes
#         data_one_guide_0hr_aligned = data_one_guide#.transpose()#[top_genes]
#         data_one_guide_3hr_aligned = data_one_guide_3hr#.transpose()#[top_genes]

        assert data_one_guide_0hr_aligned.shape[1] == data_one_guide_3hr_aligned.shape[1]
        
        # append to this gene's list of expression
        all_data_0hr.append(data_one_guide_0hr_aligned)
        all_data_3hr.append(data_one_guide_3hr_aligned)
        
    all_data_0hr = pd.concat(all_data_0hr)
    all_data_3hr = pd.concat(all_data_3hr)
    all_data_curr_guide = pd.concat([all_data_0hr, all_data_3hr])
    
    # Save total counts for each cell
    # size_factor_dir = pjoin("./perturb_seq/data/size_factors", one_gene)
    # if not os.path.isdir(size_factor_dir):
    #     os.makedirs(size_factor_dir)
    # size_factors_0hr = all_data_0hr.sum(1).to_csv(pjoin(size_factor_dir, "size_factor_counts_0hr_{}.csv".format(one_gene)))
    # size_factors_3hr = all_data_3hr.sum(1).to_csv(pjoin(size_factor_dir, "size_factor_counts_3hr_{}.csv".format(one_gene)))
    
#     gene_vars = np.var(np.log(all_data_curr_guide.values + 1), axis=0)
    gene_vars = np.var(np.log(all_data_3hr.values + 1), axis=0)
    sorted_idx = np.argsort(-gene_vars)
#     top_genes = all_data_curr_guide.columns.values[sorted_idx[:NUM_GENES]]
#     top_genes = all_data_curr_guide.columns.values[sorted_idx[2000:2000+NUM_GENES]]
    
    
    devs, gene_names = deviance_feature_selection(all_data_curr_guide.T)
    top_genes = gene_names[np.argsort(-devs)[:NUM_GENES]]
    # print(top_genes[:10])
    
    all_data_0hr = all_data_0hr[top_genes]
    all_data_3hr = all_data_3hr[top_genes]
    
    
    
    assert all_data_0hr.shape[1] == all_data_3hr.shape[1]
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    all_data_0hr.to_csv(pjoin(save_dir, "perturb_seq_0hr_{}.csv".format(one_gene)))
    all_data_3hr.to_csv(pjoin(save_dir, "perturb_seq_3hr_{}.csv".format(one_gene)))
