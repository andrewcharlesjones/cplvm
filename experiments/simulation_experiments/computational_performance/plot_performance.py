import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

times_df_melted_numgenes = pd.read_csv("../out/time_performance_num_genes.csv", index_col=0)
times_df_melted_numgenes.model[times_df_melted_numgenes.model == "cplvm"] = "CPLVM"
times_df_melted_numgenes.model[times_df_melted_numgenes.model == "clvm"] = "CLVM"
times_df_melted_numgenes.model[times_df_melted_numgenes.model == "cpca"] = "CPCA"
times_df_melted_numgenes.model[times_df_melted_numgenes.model == "pcpca"] = "PCPCA"

times_df_melted_numsamples = pd.read_csv("../out/time_performance_num_samples.csv", index_col=0)
times_df_melted_numsamples.model[times_df_melted_numsamples.model == "cplvm"] = "CPLVM"
times_df_melted_numsamples.model[times_df_melted_numsamples.model == "clvm"] = "CLVM"
times_df_melted_numsamples.model[times_df_melted_numsamples.model == "cpca"] = "CPCA"
times_df_melted_numsamples.model[times_df_melted_numsamples.model == "pcpca"] = "PCPCA"
# import ipdb; ipdb.set_trace()

plt.figure(figsize=(16, 7))

plt.subplot(121)
g = sns.lineplot(data=times_df_melted_numgenes, x="variable", y="value", ci=95, err_style="bars", hue="model")
g.legend_.set_title(None)
plt.xlabel("Number of genes")
plt.ylabel("Time (s)")
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20)
plt.tight_layout()

plt.subplot(122)
g = sns.lineplot(data=times_df_melted_numsamples, x="variable", y="value", ci=95, err_style="bars", hue="model")
g.legend_.set_title(None)
plt.xlabel("Number of cells")
plt.ylabel("Time (s)")
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20)
plt.tight_layout()


plt.savefig("../out/time_performance.png")
plt.show()