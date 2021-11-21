import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

results_df = pd.read_csv("../out/data_dimension_vs_ebfs.csv", index_col=0)

plt.figure(figsize=(7, 5))
g = sns.lineplot(
    data=results_df,
    x="variable",
    y="value",
    style="context",
    err_style="bars",
    color="black",
)
g.legend_.set_title(None)
plt.legend(fontsize=20)
plt.xlabel("Data dimension")
plt.ylabel("EBF")
plt.xscale("log")
# plt.yscale("log")
plt.xticks(results_df.variable.unique().astype(int))
plt.tight_layout()
plt.savefig("../out/data_dimension_vs_ebfs.png")
plt.show()
plt.close()
import ipdb

ipdb.set_trace()
