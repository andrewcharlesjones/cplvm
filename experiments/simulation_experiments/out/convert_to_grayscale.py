from skimage import color
from skimage import io
import matplotlib.pyplot as plt

img_fname = "bfs_gene_sets_robustness"
img = color.rgb2gray(io.imread(img_fname + '.png'))
plt.imsave(fname=img_fname + "_grayscale.png", arr=img, cmap="gray")