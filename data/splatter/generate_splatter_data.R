library(splatter)
library(scater)
library(magrittr)
library(ggplot2)
set.seed(1)


## ----groups-------------------------------------------------------------------
sim.groups <- splatSimulate(batchCells = 200,
                            nGenes = 500,
                            group.prob = c(0.75, 0.25),
                            method = "groups",
                            verbose = F)

count_matrix <- sim.groups@assays@data$counts %>% t() %>% as.data.frame()

# Split into responsive and nonresponsive cells
nonresponsive_cells <- count_matrix[which(sim.groups$Group == "Group1"),]
responsive_cells <- count_matrix[which(sim.groups$Group == "Group2"),]

# Split nonresponsive cells into background and foreground
n_nonresposive <- nrow(nonresponsive_cells)
n_bg <- (n_nonresposive / 2) %>% round()
bg_idx <- sample(seq(n_nonresposive), size = n_bg, replace = F)
fg_nonresponsive_idx <- setdiff(seq(n_nonresposive), bg_idx)
stopifnot(length(intersect(bg_idx, fg_nonresponsive_idx)) == 0)

bg_data <- nonresponsive_cells[bg_idx,]
fg_nonresponsive_data <- nonresponsive_cells[fg_nonresponsive_idx,]
fg_data <- rbind(fg_nonresponsive_data, responsive_cells)
fg_labels <- c(rep(0, nrow(fg_nonresponsive_data)), rep(1, nrow(responsive_cells)))

# Save
bg_data %>% write.csv("~/Documents/beehive/cplvm/data/splatter/two_clusters/bg.csv")
fg_data %>% write.csv("~/Documents/beehive/cplvm/data/splatter/two_clusters/fg.csv")
fg_labels %>% data.frame() %>% 
  set_colnames("fg_label") %>% 
  write.csv("~/Documents/beehive/cplvm/data/splatter/two_clusters/fg_labels.csv")


