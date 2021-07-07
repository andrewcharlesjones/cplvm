library(equalCovs)
library(magrittr)

# setwd("~/Documents/beehive/cplvm/experiments/simulation_experiments/hypothesis_testing/")

data_dir = "./tmp"
X_path = file.path(data_dir, "X.csv")
Y_path = file.path(data_dir, "Y.csv")

X <- read.csv(X_path, row.names = 1) %>% as.matrix() #%>% t()
Y <- read.csv(Y_path, row.names = 1) %>% as.matrix() #%>% t()

X <- X[1:100,]
Y <- Y[1:100,]

stopifnot(ncol(X) == ncol(Y))

output1 <- equalCovs(X, Y, size1 = nrow(X), size2 = nrow(Y))

output1 %>% 
  as.data.frame() %>% 
  t() %>% 
  set_colnames(c("test_stat", "pval")) %>% 
  write.csv(file.path(data_dir, "curr_li2012_output.csv"))


