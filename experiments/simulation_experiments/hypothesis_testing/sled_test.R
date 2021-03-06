library(sLED)
library(magrittr)

data_dir = "./tmp"
X_path = file.path(data_dir, "X.csv")
Y_path = file.path(data_dir, "Y.csv")

X <- read.csv(X_path, row.names = 1) %>% as.matrix()
Y <- read.csv(Y_path, row.names = 1) %>% as.matrix()

stopifnot(ncol(X) == ncol(Y))

output1 <- sLED(X, Y, npermute = 50)

output1 %>% 
  as.data.frame() %>% 
  write.csv(file.path(data_dir, "curr_sled_output.csv"))
