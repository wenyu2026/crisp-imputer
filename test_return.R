.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(robCompositions))
X <- as.matrix(read.csv("tmp/ge_MCAR_10/X_missing.csv"))
r <- impKNNa(X, k = 5, metric = "Euclidean")
cat("class:", class(r), "\n")
cat("names:", names(r), "\n")
cat("dim(r):", dim(r), "\n")
if (is.list(r)) {
  cat("x dim:", dim(r$x), "\n")
  cat("x head:\n")
  print(head(r$x))
} else {
  cat("矩阵, head:\n")
  print(head(r))
}
