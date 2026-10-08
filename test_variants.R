.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(robCompositions))
X <- as.matrix(read.csv("tmp/ge_MCAR_10/X_missing.csv"))
cat("dim:", dim(X), " NA:", sum(is.na(X)), " 0:", sum(X == 0, na.rm = TRUE), "\n")
for (m in c("Aitchison", "Euclidean", "ilr", "knn")) {
  r <- tryCatch(impKNNa(X, k = 5, metric = m), error = function(e) paste("ERR:", conditionMessage(e)))
  if (is.list(r)) cat("impKNNa metric", m, "OK dim", dim(r$x), "\n") else cat("impKNNa metric", m, "->", r, "\n")
}
for (mth in c("ltsReg", "lm", "class")) {
  r <- tryCatch(impCoda(X, method = mth), error = function(e) paste("ERR:", conditionMessage(e)))
  if (is.list(r)) cat("impCoda", mth, "OK dim", dim(r$x), "\n") else cat("impCoda", mth, "->", r, "\n")
}
