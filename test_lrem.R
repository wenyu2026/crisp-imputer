.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(zCompositions))
X <- as.matrix(read.csv("tmp/ge_MCAR_10/X_missing.csv"))
cat("NA:", sum(is.na(X)), " 0:", sum(X == 0, na.rm = TRUE), "\n")
r <- tryCatch({
  zCompositions::lrEM(X, label = 0, dl = rep(0.05, ncol(X)), imp.missing = TRUE)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  NULL
})
if (is.matrix(r)) cat("OK dim", dim(r), "\n")
