# run_r_baseline.R — 用官方 robCompositions 跑 impKNNa / impCoda 基线
# 用法: Rscript run_r_baseline.R <impKNNa|impCoda> <tmp目录> [library路径]
args <- commandArgs(trailingOnly = TRUE)
method <- if (length(args) >= 1) args[1] else "impKNNa"
tmpdir <- if (length(args) >= 2) args[2] else "tmp"
libs <- if (length(args) >= 3) args[3] else "C:/Users/wenyu/Documents/R/win-library/4.6"
.libPaths(c(libs, .libPaths()))
suppressMessages(library(robCompositions))

cat("method =", method, "\n")
cat("tmpdir =", tmpdir, "\n")

dirs <- list.dirs(tmpdir, recursive = FALSE)
cat("found", length(dirs), "folders\n")
for (d in dirs) {
  infile <- file.path(d, "X_missing.csv")
  if (!file.exists(infile)) next
  X <- as.matrix(read.csv(infile, check.names = FALSE))
  cat("  processing", basename(d), "dim", nrow(X), "x", ncol(X), "\n")
  res <- tryCatch({
    if (method == "impKNNa") {
      impKNNa(X, k = 5, metric = "Euclidean")
    } else if (method == "missForest") {
      missForest::missForest(X, maxiter = 10, ntree = 100, verbose = FALSE)
    } else if (method == "lrEM") {
      zCompositions::lrEM(X, label = 0, dl = rep(0.05, ncol(X)), imp.missing = TRUE)
    } else {
      impCoda(X, method = "ltsReg", dl = rep(0.05, ncol(X)), init = "KNN", k = 5)
    }
  }, error = function(e) { cat("    ERROR:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) next
  Ximp <- if (method == "impKNNa") res$xImp
          else if (method == "missForest") res$ximp
          else if (method == "lrEM") res
          else res$x
  write.csv(Ximp, file.path(d, paste0(method, ".csv")), row.names = FALSE)
}
cat("done\n")
