.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(robCompositions))
suppressMessages(library(zCompositions))
X <- as.matrix(read.csv("tmp/ge_MCAR_10/X_missing.csv"))
cat("dim:", dim(X), " NA:", sum(is.na(X)), " 0:", sum(X == 0, na.rm = TRUE), "\n")

# zCompositions lrEM（label=0 表示结构零，NA 缺失）
cat("\n--- zCompositions lrEM ---\n")
r <- tryCatch(lrEM(X, label = 0, dl = rep(0.05, ncol(X)), method = "splr"),
              error = function(e) paste("ERR:", conditionMessage(e)))
if (is.matrix(r)) cat("lrEM OK dim", dim(r), "\n") else cat("lrEM:", r, "\n")

# zCompositions lrSVD
cat("\n--- zCompositions lrSVD ---\n")
r2 <- tryCatch(lrSVD(X, label = 0, dl = rep(0.05, ncol(X))),
               error = function(e) paste("ERR:", conditionMessage(e)))
if (is.matrix(r2)) cat("lrSVD OK dim", dim(r2), "\n") else cat("lrSVD:", r2, "\n")

# impCoda workaround：closed=TRUE
cat("\n--- robCompositions impCoda (closed=TRUE) ---\n")
r3 <- tryCatch(impCoda(X, method = "ltsReg", dl = rep(0.05, ncol(X)), closed = TRUE, init = "KNN", k = 5),
               error = function(e) paste("ERR:", conditionMessage(e)))
if (is.list(r3)) cat("impCoda OK dim", dim(r3$x), "\n") else cat("impCoda:", r3, "\n")

# impKNNa 确认仍可用
cat("\n--- impKNNa (Euclidean) ---\n")
r4 <- impKNNa(X, k = 5, metric = "Euclidean")
cat("impKNNa OK dim", dim(r4$xImp), "\n")
