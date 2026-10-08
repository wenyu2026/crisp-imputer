.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(zCompositions))
X <- as.matrix(read.csv("tmp/ge_MCAR_10/X_missing.csv"))
cat("dim:", dim(X), " NA:", sum(is.na(X)), " 0:", sum(X == 0, na.rm = TRUE), "\n")

cat("\n--- cmultRepl（结构零替换，非缺失插补） ---\n")
r <- tryCatch(cmultRepl(X, label = 0, dl = rep(0.05, ncol(X)), z.warning = 1),
              error = function(e) paste("ERR:", conditionMessage(e)))
if (is.matrix(r)) { cat("cmultRepl OK dim", dim(r), " 输出0数:", sum(r == 0), "\n") } else { cat("cmultRepl:", r, "\n") }

cat("\n--- impRZilr（zeros + 缺失多元插补） ---\n")
r2 <- tryCatch(impRZilr(X, label = 0, dl = rep(0.05, ncol(X)), method = "lm"),
               error = function(e) paste("ERR:", conditionMessage(e)))
if (is.matrix(r2)) { cat("impRZilr OK dim", dim(r2), "\n") } else { cat("impRZilr:", r2, "\n") }
