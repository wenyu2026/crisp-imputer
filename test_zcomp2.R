.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(zCompositions))
cat("cmultRepl args:\n"); print(args(cmultRepl)); cat("\n")
X <- as.matrix(read.csv("tmp/ge_MCAR_10/X_missing.csv"))

# 完整标准管线：cmultRepl 替换结构零 -> lrEM 插补缺失
cat("--- 管线: cmultRepl(替换0) -> lrEM(插补缺失) ---\n")
r1 <- tryCatch(cmultRepl(X, label = 0, z.warning = 1),
               error = function(e) paste("ERR1:", conditionMessage(e)))
if (is.matrix(r1)) {
  cat("cmultRepl OK, 输出0数:", sum(r1 == 0), "\n")
  r2 <- tryCatch(lrEM(r1, imp.missing = TRUE),
                 error = function(e) paste("ERR2:", conditionMessage(e)))
  if (is.matrix(r2)) cat("lrEM OK dim", dim(r2), " NA:", sum(is.na(r2)), "\n")
  else cat("lrEM:", r2, "\n")
} else { cat("cmultRepl:", r1, "\n") }

# multRepl（乘法替换 zeros）单独试
cat("\n--- multRepl ---\n")
r3 <- tryCatch(multRepl(X, label = 0, z.warning = 1),
               error = function(e) paste("ERR3:", conditionMessage(e)))
if (is.matrix(r3)) { cat("multRepl OK dim", dim(r3), " 输出0数:", sum(r3 == 0), "\n") } else { cat("multRepl:", r3, "\n") }
