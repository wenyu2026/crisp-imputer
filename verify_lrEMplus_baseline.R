# 核实 lrEMplus（zCompositions 官方"零 + 缺失"同时处理函数）能否作为 CRISP 的基线
# 用法：工作目录 = crisp-imputer，Rscript verify_lrEMplus_baseline.R
.libPaths("C:/Users/wenyu/Documents/R/win-library/4.6")
suppressMessages(library(zCompositions))

cat("zCompositions version:", as.character(packageVersion("zCompositions")), "\n")
f <- ls("package:zCompositions")
cat("has lrEMplus:", "lrEMplus" %in% f, "| has impRZilr:", "impRZilr" %in% f, "\n")
cat("functions:", paste(f, collapse = " | "), "\n\n")

mae <- function(A, B, mask) mean(abs(as.matrix(A)[mask] - B[mask]))
report <- function(tag, expr) {
  r <- tryCatch(expr, error = function(e) paste("ERR:", conditionMessage(e)))
  if (is.matrix(r) || is.data.frame(r)) {
    cat(sprintf("%-44s OK  NA_left=%d  MAE=%.4f\n", tag, sum(is.na(r)), mae(r, Xt, mask)))
  } else {
    cat(sprintf("%-44s %s\n", tag, r))
  }
}

for (dir in c("tmp/ge_MCAR_10", "tmp/cement_MCAR_10", "tmp/glass_MCAR_10")) {
  Xm <- as.matrix(read.csv(file.path(dir, "X_missing.csv")))
  Xt <- as.matrix(read.csv(file.path(dir, "X_true.csv")))
  mask <- is.na(Xm) & (Xt != 0)
  cat("\n=====", dir, "| dim:", dim(Xm), "| NA:", sum(is.na(Xm)),
      "| zero:", sum(Xm == 0, na.rm = TRUE), "| to impute:", sum(mask), "=====\n")
  report("lrEMplus(dl=NULL, ini.cov=complete.obs)", lrEMplus(Xm, suppress.print = TRUE))
  report("lrEMplus(ini.cov='multRepl')", lrEMplus(Xm, ini.cov = "multRepl", suppress.print = TRUE))
  report("cmultRepl(仅结构零, 不吃 NA)", cmultRepl(Xm, label = 0, z.warning = 1))
}
cat("\n[DONE]\n")
