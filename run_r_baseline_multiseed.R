# run_r_baseline_multiseed.R — R 官方基线的多种子版本
#
# 原 run_r_baseline.R 每个配置只跑一次缺失抽取（seed 42），因此 R 基线被排除在配对
# Wilcoxon 检验之外。本脚本读取 stage_r_multiseed.py 落盘的逐种子缺失矩阵，在**单个
# R 进程**内循环所有配置 × 所有种子（避免上千次进程启动），并把逐次结果写成 tidy CSV。
#
# 方法：robCompositions::impKNNa（Euclidean 模式）、missForest::missForest、
#       zCompositions::lrEMplus（官方"零+缺失"函数，多数配置会因 n>d 失败，失败原因一并记录）
#
# 用法：cd crisp-imputer && Rscript run_r_baseline_multiseed.R [ncores]

.libPaths(c("C:/Users/wenyu/Documents/R/win-library/4.6", .libPaths()))
suppressMessages(library(robCompositions))
suppressMessages(library(zCompositions))

STAGE <- "tmp_rms"
OUT <- "final_results_r_multiseed.csv"

if (!dir.exists(STAGE)) stop("先运行 python stage_r_multiseed.py")

dirs <- list.dirs(STAGE, recursive = FALSE)
cat("配置数:", length(dirs), "\n")

res <- list()
push <- function(config, seed, method, mae, sumdev, status) {
  res[[length(res) + 1]] <<- data.frame(
    config = config, seed = seed, method = method,
    mae = mae, sumdev = sumdev, status = status, stringsAsFactors = FALSE)
}

for (d in dirs) {
  cfg <- basename(d)
  Xt <- as.matrix(read.csv(file.path(d, "X_true.csv"), check.names = FALSE))
  files <- sort(list.files(d, pattern = "^X_missing_s[0-9]+\\.csv$", full.names = TRUE))
  if (length(files) == 0) next
  for (f in files) {
    seed <- as.integer(sub(".*_s([0-9]+)\\.csv$", "\\1", basename(f)))
    Xm <- as.matrix(read.csv(f, check.names = FALSE))
    miss <- is.na(Xm)
    if (sum(miss) == 0) next

    runs <- list(
      impKNNa = tryCatch(impKNNa(Xm, k = 5, metric = "Euclidean")$xImp,
                         error = function(e) paste("ERR:", conditionMessage(e))),
      missForest = tryCatch(
        missForest::missForest(Xm, maxiter = 10, ntree = 100, verbose = FALSE)$ximp,
        error = function(e) paste("ERR:", conditionMessage(e))),
      lrEMplus = tryCatch(
        as.matrix(lrEMplus(Xm, dl = rep(0.05, ncol(Xm)), ini.cov = "multRepl",
                           z.warning = 1, suppress.print = TRUE)),
        error = function(e) paste("ERR:", conditionMessage(e)))
    )

    for (nm in names(runs)) {
      r <- runs[[nm]]
      if (!is.matrix(r) || !identical(dim(r), dim(Xm))) {
        push(cfg, seed, nm, NA_real_, NA_real_,
             if (is.character(r)) r else "dim_mismatch")
      } else {
        push(cfg, seed, nm,
             mean(abs(Xt[miss] - r[miss])),
             mean(abs(rowSums(r) - 100)),
             "ok")
      }
    }
  }
  cat(sprintf("%-34s %d seeds done\n", cfg, length(files)))
}

df <- do.call(rbind, res)
write.csv(df, OUT, row.names = FALSE)

cat("\n=== 汇总 ===\n")
cat("总记录:", nrow(df), "\n")
cat("\n各方法成功/失败:\n")
print(table(df$method, df$status != "ok"))

ok <- df[df$status == "ok", ]
if (nrow(ok)) {
  cat("\n成功配置的 MAE（按方法）:\n")
  print(aggregate(mae ~ method, data = ok, FUN = function(x) round(mean(x), 4)))
}
cat("\n输出:", OUT, "\n")
