# run_lremplus_baseline.R — 把 lrEMplus 补为正式基线（论文缺失的真正 SOTA）
#
# 背景：paper_draft.md 声称 "zCompositions cannot run on mixed structural-zero/NA data at all"。
# 实测证伪：zCompositions 1.6.2 的 lrEMplus 正是为此设计（零=0、缺失=NA）。
# 本脚本在全部数据集上跑 lrEMplus，并同时给出公平对照：
#   raw            = lrEMplus 原样输出
#   +backfill      = 单缺失行用 x = 100 - Σ已知 精确回填，其余行归一化到 100
# 输出：lremplus_baseline.csv（每个文件夹一行）+ 各文件夹内 lrEMplus.csv（成功时）
#
# 用法：cd crisp-imputer && Rscript run_lremplus_baseline.R

.libPaths(c("C:/Users/wenyu/Documents/R/win-library/4.6", .libPaths()))
suppressMessages(library(zCompositions))

rows <- list()

closure_project <- function(X) {
  Y <- pmax(as.matrix(X), 0)
  rs <- rowSums(Y); rs[rs <= 0] <- 1
  Y / rs * 100
}

residual_backfill <- function(Xp, Xm) {
  Y <- pmax(as.matrix(Xp), 0)
  nm <- rowSums(is.na(Xm))
  known <- rowSums(Xm, na.rm = TRUE)
  for (i in which(nm == 1)) {
    j <- which(is.na(Xm[i, ]))[1]
    Y[i, j] <- max(100 - known[i], 0)
  }
  closure_project(Y)
}

for (tmpdir in c("tmp", "tmp_nd", "tmp_realnd")) {
  if (!dir.exists(tmpdir)) next
  for (d in list.dirs(tmpdir, recursive = FALSE)) {
    fin <- file.path(d, "X_missing.csv")
    ftru <- file.path(d, "X_true.csv")
    if (!file.exists(fin) || !file.exists(ftru)) next
    Xm <- as.matrix(read.csv(fin, check.names = FALSE))
    Xt <- as.matrix(read.csv(ftru, check.names = FALSE))
    miss <- is.na(Xm)
    if (sum(miss) == 0) {
      rows[[length(rows) + 1]] <- data.frame(
        folder = d, n_miss = 0, status = "no_missing", mae_raw = NA,
        mae_backfill = NA, sumdev_raw = NA, sumdev_backfill = NA)
      next
    }
    dl <- rep(0.05, ncol(Xm))
    r <- tryCatch(
      lrEMplus(Xm, dl = dl, ini.cov = "multRepl", z.warning = 1, suppress.print = TRUE),
      error = function(e) paste("ERR:", conditionMessage(e)))
    if (!is.data.frame(r) && !is.matrix(r)) {
      rows[[length(rows) + 1]] <- data.frame(
        folder = d, n_miss = sum(miss), status = as.character(r),
        mae_raw = NA, mae_backfill = NA, sumdev_raw = NA, sumdev_backfill = NA)
      next
    }
    Xp <- as.matrix(r)
    if (!identical(dim(Xp), dim(Xt))) {
      rows[[length(rows) + 1]] <- data.frame(
        folder = d, n_miss = sum(miss), status = "dim_mismatch",
        mae_raw = NA, mae_backfill = NA, sumdev_raw = NA, sumdev_backfill = NA)
      next
    }
    Xb <- residual_backfill(Xp, Xm)
    rows[[length(rows) + 1]] <- data.frame(
      folder = d, n_miss = sum(miss), status = "ok",
      mae_raw = round(mean(abs(Xt[miss] - Xp[miss])), 4),
      mae_backfill = round(mean(abs(Xt[miss] - Xb[miss])), 4),
      sumdev_raw = round(mean(abs(rowSums(Xp) - 100)), 4),
      sumdev_backfill = round(mean(abs(rowSums(Xb) - 100)), 6))
    write.csv(Xp, file.path(d, "lrEMplus.csv"), row.names = FALSE)
    cat(sprintf("%-38s n_miss=%-4d MAE_raw=%-8.4f MAE_backfill=%-8.4f\n",
                d, sum(miss),
                mean(abs(Xt[miss] - Xp[miss])), mean(abs(Xt[miss] - Xb[miss]))))
  }
}

out <- do.call(rbind, rows)
write.csv(out, "lremplus_baseline.csv", row.names = FALSE)
cat("\n=== 汇总 ===\n")
cat("总数:", nrow(out), " 成功:", sum(out$status == "ok"),
    " 失败:", sum(out$status != "ok"), "\n")
cat("\n失败原因分布:\n")
print(table(out$status[out$status != "ok"]))
