=== BLOQUE 3 === FEATURE ENGINEERING + CLUSTERING DTW EN VM (R) ===
  
  # 1. Librerías
  library(tidyverse)
library(lubridate)
library(data.table)
library(bigrquery)
library(arrow)
library(tsfeatures)
library(TSclust)         # para DTW
library(zoo)             # para rolling
library(GCSConnection)   # acceso a GCS

# 2. Configuración de acceso a GCS
gcs_set_bucket("bukeli", billing_project = "carbide-crowbar-463114-d5")

# 3. Leer panel desde GCS
cat("📥 Cargando df_panel.parquet desde bucket...\n")
df_pred <- read_parquet("gs://bukeli/panel/df_panel.parquet") %>% as.data.table()

# 4. Features temporales
df_pred[, `:=`(
  mes = periodo %% 100,
  año = floor(periodo / 100),
  mes_sin = sin(2 * pi * (periodo %% 100) / 12),
  mes_cos = cos(2 * pi * (periodo %% 100) / 12),
  trimestre = ((mes - 1) %/% 3) + 1,
  trimestre_sin = sin(2 * pi * trimestre / 4),
  trimestre_cos = cos(2 * pi * trimestre / 4),
  fin_año = as.integer(mes >= 11),
  inicio_año = as.integer(mes <= 2)
)]

# 5. Features de producto
df_pred[, tn_media_prod := mean(tn), by = product_id]
df_pred[, tn_max_prod := max(tn), by = product_id]
df_pred[, tn_min_prod := min(tn), by = product_id]
df_pred[, tn_mediana_prod := median(tn), by = product_id]
df_pred[, tn_volatilidad_prod := shift(tn) |> rollapply(12, sd, fill = 0, align = "right", partial = TRUE), by = product_id]

# Tendencia con regresión lineal
tendencia <- function(x, n) {
  x <- shift(x)
  rollapply(x, width = n, FUN = function(y) {
    if (length(unique(y)) > 1 && sd(y) > 0) coef(lm(y ~ seq_along(y)))[2] else 0
  }, fill = 0, align = "right", partial = TRUE)
}
df_pred[, tn_tendencia_12m := tendencia(tn, 12), by = product_id]
df_pred[, tn_tendencia_6m := tendencia(tn, 6), by = product_id]
df_pred[, tn_tendencia_3m := tendencia(tn, 3), by = product_id]

# 6. Cliente, interacciones, rolling, diferencias, lags
df_pred[, tn_cliente_media_6m := shift(tn) |> rollmean(6, fill = NA, align = "right", na.rm = TRUE), by = customer_id]
df_pred[, tn_media_cliente := mean(tn), by = customer_id]
df_pred[, tn_volatilidad_cliente := shift(tn) |> rollapply(12, sd, fill = 0, align = "right", partial = TRUE), by = customer_id]
df_pred[, tn_crecimiento_cliente := (shift(tn, 1) - shift(tn, 13)) / (shift(tn, 13) + 0.001), by = customer_id]

df_pred[, consistencia_prod_cliente := rollmean(as.numeric(shift(tn) > 0), 6, fill = NA, align = "right"), by = .(product_id, customer_id)]
df_pred[, ratio_actual_historico := tn / (shift(tn) |> rollmean(12, fill = NA, align = "right") + 0.001), by = .(product_id, customer_id)]

for (lag in c(1, 2, 6, 12)) {
  df_pred[, paste0("tn_lag_", lag) := shift(tn, lag), by = .(product_id, customer_id)]
}

df_pred[, trend := seq_len(.N) - 1, by = .(product_id, customer_id)]
df_pred[, tn_diff_1 := diff(tn, lag = 1), by = .(product_id, customer_id)]
df_pred[, tn_diff_12 := diff(tn, lag = 12), by = .(product_id, customer_id)]

# 7. Eventos externos
df_pred[, evento_agosto2019 := as.integer(periodo == 201906)]
df_pred[, evento_crisis_post_paso := as.integer(periodo %in% c(201906, 201907))]
df_pred[, evento_control_precios_2020 := as.integer(periodo >= 201911)]
df_pred[, es_precios_cuidados := as.integer(replace_na(plan_precios_cuidados, FALSE))]

# 8. Demanda latente
df_pred[, demanda_latente := as.integer(cust_request_tn > 0 & tn == 0)]
df_pred[, demanda_latente_3m_prod := rollsum(shift(demanda_latente), 3, fill = 0, align = "right"), by = product_id]

# 9. Lags de stock y pedidos
df_pred[, stock_final_lag1 := shift(stock_final), by = .(product_id, customer_id)]
df_pred[, cust_request_tn_lag1 := shift(cust_request_tn), by = .(product_id, customer_id)]
df_pred[, cust_request_qty_lag1 := shift(cust_request_qty), by = .(product_id, customer_id)]

# 10. Variación y std
df_pred[, tn_crecimiento_3m := (shift(tn, 1) - shift(tn, 4)) / (shift(tn, 4) + 0.001), by = .(product_id, customer_id)]
df_pred[, tn_crecimiento_6m := (shift(tn, 1) - shift(tn, 7)) / (shift(tn, 7) + 0.001), by = .(product_id, customer_id)]
df_pred[, tn_std_3m := rollapply(shift(tn), 3, sd, fill = NA, align = "right"), by = .(product_id, customer_id)]
df_pred[, tn_std_6m := rollapply(shift(tn), 6, sd, fill = NA, align = "right"), by = .(product_id, customer_id)]

# 11. Antigüedad producto
df_pred[, antiguedad_producto_meses := periodo - min(periodo), by = product_id]
df_pred[, producto_preexistente := as.integer(periodo == min(periodo)), by = product_id]

# 12. CLUSTERING DTW
cat("📊 Ejecutando clustering DTW por producto...\n")
df_ts <- df_pred[, .(tn = sum(tn)), by = .(product_id, periodo)]
df_pivot <- dcast(df_ts, product_id ~ periodo, value.var = "tn", fill = 0)

dist_matrix <- diss(df_pivot[,-1], METHOD = "DTW")
clust_result <- pam(dist_matrix, k = 5)
df_clusters <- data.table(product_id = df_pivot$product_id, cluster_dtw = clust_result$clustering)
df_pred <- merge(df_pred, df_clusters, by = "product_id", all.x = TRUE)

# 13. Guardar en GCS
cat("💾 Guardando df_panel_features.parquet en bucket...\n")
write_parquet(df_pred, sink = "gs://bukeli/panel/df_panel_features.parquet")

cat("✅ Feature engineering + clustering DTW terminado. Registros:", nrow(df_pred), "\n")