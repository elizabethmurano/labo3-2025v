# === BLOQUE 3 === FEATURE ENGINEERING + CLUSTERING DTW EN VM ===

# Instalar librerías necesarias si estás en un entorno Jupyter Notebook
!pip install gcsfs tslearn --quiet

# === IMPORTACIONES ===
import pandas as pd
import numpy as np
import gc
import gcsfs
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

# === CONFIGURACIÓN DE BUCKET Y PROYECTO ===
BUCKET = 'bukeli'
PROYECTO = 'carbide-crowbar-463114-d5'
INPUT_PATH = f'gs://{BUCKET}/panel/df_panel.parquet'
OUTPUT_PATH = f'gs://{BUCKET}/panel/df_panel_features.parquet'

# === INICIALIZAR SISTEMA DE ARCHIVOS ===
fs = gcsfs.GCSFileSystem(project=PROYECTO)

# === CARGAR DATASET DESDE BUCKET ===
print("📥 Cargando df_panel.parquet desde bucket...")
with fs.open(INPUT_PATH, 'rb') as f:
    df_pred = pd.read_parquet(f)


# === FEATURES TEMPORALES ===
df_pred['mes'] = df_pred['periodo'] % 100
df_pred['año'] = df_pred['periodo'] // 100
df_pred['mes_sin'] = np.sin(2 * np.pi * df_pred['mes'] / 12)
df_pred['mes_cos'] = np.cos(2 * np.pi * df_pred['mes'] / 12)
df_pred['trimestre'] = ((df_pred['mes'] - 1) // 3) + 1
df_pred['trimestre_sin'] = np.sin(2 * np.pi * df_pred['trimestre'] / 4)
df_pred['trimestre_cos'] = np.cos(2 * np.pi * df_pred['trimestre'] / 4)
df_pred['fin_año'] = (df_pred['mes'] >= 11).astype(int)
df_pred['inicio_año'] = (df_pred['mes'] <= 2).astype(int)

# === FEATURES DE PRODUCTO ===
df_pred['tn_media_prod'] = df_pred.groupby('product_id')['tn'].transform('mean')
df_pred['tn_max_prod'] = df_pred.groupby('product_id')['tn'].transform('max')
df_pred['tn_min_prod'] = df_pred.groupby('product_id')['tn'].transform('min')
df_pred['tn_mediana_prod'] = df_pred.groupby('product_id')['tn'].transform('median')
df_pred['tn_volatilidad_prod'] = df_pred.groupby('product_id')['tn'].transform(lambda x: x.shift(1).rolling(12, min_periods=3).std().fillna(0))
df_pred['tn_tendencia_12m'] = df_pred.groupby('product_id')['tn'].transform(lambda x: x.shift(1).rolling(12, min_periods=6).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 and y.std() > 0 else 0, raw=False).fillna(0))
df_pred['tn_tendencia_6m'] = df_pred.groupby('product_id')['tn'].transform(lambda x: x.shift(1).rolling(6, min_periods=3).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 and y.std() > 0 else 0, raw=False).fillna(0))
df_pred['tn_tendencia_3m'] = df_pred.groupby('product_id')['tn'].transform(lambda x: x.shift(1).rolling(3, min_periods=2).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 and y.std() > 0 else 0, raw=False).fillna(0))

# === FEATURES DE CLIENTE ===
df_pred['tn_cliente_media_6m'] = df_pred.groupby('customer_id')['tn'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
df_pred['tn_media_cliente'] = df_pred.groupby('customer_id')['tn'].transform('mean')
df_pred['tn_volatilidad_cliente'] = df_pred.groupby('customer_id')['tn'].transform(lambda x: x.shift(1).rolling(12, min_periods=3).std().fillna(0))
df_pred['tn_crecimiento_cliente'] = df_pred.groupby('customer_id')['tn'].transform(lambda x: (x.shift(1) - x.shift(13)) / (x.shift(13) + 0.001))

# === INTERACCIÓN CLIENTE - PRODUCTO ===
df_pred['consistencia_prod_cliente'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: (x.shift(1) > 0).rolling(6, min_periods=1).mean())
df_pred['ratio_actual_historico'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x / (x.shift(1).rolling(12, min_periods=1).mean() + 0.001))

# === ROLLING Y LAGS ===
df_pred['tn_roll_mean_3'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df_pred['tn_roll_std_3'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))
df_pred['tn_roll_max_6'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).max().fillna(0))
df_pred['tn_roll_min_6'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).min().fillna(0))
for lag in [1, 2, 6, 12]:
    df_pred[f'tn_lag_{lag}'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].shift(lag)

# === DIFERENCIAS Y TENDENCIA ===
df_pred['trend'] = df_pred.groupby(['product_id', 'customer_id']).cumcount()
df_pred['tn_diff_1'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].diff(1)
df_pred['tn_diff_12'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].diff(12)

# === EVENTOS EXTERNOS Y CONTROLES ===
df_pred['evento_agosto2019'] = (df_pred['periodo'] == 201906).astype(int)
df_pred['evento_crisis_post_paso'] = df_pred['periodo'].isin([201906, 201907]).astype(int)
df_pred['evento_control_precios_2020'] = (df_pred['periodo'] >= 201911).astype(int)
df_pred['plan_precios_cuidados'] = df_pred['plan_precios_cuidados'].fillna(False)
df_pred['es_precios_cuidados'] = df_pred['plan_precios_cuidados'].astype(int)

# === DEMANDA LATENTE ===
df_pred['demanda_latente'] = ((df_pred['cust_request_tn'] > 0) & (df_pred['tn'] == 0)).astype(int)
df_pred['demanda_latente_3m_prod'] = df_pred.groupby('product_id')['demanda_latente'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())

# === LAGS DE STOCK Y PEDIDOS ===
df_pred['stock_final_lag1'] = df_pred.groupby(['product_id', 'customer_id'])['stock_final'].shift(1)
df_pred['cust_request_tn_lag1'] = df_pred.groupby(['product_id', 'customer_id'])['cust_request_tn'].shift(1)
df_pred['cust_request_qty_lag1'] = df_pred.groupby(['product_id', 'customer_id'])['cust_request_qty'].shift(1)

# === VARIACIÓN Y STD ===
df_pred['tn_crecimiento_3m'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: (x.shift(1) - x.shift(4)) / (x.shift(4) + 0.001))
df_pred['tn_crecimiento_6m'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: (x.shift(1) - x.shift(7)) / (x.shift(7) + 0.001))
df_pred['tn_std_3m'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
df_pred['tn_std_6m'] = df_pred.groupby(['product_id', 'customer_id'])['tn'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).std())

# === ANTIGÜEDAD PRODUCTO ===
primera_fecha_prod = df_pred.groupby('product_id')['periodo'].transform('min')
df_pred['antiguedad_producto_meses'] = df_pred['periodo'] - primera_fecha_prod
df_pred['producto_preexistente'] = (primera_fecha_prod == df_pred['periodo'].min()).astype(int)

# === CLUSTERING DTW ===

print("📊 Ejecutando clustering DTW por producto...")
df_ts = df_pred.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()
df_pivot = df_ts.pivot(index='product_id', columns='periodo', values='tn').fillna(0)

scaler = TimeSeriesScalerMeanVariance()
ts_data = scaler.fit_transform(to_time_series_dataset(df_pivot.values))

model = TimeSeriesKMeans(n_clusters=5, metric="dtw", random_state=42, verbose=True)
cluster_labels = model.fit_predict(ts_data)

df_clusters = pd.DataFrame({'product_id': df_pivot.index, 'cluster_dtw': cluster_labels})
df_pred = df_pred.merge(df_clusters, on='product_id', how='left')
df_pred['cluster_dtw'] = df_pred['cluster_dtw'].astype('category')

# === GUARDAR EN GCS ===
print("💾 Guardando df_panel_features.parquet en bucket...")
with fs.open(OUTPUT_PATH, 'wb') as f:
    df_pred.to_parquet(f, index=False)

print("✅ Feature engineering + clustering DTW terminado. Registros:", df_pred.shape[0])
gc.collect()

