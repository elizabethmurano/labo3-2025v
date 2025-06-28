# === BLOQUE 5 — PREDICCIÓN FINAL Y EXPORTACIÓN DESDE VM ===

import numpy as np
import pandas as pd
import lightgbm as lgb
import gcsfs
import pickle
import gc
import os

# === CONFIGURACIÓN DE BUCKET Y PROYECTO ===
BUCKET = 'bukeli'
PROYECTO = 'carbide-crowbar-463114-d5'
FS = gcsfs.GCSFileSystem(project=PROYECTO)

# === RUTAS ===
INPUT_FEATURES = f'gs://{BUCKET}/panel/df_panel_features.parquet'
MODEL_PATH = f'gs://{BUCKET}/modelos/modelo_lgbm.txt'
FEATURES_PATH = f'gs://{BUCKET}/modelo/features_lgbm.pkl'
OUTPUT_CSV = f'gs://{BUCKET}/exp/prediccion_productos_mes_objetivo.csv'

# === 1. CARGAR PREDICTORES CON FEATURES ===
print("📥 Cargando archivo con features desde bucket...")
with FS.open(INPUT_FEATURES, 'rb') as f:
    df_pred = pd.read_parquet(f)

# === 2. CREAR periodo_int POR SI FALTA ===
df_pred['periodo_int'] = df_pred['periodo'].astype(int)

# === 3. DETERMINAR EL ÚLTIMO PERIODO OBSERVADO ===
ultimo_periodo = sorted(df_pred['periodo_int'].unique())[-1]
fecha_base = pd.to_datetime(str(ultimo_periodo), format="%Y%m")
fecha_target = fecha_base + pd.DateOffset(months=2)
periodo_target = int(fecha_target.strftime("%Y%m"))
print(f"✅ Prediciendo tn_mes+2 para: {periodo_target} (basado en datos de {ultimo_periodo})")

# === 4. FILTRAR SOLO FILAS DEL ÚLTIMO PERIODO ===
df_pred_final = df_pred[df_pred['periodo_int'] == ultimo_periodo].copy()

# === 5. OPTIMIZAR TIPOS ===
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df_pred_final = optimize_dtypes(df_pred_final)

# === 6. CARGAR FEATURES DEL ENTRENAMIENTO ===
print("📥 Cargando lista de features...")
with FS.open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

# === 7. CARGAR MODELO LIGHTGBM DESDE GCS ===
print("📦 Cargando modelo LightGBM desde bucket...")
with FS.open(MODEL_PATH, 'rb') as f:
    with open('modelo_lgbm_tmp.txt', 'wb') as f_local:
        f_local.write(f.read())
model = lgb.Booster(model_file='modelo_lgbm_tmp.txt')

# === 8. PREDICCIÓN ===
print("🔍 Ejecutando predicción de clase...")
X_pred = df_pred_final[features]
df_pred_final['pred_clase'] = model.predict(X_pred)

# === 9. RECONSTRUCCIÓN DE tn_mes+2 DESDE ESCALA LOG ===
print("🧮 Reconstruyendo tn_mes+2 desde clase (escala log multiplicativa)...")
df_pred_final['pred_tn_mes+2'] = np.expm1(df_pred_final['pred_clase']) * (df_pred_final['tn'] + 1e-5)
df_pred_final['pred_tn_mes+2'] = df_pred_final['pred_tn_mes+2'].clip(lower=0)

# === 10. CREAR CSV DE ENTREGA POR PRODUCTO ===
df_pred_final['prediccion_toneladas'] = df_pred_final['pred_tn_mes+2']
df_entrega = df_pred_final.groupby('product_id', as_index=False)['prediccion_toneladas'].sum()
df_entrega.columns = ['product_id', 'prediccion_toneladas']

# === 11. EXPORTAR CSV AL BUCKET ===
print(f"💾 Guardando predicciones en: {OUTPUT_CSV}")
with FS.open(OUTPUT_CSV, 'w') as f:
    df_entrega.to_csv(f, index=False)

print(f"✅ CSV guardado exitosamente con shape: {df_entrega.shape}")
gc.collect()