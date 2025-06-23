# === BLOQUE 5 — PREDICCIÓN FINAL Y EXPORTACIÓN A GCS ===

import numpy as np
import pandas as pd
import lightgbm as lgb
import gcsfs
import gc

# === CONFIGURACIÓN DE BUCKET ===
BUCKET = 'bukeli'
PROYECTO = 'carbide-crowbar-463114-d5'
PREDICTORS_PATH = f'gs://{BUCKET}/panel/df_panel_features.parquet'
OUTPUT_PATH = f'gs://{BUCKET}/predicciones/prediccion_productos_mes_objetivo.csv'

# === INICIALIZAR SISTEMA DE ARCHIVOS ===
fs = gcsfs.GCSFileSystem(project=PROYECTO)

# === CARGAR PANEL CON FEATURES DESDE GCS ===
print("📥 Cargando df_panel_features.parquet desde GCS...")
with fs.open(PREDICTORS_PATH, 'rb') as f:
    df_pred = pd.read_parquet(f)

# === CREAR periodo_int POR SI FALTA ===
df_pred['periodo_int'] = df_pred['periodo'].astype(int)

# === DETERMINAR EL ÚLTIMO PERIODO DISPONIBLE ===
ultimo_periodo = sorted(df_pred['periodo_int'].unique())[-1]
fecha_base = pd.to_datetime(str(ultimo_periodo), format="%Y%m")
fecha_target = fecha_base + pd.DateOffset(months=2)
periodo_target = int(fecha_target.strftime("%Y%m"))
print(f"✅ Prediciendo tn_mes+2 para: {periodo_target} (basado en datos de {ultimo_periodo})")

# === FILTRAR SOLO FILAS DEL ÚLTIMO PERIODO ===
df_pred_final = df_pred[df_pred['periodo_int'] == ultimo_periodo].copy()

# === OPTIMIZAR TIPOS ===
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df

df_pred_final = optimize_dtypes(df_pred_final)

# === DEFINIR FEATURES ===
features = [
    col for col in df_pred_final.columns
    if col not in ['tn_mes+2', 'clase', 'product_id', 'customer_id', 'periodo', 'periodo_int']
    and np.issubdtype(df_pred_final[col].dtype, np.number)
]

# === CARGAR MODELO ENTRENADO DESDE ARCHIVO LOCAL O VARIABLE ===
# Si el modelo fue entrenado en el mismo script, ya lo tenés como `model`
# Si lo guardaste, podrías hacer: model = lgb.Booster(model_file='modelo.txt')

print("🔍 Ejecutando predicción...")
X_pred = df_pred_final[features]
df_pred_final['pred_clase'] = model.predict(X_pred)

# === RECONSTRUIR tn_mes+2 DESDE ESCALA LOGARÍTMICA ===
df_pred_final['pred_tn_mes+2'] = np.expm1(df_pred_final['pred_clase']) * (df_pred_final['tn'] + 1e-5)
df_pred_final['pred_tn_mes+2'] = df_pred_final['pred_tn_mes+2'].clip(lower=0)

# === EXPORTAR RESULTADOS AGRUPADOS POR PRODUCTO ===
df_pred_final['prediccion_toneladas'] = df_pred_final['pred_tn_mes+2']
df_entrega = df_pred_final.groupby('product_id', as_index=False)['prediccion_toneladas'].sum()
df_entrega.columns = ['product_id', 'prediccion_toneladas']

# === GUARDAR CSV EN GCS ===
print("💾 Guardando predicciones en GCS...")
with fs.open(OUTPUT_PATH, 'w') as f:
    df_entrega.to_csv(f, index=False)

print(f"✅ CSV guardado en bucket: {OUTPUT_PATH}")
print(f"📊 Forma final del archivo: {df_entrega.shape}")
gc.collect()