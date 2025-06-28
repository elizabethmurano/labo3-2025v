# === BLOQUE 4: Entrenamiento LightGBM para predecir clase (tn_mes+2 - tn) ===

import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import gcsfs
import joblib
from sklearn.model_selection import train_test_split
from lightgbm.callback import early_stopping, log_evaluation

# === CONFIGURACIÓN ===
BUCKET = 'bukeli'
PROYECTO = 'carbide-crowbar-463114-d5'
INPUT_PATH = f'gs://{BUCKET}/panel/df_panel_features.parquet'
OUTPUT_PATH = f'gs://{BUCKET}/panel/df_pred_con_features.parquet'
FEATURES_PATH = f'gs://{BUCKET}/modelo/features_lgbm.pkl'
MODEL_PATH = f'gs://{BUCKET}/modelo/modelo_lgbm.txt'

# === INICIALIZAR SISTEMA DE ARCHIVOS ===
fs = gcsfs.GCSFileSystem(project=PROYECTO)

# === CARGAR PANEL CON FEATURES DESDE GCS ===
print("📥 Cargando df_panel_features.parquet desde bucket...")
with fs.open(INPUT_PATH, 'rb') as f:
    df_pred = pd.read_parquet(f)

# === OPTIMIZACIÓN DE TIPOS ===
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df

df_pred = optimize_dtypes(df_pred)

# === QUITAR COLUMNAS NO NUMÉRICAS O NO ÚTILES ===
df_pred = df_pred.drop(columns=['fecha'], errors='ignore')
df_pred['periodo_int'] = df_pred['periodo'].astype(int)
ultimos_periodos = sorted(df_pred['periodo_int'].unique())[-2:]

# === FILTRAR PARA TRAINING ===
df_train = df_pred[~df_pred['periodo_int'].isin(ultimos_periodos)].copy()
df_train = df_train[df_train['clase'].notnull()]

# === FEATURES ===
features = [
    col for col in df_train.columns
    if col not in ['tn_mes+2', 'tn', 'clase', 'product_id', 'customer_id', 'periodo', 'periodo_int']
    and df_train[col].dtype != 'object'
]

X = df_train[features].copy()
y = df_train['clase'].copy()

# === GUARDAR BACKUP EN BUCKET ===
print("💾 Guardando df_pred_con_features.parquet en bucket...")
with fs.open(OUTPUT_PATH, 'wb') as f:
    df_pred.to_parquet(f, index=False)

# === GUARDAR LISTA DE FEATURES ===
print("💾 Guardando lista de features...")
with fs.open(FEATURES_PATH, 'wb') as f:
    joblib.dump(features, f)

del df_pred, df_train
gc.collect()

# === SPLIT TEMPORAL ===
print("🔀 Split temporal de entrenamiento/validación")
print("Shape X:", X.shape)
print("Shape y:", y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
del X, y
gc.collect()

# === PARÁMETROS LGBM ===
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 200,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'max_depth': 6,
    'random_state': 42,
    'verbosity': -1,
    'force_col_wise': True,
    'max_bin': 512
}

# === ENTRENAMIENTO ===
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)
del X_train, X_val, y_train, y_val

print("🚀 Entrenando modelo LightGBM...")
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    num_boost_round=500,
    callbacks=[early_stopping(30), log_evaluation(50)]
)
del lgb_train, lgb_eval
gc.collect()

# === GUARDAR MODELO ENTRENADO EN GCS ===
print("💾 Guardando modelo LightGBM en bucket...")
model.save_model(MODEL_PATH)
print("✅ Entrenamiento finalizado.")
