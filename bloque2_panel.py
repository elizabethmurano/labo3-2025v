import pandas as pd
import numpy as np
import gcsfs
import gc

# === CONFIGURACIÓN ===
BUCKET = 'bukeli'
PROYECTO = 'carbide-crowbar-463114-d5'
INPUT_PATH = f'gs://{BUCKET}/intermedios/df_limpio.pkl'
OUTPUT_PATH = f'gs://{BUCKET}/panel/df_panel.parquet'

# === INICIALIZAR SISTEMA DE ARCHIVOS ===
fs = gcsfs.GCSFileSystem(project=PROYECTO)

# === CARGAR DF LIMPIO DESDE GCS ===
print("📥 Cargando df_limpio.pkl desde bucket...")
with fs.open(INPUT_PATH, 'rb') as f:
    df = pd.read_pickle(f)

# === CONVERTIR FECHA A Period[M] ===
df['fecha'] = df['fecha'].astype('period[M]')

# === COMPLETAR SERIES TEMPORALES ===
def completar_series_temporales(df, umbral_meses=8):
    columnas_a_rellenar = ['cust_request_qty', 'cust_request_tn', 'tn']
    fecha_max_dataset = df['fecha'].max()
    umbral_fecha = fecha_max_dataset - umbral_meses

    fechas_clientes = df.groupby('customer_id')['fecha'].agg(fecha_ini_c='min', fecha_fin_c='max').reset_index()
    fechas_productos = df.groupby('product_id')['fecha'].agg(fecha_ini_p='min', fecha_fin_p='max').reset_index()

    fechas_clientes['inactivo'] = fechas_clientes['fecha_fin_c'] <= umbral_fecha
    fechas_productos['inactivo'] = fechas_productos['fecha_fin_p'] <= umbral_fecha

    fechas_clientes['fecha_fin_c'] = fechas_clientes.apply(
        lambda x: x['fecha_fin_c'] if x['inactivo'] else fecha_max_dataset, axis=1
    )
    fechas_productos['fecha_fin_p'] = fechas_productos.apply(
        lambda x: x['fecha_fin_p'] if x['inactivo'] else fecha_max_dataset, axis=1
    )

    fechas = pd.period_range(df['fecha'].min(), fecha_max_dataset, freq='M')
    fechas_df = pd.DataFrame({'fecha': fechas})

    clientes_fechas = fechas_clientes.merge(fechas_df, how='cross')
    clientes_fechas = clientes_fechas[
        (clientes_fechas['fecha'] >= clientes_fechas['fecha_ini_c']) &
        (clientes_fechas['fecha'] <= clientes_fechas['fecha_fin_c'])
    ][['customer_id', 'fecha']]

    productos_fechas = fechas_productos.merge(fechas_df, how='cross')
    productos_fechas = productos_fechas[
        (productos_fechas['fecha'] >= productos_fechas['fecha_ini_p']) &
        (productos_fechas['fecha'] <= productos_fechas['fecha_fin_p'])
    ][['product_id', 'fecha']]

    combinaciones_validas = productos_fechas.merge(clientes_fechas, on='fecha', how='inner')
    df_completo = combinaciones_validas.merge(df, on=['product_id', 'customer_id', 'fecha'], how='left')

    df_completo[columnas_a_rellenar] = df_completo[columnas_a_rellenar].fillna(0)
    df_completo = df_completo.sort_values(['product_id', 'customer_id', 'fecha'])
    return df_completo.reset_index(drop=True)

df_panel = completar_series_temporales(df)

# === COMPLETAR INFO PRODUCTO Y TARGET ===
columnas_info_producto = ['cat1', 'cat2', 'cat3', 'brand', 'sku_size']
productos_info = df.drop_duplicates('product_id')[['product_id'] + columnas_info_producto]
df_panel = df_panel.drop(columns=columnas_info_producto, errors='ignore')
df_panel = df_panel.merge(productos_info, on='product_id', how='left')

df_panel['periodo'] = df_panel['fecha'].dt.strftime('%Y%m').astype('int32')
df_panel['fecha'] = df_panel['fecha'].dt.to_timestamp()

df_panel = df_panel.sort_values(['product_id', 'customer_id', 'periodo'])
df_panel['tn_mes+2'] = df_panel.groupby(['product_id', 'customer_id'])['tn'].shift(-2)
df_panel['clase'] = np.log1p(df_panel['tn_mes+2'] / (df_panel['tn'] + 1e-5))

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

df_panel = optimize_dtypes(df_panel)

# === GUARDAR PANEL EN PARQUET EN EL BUCKET ===
print("💾 Guardando df_panel en bucket...")
with fs.open(OUTPUT_PATH, 'wb') as f:
    df_panel.to_parquet(f, index=False)

print("✅ Panel guardado. Registros finales:", df_panel.shape[0])
