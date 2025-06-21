#BLOQUE 1 - Carga,limpieza y unión
import pandas as pd
import numpy as np
import gcsfs

# === CONFIGURACIÓN ===
BUCKET = 'bukeli'
PROYECTO = 'carbide-crowbar-463114-d5'  # Cambiá si tu proyecto en GCP tiene otro nombre
BASE_PATH = f'gs://{BUCKET}/datasets/'
OUTPUT_PATH = f'gs://{BUCKET}/intermedios/'

# === INICIALIZAR ACCESO AL BUCKET ===
fs = gcsfs.GCSFileSystem(project=PROYECTO)

# === CARGA DE ARCHIVOS ===
print("🔄 Cargando archivos desde el bucket...")

productos_pred = pd.read_csv(fs.open(BASE_PATH + "productos_pred.txt"), sep="\t")
df = pd.read_csv(fs.open(BASE_PATH + "sell-in.txt"), sep="\t")
productos = pd.read_csv(fs.open(BASE_PATH + "tb_productos.txt"), sep="\t")
stock = pd.read_csv(fs.open(BASE_PATH + "tb_stocks.txt"), sep="\t")

# === TIPADO Y LIMPIEZA ===
productos_pred["product_id"] = productos_pred["product_id"].astype("category")
df["customer_id"] = df["customer_id"].astype("category")
df["product_id"] = df["product_id"].astype("category")
df["plan_precios_cuidados"] = df["plan_precios_cuidados"].fillna(False).astype("bool")
df["periodo"] = df["periodo"].astype("int32")
df["cust_request_qty"] = df["cust_request_qty"].astype("float32")
df["cust_request_tn"] = df["cust_request_tn"].astype("float32")
df["tn"] = df["tn"].astype("float32")
df["fecha"] = pd.to_datetime(df["periodo"].astype(str) + "01", format="%Y%m%d")

productos_clean = productos.drop_duplicates().groupby('product_id').agg({
    'cat1': 'first',
    'cat2': 'first',
    'cat3': lambda x: ','.join(sorted(set(x))),
    'brand': 'first',
    'sku_size': 'first'
}).reset_index()

assert productos_clean['product_id'].is_unique, "❌ Duplicados en productos_clean"
assert stock[['periodo', 'product_id']].drop_duplicates().shape[0] == stock.shape[0], "❌ Duplicados en stock"

# === MERGE DE TABLAS ===
df = df[df["product_id"].isin(productos_pred["product_id"])]
df = df.merge(productos_clean, on='product_id', how='left')
df = df.merge(stock, on=['periodo', 'product_id'], how='left')

for col in ['cat1', 'cat2', 'cat3', 'brand']:
    df[col] = df[col].astype('category')

# === LOG FINAL ===
print("✅ Carga y procesamiento completado.")
print("Registros en df:", df.shape[0])
print("Columnas:", df.columns.tolist())

# === GUARDAR RESULTADO EN BUCKET ===
df.to_csv(fs.open(OUTPUT_PATH + "df_limpio.csv.gz", 'wb'), index=False, compression='gzip')
df.to_pickle(fs.open(OUTPUT_PATH + "df_limpio.pkl", 'wb'))

print("📦 Archivos guardados en:", OUTPUT_PATH)
