import pandas as pd
import numpy as np
import sys

# Configurar salida de consola en UTF-8 para Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. CARGA DE DATOS Y NORMALIZACIÓN INICIAL
# ==========================================
csv_path = "Evaluaciones_Agropecuarias_Municipales_EVA_20260915.csv"
df = pd.read_csv(csv_path, low_memory=False)

# Estandarización de nombres de columnas
cols_clean = []
for col in df.columns:
    c = ' '.join(col.split())
    if 'DEP' in c and 'C' in c and 'MUN' not in c: c = 'COD_DEP'
    elif 'MUN' in c and 'C' in c and 'DEPARTAMENTO' not in c and 'MUNICIPIO' not in c: c = 'COD_MUN'
    elif 'A' in c and 'O' in c and len(c) <= 4: c = 'AÑO'
    elif 'Producc' in c: c = 'Producción (t)'
    elif 'DESAGREGAC' in c: c = 'DESAGREGACION_REGIONAL'
    elif 'rea Sembrada' in c: c = 'Área Sembrada (ha)'
    elif 'rea Cosechada' in c: c = 'Área Cosechada (ha)'
    elif 'Rendimiento' in c: c = 'Rendimiento (t/ha)'
    elif 'NOMBRE' in c and 'CIENTIFICO' in c: c = 'NOMBRE CIENTIFICO'
    cols_clean.append(c)

df.columns = cols_clean

# ==========================================
# 2. CONVERSIÓN DE TIPOS DE DATOS Y FORMATOS
# ==========================================

# AÑO: De decimal escupido a entero (ej: 2.019 -> 2019)
df['AÑO'] = (df['AÑO'] * 1000).round().astype(int)

# Función para convertir cadenas con comas/puntos a numérico flotante
def limpiar_numeros(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().replace('.', '').replace(',', '.')
    try:
        return float(val)
    except:
        return np.nan

df['Producción (t)'] = df['Producción (t)'].apply(limpiar_numeros)
df['Rendimiento (t/ha)'] = df['Rendimiento (t/ha)'].apply(limpiar_numeros)
df['Área Sembrada (ha)'] = df['Área Sembrada (ha)'].apply(limpiar_numeros)
df['Área Cosechada (ha)'] = df['Área Cosechada (ha)'].apply(limpiar_numeros)

# Codigos DANE a texto con ceros a la izquierda
df['COD_DEP'] = df['COD_DEP'].apply(lambda x: f"{int(x):02d}" if pd.notnull(x) else "")
df['COD_MUN'] = df['COD_MUN'].apply(lambda x: f"{int(round(x * 1000)):05d}" if pd.notnull(x) else "")

# ==========================================
# 3. ESTANDARIZACIÓN DE TEXTO (DIGITACIÓN)
# ==========================================
text_cols = ['DEPARTAMENTO', 'MUNICIPIO', 'GRUPO DE CULTIVO', 'SUBGRUPO DE CULTIVO', 
             'CULTIVO', 'DESAGREGACION_REGIONAL', 'PERIODO', 'ESTADO FISICO PRODUCCION', 
             'NOMBRE CIENTIFICO', 'CICLO DE CULTIVO']

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace({'NAN': np.nan, 'NONE': np.nan, '': np.nan})

# ==========================================
# 4. GESTIÓN DE INFORMACIÓN FALTANTE (NULLS)
# ==========================================

# Imputar nombre científico faltante si el cultivo está disponible
cultivo_map = df.dropna(subset=['NOMBRE CIENTIFICO']).set_index('CULTIVO')['NOMBRE CIENTIFICO'].to_dict()
df['NOMBRE CIENTIFICO'] = df['NOMBRE CIENTIFICO'].fillna(df['CULTIVO'].map(cultivo_map)).fillna('NO REGISTRADO')

# Recalcular Rendimiento nulo cuando Área Cosechada > 0
mask_rend_null = df['Rendimiento (t/ha)'].isnull() & (df['Área Cosechada (ha)'] > 0)
df.loc[mask_rend_null, 'Rendimiento (t/ha)'] = (
    df.loc[mask_rend_null, 'Producción (t)'] / df.loc[mask_rend_null, 'Área Cosechada (ha)']
).round(2)
df['Rendimiento (t/ha)'] = df['Rendimiento (t/ha)'].fillna(0)

# ==========================================
# 5. FILTRADO DE INCONSISTENCIAS LÓGICAS
# ==========================================
filas_iniciales = len(df)

# Inconsistencia 1: Área Cosechada > Área Sembrada
cond_cosecha_valida = df['Área Cosechada (ha)'] <= df['Área Sembrada (ha)']

# Inconsistencia 2: Producción > 0 con Área Cosechada = 0
cond_prod_valida = ~((df['Producción (t)'] > 0) & (df['Área Cosechada (ha)'] == 0))

# Aplicar filtro para dataset limpio
df_limpio = df[cond_cosecha_valida & cond_prod_valida].copy()

# Recalcular Rendimiento (t/ha) para asegurar consistencia exacta en todo el dataset limpio
mask_cosecha_pos = df_limpio['Área Cosechada (ha)'] > 0
df_limpio.loc[mask_cosecha_pos, 'Rendimiento (t/ha)'] = (
    df_limpio.loc[mask_cosecha_pos, 'Producción (t)'] / df_limpio.loc[mask_cosecha_pos, 'Área Cosechada (ha)']
).round(2)

# Extracción y etiquetado de registros descartados para auditoría y visualización
df_descartados = df[~cond_cosecha_valida | ~cond_prod_valida].copy()

cond_inc1_sub = df_descartados['Área Cosechada (ha)'] > df_descartados['Área Sembrada (ha)']
cond_inc2_sub = (df_descartados['Producción (t)'] > 0) & (df_descartados['Área Cosechada (ha)'] == 0)

df_descartados['MOTIVO_DESCARTADO'] = np.select(
    [cond_inc1_sub & cond_inc2_sub, cond_inc1_sub, cond_inc2_sub],
    ['Cosecha > Sembrada Y Prod > 0 sin Cosecha', 'Área Cosechada > Área Sembrada', 'Producción > 0 con Área Cosechada = 0'],
    default='Otra Inconsistencia'
)

# ==========================================
# 6. GUARDAR DATASET LIMPIO Y IMPRIMIR AUDITORÍA
# ==========================================
output_path = "EVA_limpio.csv"
df_limpio.to_csv(output_path, index=False, encoding='utf-8-sig')

filas_finales = len(df_limpio)
filas_eliminadas = len(df_descartados)

# Configuración de pandas para visualización ordenada en consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 30)

print("="*80)
print("             INFORME DE LIMPIEZA Y PREPROCESAMIENTO DE DATOS")
print("="*80)
print(f"Registros iniciales:                      {filas_iniciales:,}")
print(f"Registros en dataset limpio:              {filas_finales:,}")
print(f"Registros descartados por inconsistencia: {filas_eliminadas:,} ({filas_eliminadas/filas_iniciales:.2%})")
print("-" * 80)

print("\n--- DESGLOSE DE REGISTROS DESCARTADOS POR MOTIVO ---")
motivos_summary = df_descartados['MOTIVO_DESCARTADO'].value_counts()
for motivo, cnt in motivos_summary.items():
    print(f" • {motivo}: {cnt:,} registros")

print("\n--- MUESTRA DE REGISTROS DESCARTADOS (PRIMEROS 10 REGISTROS FILTRADOS) ---")
cols_muestra = ['DEPARTAMENTO', 'MUNICIPIO', 'CULTIVO', 'AÑO', 'Área Sembrada (ha)', 'Área Cosechada (ha)', 'Producción (t)', 'MOTIVO_DESCARTADO']
cols_presentes = [c for c in cols_muestra if c in df_descartados.columns]
print(df_descartados[cols_presentes].head(10).to_string(index=False))

print("\n--- MUESTRA DEL DATASET LIMPIO (PRIMEROS 5 REGISTROS FINALIZADOS) ---")
cols_limpias_vista = ['DEPARTAMENTO', 'MUNICIPIO', 'CULTIVO', 'AÑO', 'Área Sembrada (ha)', 'Área Cosechada (ha)', 'Producción (t)', 'Rendimiento (t/ha)']
cols_limpias_presentes = [c for c in cols_limpias_vista if c in df_limpio.columns]
print(df_limpio[cols_limpias_presentes].head(5).to_string(index=False))

print("\n" + "-" * 80)
print(f"Dataset depurado guardado exitosamente en: '{output_path}'")
print("="*80)
