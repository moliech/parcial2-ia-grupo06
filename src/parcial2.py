import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from perceptron import (
        Estandarizador,
        PerceptronSigmoide,
        graficar_curva_aprendizaje,
        graficar_frontera_decision,
    )
except ImportError:
    from src.perceptron import (
        Estandarizador,
        PerceptronSigmoide,
        graficar_curva_aprendizaje,
        graficar_frontera_decision,
    )


def tarea2_carga_e_inspeccion(ruta_csv: str) -> pd.DataFrame:
    print("=" * 70)
    print("TAREA 2: CARGA E INSPECCION DE DATOS (grupo_06.csv)")
    print("=" * 70)

    df_crudo = pd.read_csv(ruta_csv)
    print("\n[1] Primeras 5 filas del archivo original:")
    print(df_crudo.head())
    print("\n[2] Forma del DataFrame (shape):", df_crudo.shape)
    print("\n[3] Tipos de datos por columna (dtypes):")
    print(df_crudo.dtypes)
    print("\n[4] Cantidad de valores nulos por columna:")
    print(df_crudo.isnull().sum())
    print("\n[5] Cantidad de filas duplicadas:", df_crudo.duplicated().sum())

    print("\n--- RESPUESTA A PREGUNTA CLAVE TAREA 2 ---")
    print("¿Alguna columna numerica aparece con tipo que no deberia tener?")
    print("-> La columna objetivo 'consumo_excesivo' aparece como float64 en lugar de int64.")
    print("   Causa: La presencia de la fila de resumen 'TOTAL' al final del archivo introduce")
    print("   un valor nulo (NaN) en la columna objetivo, forzando a Pandas a promover")
    print("   la columna a float64 para representar el valor ausente.")
    return df_crudo


def tarea3_calidad_y_limpieza(df_crudo: pd.DataFrame, ruta_salida_limpio: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("TAREA 3: CALIDAD Y LIMPIEZA DE DATOS CON PANDAS")
    print("=" * 70)

    filas_iniciales = len(df_crudo)
    df = df_crudo.copy()

    # 1. Fila de resumen 'TOTAL'
    df = df[df['hogar_id'] != 'TOTAL'].copy()
    print("[Problema 1] Fila TOTAL eliminada correctamente.")

    # 2. Outlier 7800.0 m3
    media_antes_outlier = df['consumo_m3_mes'].mean()
    df.loc[df['hogar_id'] == 'A016', 'consumo_m3_mes'] = 7.8
    media_despues_outlier = df['consumo_m3_mes'].mean()
    print(f"[Problema 2] Media con outlier: {media_antes_outlier:.2f} m3 | Media corregida: {media_despues_outlier:.2f} m3")

    # 3. Imputacion de nulos
    mediana_consumo = df['consumo_m3_mes'].median()
    df['consumo_m3_mes'] = df['consumo_m3_mes'].fillna(mediana_consumo)
    print(f"[Problema 3] Nulos imputados con la mediana: {mediana_consumo:.2f} m3.")

    # 4. Normalizacion de categorias de barrio
    df['barrio'] = df['barrio'].str.strip().str.title()
    df['consumo_excesivo'] = df['consumo_excesivo'].astype(int)

    print(f"[Resumen] Filas antes: {filas_iniciales} | Filas despues: {len(df)}")
    print("Distribucion de clases depuradas:\n", df['consumo_excesivo'].value_counts())

    os.makedirs(os.path.dirname(ruta_salida_limpio), exist_ok=True)
    df.to_csv(ruta_salida_limpio, index=False)
    print(f"Archivo guardado exitosamente en: {ruta_salida_limpio}")
    return df


def tarea4_visualizaciones_seaborn(df: pd.DataFrame, dir_salida: str = "data") -> None:
    print("\n" + "=" * 70)
    print("TAREA 4: VISUALIZACION DE DATOS CON SEABORN")
    print("=" * 70)

    sns.set_theme(style="whitegrid")
    palette = {0: "#10b981", 1: "#ef4444"}

    # Grafico 1: Dispersion
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="consumo_m3_mes", y="personas_hogar", hue="consumo_excesivo",
        palette=palette, s=80, alpha=0.85, edgecolor="black"
    )
    plt.title("Dispersión: Consumo Mensual vs Habitantes por Hogar (Cartago)", fontsize=13, fontweight="bold")
    plt.xlabel("Consumo de Agua (m³/mes)", fontsize=11)
    plt.ylabel("Personas en el Hogar", fontsize=11)
    plt.legend(title="Consumo Excesivo", labels=["0: Normal", "1: Excesivo"])
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, "dispersion.png"), dpi=300)
    plt.close()

    # Grafico 2: Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="consumo_excesivo", y="consumo_m3_mes", palette=palette)
    plt.title("Comparación de Consumo (m³/mes) según Clasificación de Exceso", fontsize=13, fontweight="bold")
    plt.xlabel("Clasificación (0: Normal, 1: Excesivo)", fontsize=11)
    plt.ylabel("Consumo Mensual (m³/mes)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, "comparacion.png"), dpi=300)
    plt.close()
    print("[Graficos Seaborn] Guardados 'dispersion.png' y 'comparacion.png'.")


def tarea5_entrenamiento_perceptron(df: pd.DataFrame, dir_salida: str = "data") -> tuple:
    print("\n" + "=" * 70)
    print("TAREA 5: PERCEPTRON DESDE CERO CON NUMPY")
    print("=" * 70)

    X = df[["consumo_m3_mes", "personas_hogar"]].to_numpy(dtype=float)
    y = df["consumo_excesivo"].to_numpy(dtype=float)

    escalador = Estandarizador()
    X_std = escalador.fit_transform(X)

    modelo = PerceptronSigmoide(tasa_aprendizaje=0.1, epocas=500, semilla=42)
    metricas = modelo.entrenar(X_std, y)

    w1_ini, w2_ini = metricas["pesos_iniciales"]
    w1_fin, w2_fin = metricas["pesos_finales"]

    print("\n--- RESULTADOS DEL ENTRENAMIENTO ---")
    print(f"Pesos iniciales : w1 = {w1_ini:.4f}, w2 = {w2_ini:.4f}, b = {metricas['sesgo_inicial']:.4f}")
    print(f"Pesos finales   : w1 = {w1_fin:.4f}, w2 = {w2_fin:.4f}, b = {metricas['sesgo_final']:.4f}")
    print(f"Perdida inicial : {metricas['perdida_inicial']:.4f} | Perdida final: {metricas['perdida_final']:.4f}")
    print(f"Exactitud final : {metricas['exactitud_final'] * 100:.2f}%")

    graficar_curva_aprendizaje(modelo.historial_perdida, os.path.join(dir_salida, "curva_aprendizaje.png"))
    graficar_frontera_decision(
        X, y, modelo.pesos, modelo.sesgo, escalador.media, escalador.desviacion,
        os.path.join(dir_salida, "frontera_decision.png")
    )
    return modelo, escalador


def tarea6_evaluacion_casos_cartago(modelo: PerceptronSigmoide, escalador: Estandarizador) -> None:
    print("\n" + "=" * 70)
    print("TAREA 6: SIMULACION DE CASOS REALISTAS PARA CARTAGO")
    print("=" * 70)

    # Caso 1 (esperado 1): 2 personas con 55 m3 | Caso 2 (esperado 0): 6 personas con 18 m3
    casos = np.array([[55.0, 2.0], [18.0, 6.0]])
    casos_std = escalador.transform(casos)
    probs = modelo.predecir_probabilidad(casos_std)
    preds = (probs >= 0.5).astype(int)

    print(f"Caso 1 (2 pers, 55.0 m3) -> Prob: {probs[0]:.4f} ({probs[0]*100:.2f}%) -> Prediccion: {preds[0]} (Esperado: 1)")
    print(f"Caso 2 (6 pers, 18.0 m3) -> Prob: {probs[1]:.4f} ({probs[1]*100:.2f}%) -> Prediccion: {preds[1]} (Esperado: 0)")


def main():
    ruta_csv_original = "data/grupo_06.csv" if os.path.exists("data/grupo_06.csv") else "Parcial2/grupo_06.csv"
    ruta_csv_limpio = "data/grupo_06_limpio.csv"

    df_crudo = tarea2_carga_e_inspeccion(ruta_csv_original)
    df_limpio = tarea3_calidad_y_limpieza(df_crudo, ruta_csv_limpio)
    tarea4_visualizaciones_seaborn(df_limpio, dir_salida="data")
    modelo, escalador = tarea5_entrenamiento_perceptron(df_limpio, dir_salida="data")
    tarea6_evaluacion_casos_cartago(modelo, escalador)


if __name__ == "__main__":
    main()
