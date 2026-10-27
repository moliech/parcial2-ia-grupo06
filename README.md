# Segundo Parcial — Inteligencia Artificial
**Docente:** Jhon James Cano Sánchez  
**Programa:** Ingeniería de Sistemas · Período 2026-02  
**Grupo Asignado:** Grupo 06  
**Integrantes:**  
- Jhon Esteban Molina Echavarría  
- Heiber Lozano Mercado  
**Tema:** Consumo de Agua en Hogares de Cartago (`grupo_06.csv`)  
**Repositorio GitHub:** [https://github.com/moliech/parcial2-ia-grupo06](https://github.com/moliech/parcial2-ia-grupo06)

---

## 1. Reproducción del Entorno (Docker) — Tarea 1 (0.5 pts)

El entorno de desarrollo está dockerizado bajo **Python 3.12** para facilitar su reproducción y mantener aisladas las dependencias del proyecto (`numpy`, `pandas`, `matplotlib` y `seaborn`).

### Comandos para reproducir el entorno:
```bash
# 1. Clonar el repositorio
git clone https://github.com/moliech/parcial2-ia-grupo06.git
cd parcial2-ia-grupo06

# 2. Construir y levantar el contenedor en segundo plano (nombre: parcial2_ia)
docker compose up -d --build

# 3. Ejecutar el pipeline completo del parcial desde el script principal
docker exec -it parcial2_ia python src/parcial2.py
```

---

## 2. Carga e Inspección de Datos (Tarea 2 - 0.4 pts)

Al ejecutar la carga del archivo `data/grupo_06.csv` mediante Pandas en `src/parcial2.py`, se registraron en consola todas las salidas exigidas por la rúbrica:

### Primeras 5 filas del archivo original (`head()`):
```text
  hogar_id     barrio  consumo_m3_mes  personas_hogar  consumo_excesivo
0     A001  La Isleta            59.0               8               1.0
1     A002   El Prado             7.2               5               0.0
2     A003  La Isleta            11.8               5               0.0
3     A004   El Prado            30.5               8               0.0
4     A005   El Prado            58.0               5               1.0
```

### Forma del DataFrame (`shape`):
- `(81, 5)` — 81 filas y 5 columnas.

### Tipos de datos por columna (`dtypes`):
```text
hogar_id             object
barrio               object
consumo_m3_mes      float64
personas_hogar        int64
consumo_excesivo    float64
dtype: object
```

### Cantidad de valores nulos por columna (`isnull().sum()`):
```text
hogar_id            0
barrio              1
consumo_m3_mes      3
personas_hogar      0
consumo_excesivo    1
dtype: int64
```

### Cantidad de filas duplicadas (`duplicated().sum()`):
- `0` filas duplicadas.

### Respuesta a la Pregunta Clave (Tarea 2)
> **¿Alguna columna numérica aparece con un tipo de dato que no debería tener? ¿Por qué ocurre?**

Sí. La columna objetivo `consumo_excesivo` aparece tipada como `float64` en lugar de `int64`. Esto ocurre porque en la última fila del archivo original venía un registro de resumen con etiqueta `hogar_id = "TOTAL"`, el cual contenía un valor nulo (`NaN`) en la columna objetivo. En Pandas, la presencia de valores `np.nan` en una columna numérica obliga a promocionar su tipo a punto flotante (`float64`) para poder representar la ausencia de datos.

---

## 3. Calidad y Limpieza de Datos con Pandas (Tarea 3 - 1.2 pts)

Se identificaron y corrigieron cuatro problemas de calidad en los datos, aplicando un orden metódico para evitar que las distorsiones afectaran los cálculos de imputación:

| # | Problema Detectado | ¿Cómo se detectó? | Decisión Tomada | Justificación Técnica |
|---|---|---|---|---|
| 1 | **Fila de resumen al final** | Fila 81 con etiqueta `hogar_id = "TOTAL"` y sumatorias acumuladas (`10274.7` m³ y `373` personas). | **Eliminar** | Representa un consolidado administrativo del archivo exportado y no un predio residencial. Al eliminarla, se remueve también el `NaN` de la variable objetivo. |
| 2 | **Valor atípico en consumo** | Registro `A016` con valor de `7800.0` m³ detectado mediante inspección y filtro `df['consumo_m3_mes'] > 100`. | **Corregir a 7.8 m³** | El rango general de consumo oscila entre 5 y 60 m³. Se interpretó como un error de unidades en el sistema de origen (7800 litros reportados en lugar de 7.8 m³). |
| 3 | **Valores faltantes (NaN)** | 3 registros (`A028`, `A050`, `A052`) con campos ausentes en `consumo_m3_mes`. | **Imputar con la mediana (30.50 m³)** | La mediana se calculó tras excluir la fila `TOTAL` y corregir el valor atípico de 7800 m³, evitando el sesgo de la media y conservando los 80 registros completos. |
| 4 | **Inconsistencias en barrios** | Presencia de 11 variantes ortográficas en la columna categórica (`ortez`, `ORTEZ`, `Santa Ana `, ` San Jerónimo`). | **Estandarizar (`.str.strip().str.title()`)** | Se eliminaron espacios en blanco y se aplicó formato de título, consolidando 5 barrios canónicos: El Prado, La Isleta, Ortez, San Jerónimo y Santa Ana. |

### Respuesta a la Pregunta Clave (Tarea 3)
> **¿Qué valor le pareció más sospechoso? Calcular la media de esa columna antes y después de corregirlo y explicar cómo habría afectado el escalado del perceptrón si no se hubiera corregido.**

El valor más sospechoso fue el consumo de **7800.0 m³** registrado para el hogar `A016`. Se consideró un posible error de unidades y se corrigió a **7.8 m³**, equivalente a 7800 litros.

- **Media antes de la corrección:** **$133.44\text{ m}^3$** (con una desviación estándar de aproximadamente **$868.5$**).
- **Media tras corregir el dato:** **$32.24\text{ m}^3$** (con una desviación estándar de aproximadamente **$18.1$**).

Si el valor atípico se hubiera conservado, la desviación estándar habría sido mucho mayor y las diferencias relativas entre los consumos habituales se habrían reducido considerablemente al estandarizar los datos mediante $z = \frac{x - \mu}{\sigma}$, concentrando casi todas las observaciones legítimas en un rango estrecho cercano a $-0.15$. Esto podría dificultar el ajuste de los pesos del perceptrón durante el descenso del gradiente y afectar su aprendizaje. Por esta razón, se revisó el registro antes de entrenar el modelo.

### Resumen Numérico de la Limpieza
- **Filas antes de la limpieza:** 81 filas.
- **Filas después de la limpieza:** 80 filas.
- **Distribución de clases finales:**
  - Clase 1 (Consumo excesivo): **45 registros** (56.25%).
  - Clase 0 (Consumo normal): **35 registros** (43.75%).
- **Archivo depurado exportado:** `data/grupo_06_limpio.csv`.

---

## 4. Visualización con Seaborn (Tarea 4 - 0.6 pts)

Se generaron dos gráficos estadísticos guardados mediante `plt.savefig()` con títulos, ejes rotulados y unidades claramente especificadas:

1. **`data/dispersion.png` (Gráfico 1: Dispersión $x_1$ vs $x_2$ coloreada por $y$):**
   - **Configuración:** Generado con `sns.scatterplot()`. Eje X: Consumo de agua (m³/mes); Eje Y: Personas en el hogar (número de habitantes); Matiz (`hue`): Clasificación de consumo excesivo (0: Normal en verde, 1: Excesivo en rojo).
   - **Qué muestra y conclusión:** Se observa una tendencia donde los hogares con menor número de habitantes (1 a 3 personas) y consumos superiores a los 25 m³ son clasificados en su mayoría como consumo excesivo, mientras que en hogares más numerosos (5 a 8 personas) se registran consumos moderados sin alcanzar dicha clasificación. Esto sugiere la existencia de una frontera de separación aproximada entre ambas variables.

2. **`data/comparacion.png` (Gráfico 2: Diagrama de Cajas de Consumo según Clasificación):**
   - **Configuración:** Generado con `sns.boxplot()`. Eje X: Clasificación de exceso (0: Normal, 1: Excesivo); Eje Y: Consumo mensual (m³/mes).
   - **Qué muestra y conclusión:** La mediana del consumo mensual para la clase normal se ubica en **18.8 m³** (con rango intercuartílico entre 10.7 y 25.6 m³), mientras que para la clase de consumo excesivo la mediana asciende a **47.9 m³** (con rango intercuartílico entre 37.5 y 56.2 m³). La diferencia entre las medianas muestra que los hogares clasificados con consumo excesivo presentan, en general, consumos mensuales más elevados; sin embargo, existe un área de solapamiento en consumos intermedios (25–35 m³) donde el número de habitantes resulta determinante para definir la clasificación.

---

## 5. Perceptrón desde Cero con NumPy (Tarea 5 - 1.3 pts)

### Arquitectura de Código y Modularidad
El proyecto sigue una arquitectura desacoplada para el trabajo colaborativo:
- **`src/parcial2.py` (Script principal orquestador):** Archivo exigido por la rúbrica que ejecuta de forma secuencial la carga de datos, la limpieza con Pandas, la generación de figuras con Seaborn y la llamada al modelo.
- **`src/perceptron.py` (Módulo del modelo):** Desarrollado desde cero en la rama `heiber` exclusivamente con NumPy, conteniendo las clases `Estandarizador` y `PerceptronSigmoide`.

### Componentes Matemáticos Implementados:
- **Estandarización de características:** Ajusta cada columna para tener **media 0 y desviación estándar 1**, almacenando la media ($\mu$) y la desviación ($\sigma$) para transformar nuevos datos:
  $$z = \frac{x - \mu}{\sigma}$$
- **Función Sigmoide:** Proyecta la combinación lineal a una probabilidad en el rango $(0, 1)$, con protección numérica `np.clip(z, -500, 500)` para evitar desbordamientos:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Pérdida Logarítmica (Binary Cross-Entropy):** Evalúa el error entre las probabilidades predichas y las etiquetas reales, empleando un factor de seguridad $\epsilon = 10^{-15}$:
  $$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(p_i + \epsilon) + (1 - y_i) \ln(1 - p_i + \epsilon) \right]$$
- **Descenso del Gradiente:** Entrenado durante 500 épocas (superando las 300 exigidas) con tasa de aprendizaje $\alpha = 0.1$, registrando la pérdida y exactitud en cada época:
  $$\nabla_w L = \frac{1}{N} X^T (p - y), \quad \nabla_b L = \frac{1}{N} \sum (p - y)$$
  $$w := w - \alpha \nabla_w L, \quad b := b - \alpha \nabla_b L$$

### Métricas Obtenidas en Consola:
- **Pesos iniciales:** $w_1 = +0.0050$, $w_2 = -0.0014$, $b = 0.0000$
- **Pesos finales:** $w_1 = +2.6583$, $w_2 = -1.9871$, $b = +0.5466$
- **Pérdida inicial (BCE):** $0.6913$
- **Pérdida final (BCE):** $0.2372$
- **Exactitud final (Accuracy):** **$90.00\%$**

### Gráficos Generados:
- **`data/curva_aprendizaje.png`:** Muestra la evolución descendente y convergente de la pérdida logarítmica a través de las 500 épocas de entrenamiento.
- **`data/frontera_decision.png`:** Muestra los puntos de las dos clases y la recta de decisión ($P=0.5$) proyectada en las **unidades originales** de metros cúbicos y habitantes por hogar, mediante la des-estandarización analítica:
  $$x_{2\text{ (personas)}} = \sigma_2 \cdot \left[ -\frac{w_1}{w_2} \left(\frac{x_{1\text{ (m3)}} - \mu_1}{\sigma_1}\right) - \frac{b}{w_2} \right] + \mu_2$$

---

## 6. Interpretación de Resultados (Tarea 6 - 0.7 pts)

### 1. ¿Qué significa el signo de w1 y de w2 en el contexto del problema? ¿Tiene sentido para Cartago?
- **Signo de $w_1$ (+2.6583):** Es positivo. Indica que a mayor consumo mensual de agua, aumenta la probabilidad estimada de que el hogar sea clasificado en la categoría de consumo excesivo, manteniendo constante el número de habitantes.
- **Signo de $w_2$ (-1.9871):** Es negativo. Indica que, para un mismo nivel de consumo mensual, un mayor número de habitantes reduce la probabilidad estimada de pertenecer a la categoría de exceso.
- **Sentido para Cartago:** Esta relación es coherente con la lógica de evaluar el consumo en función del tamaño del hogar, ya que no representa la misma exigencia de uso un volumen de 30 m³ en una vivienda de una sola persona que en una familia de 6 integrantes. No obstante, estas conclusiones reflejan la relación estadística aprendida a partir de las etiquetas del archivo y no constituyen por sí solas una explicación causal del consumo en el municipio.

### 2. Según la frontera de decisión, ¿los datos son aproximadamente linealmente separables? ¿Cuántos registros quedan del lado equivocado?
Los datos son **aproximadamente linealmente separables**. Con una exactitud del **90.00%**, el perceptrón clasifica correctamente 72 de los 80 registros de la muestra, quedando **8 registros ubicados del lado opuesto de la frontera de decisión**. Estos casos corresponden a observaciones en la zona intermedia (entre 22 y 37 m³) donde el tamaño del hogar por sí solo no resulta suficiente para diferenciar perfectamente ambas clases.

### 3. Simulación de dos casos nuevos realistas
Se evaluaron dos casos hipotéticos empleando la media y la desviación estándar calculadas durante el entrenamiento ($\mu_1 = 32.24$, $\sigma_1 = 18.06$; $\mu_2 = 4.66$, $\sigma_2 = 1.95$):

- **Caso 1 (2 personas, 55.0 m³/mes):**
  - Probabilidad estimada por el modelo: **99.90%**
  - Clase predicha: **1 (Consumo Excesivo)**. El resultado coincide con la clase esperada para este ejemplo, dado que presenta un consumo mensual muy superior al promedio en un hogar pequeño.
- **Caso 2 (6 personas, 18.0 m³/mes):**
  - Probabilidad estimada por el modelo: **4.52%**
  - Clase predicha: **0 (Consumo Normal)**. El resultado coincide con la clase esperada para este ejemplo, pues corresponde a un consumo moderado distribuido entre varios habitantes.

### 4. Recomendación concreta a la entidad local de Cartago
A partir de los resultados cuantitativos obtenidos en el análisis, se proponen dos recomendaciones técnicas como apoyo para la gestión operativa:

1. **Priorización de revisiones según el promedio general:** Tomando como referencia la media depurada de **$32.24\text{ m}^3$** y el peso negativo asociado a las personas ($w_2 = -1.9871$), se recomienda priorizar visitas técnicas a hogares pequeños (1 a 2 habitantes) cuyos consumos superen dicho promedio, ya que para estos perfiles el modelo estima probabilidades de exceso superiores al $85\%$.
2. **Monitoreo en la zona de solapamiento a partir de la mediana de exceso:** Considerando que la mediana del consumo excesivo identificada en el análisis exploratorio (Tarea 4) fue de **$47.9\text{ m}^3$**, se plantea utilizar este valor como umbral de referencia para auditar hogares medianos (3 a 5 integrantes) que se ubiquen en esta franja de consumo.

*Nota de alcance:* Estas propuestas constituyen un criterio inicial de tamizaje estadístico y deben ser validadas con datos de nuevos períodos antes de ser utilizadas para decisiones sancionatorias o tarifarias definitivas.

### 5. Limitaciones del modelo para tomar decisiones reales
1. **Ausencia de información temporal y climática:** El conjunto de datos no contempla variaciones estacionales ni períodos secos o lluviosos, los cuales modifican los patrones habituales de consumo hídrico en Cartago.
2. **Supuesto de linealidad:** El perceptrón asume una frontera lineal entre consumo y personas, simplificando la realidad de que en una vivienda existen consumos fijos (aseo, cocina) que no aumentan de manera proporcional por cada habitante adicional.
3. **Falta de variables prediales y socioeconómicas:** No se dispone de variables como el estrato socioeconómico, el área del inmueble ni la presencia de actividades comerciales o áreas verdes, factores indispensables para una evaluación integral.

---

## 7. Repositorio y Documentación (Tarea 7 - 0.3 pts)

El repositorio incluye la totalidad de los archivos exigidos para la evaluación:
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `src/parcial2.py` (Script principal)
- `src/perceptron.py` (Módulo del modelo)
- `data/grupo_06.csv` (CSV original asignado)
- `data/grupo_06_limpio.csv` (CSV depurado tras la limpieza)
- `data/dispersion.png` (Gráfico 1 Seaborn)
- `data/comparacion.png` (Gráfico 2 Seaborn)
- `data/curva_aprendizaje.png` (Pérdida por época)
- `data/frontera_decision.png` (Frontera en unidades originales)
- `README.md` (Documentación completa con todas las respuestas)

Se registraron al menos tres commits a lo largo del desarrollo para evidenciar el proceso evolutivo del proyecto.
