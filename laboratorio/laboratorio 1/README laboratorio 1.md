# Laboratorio 01 (02/2026) — Clasificación de imágenes con MLP

**Estudiante:** Luis Fernando Quispe Sullca  
**Materia:** Inteligencia Artificial II (IA 2)  
**Semestre:** 02/2026  
**Universidad:** Universidad Mayor, Real y Pontificia de San Francisco Xavier de Chuquisaca (USFX)  

---

### Recursos y Enlaces
* **Video explicativo:** [Ver en YouTube](https://youtu.be/BZU83q32I2I) (`https://youtu.be/BZU83q32I2I`)
* **Dataset:** [Caltech-256 en Kaggle](https://www.kaggle.com/datasets/jessicali9530/caltech256) (`https://www.kaggle.com/datasets/jessicali9530/caltech256`)

Materia de Redes Neuronales — USFX. Cuadernillo construido a partir de `06_optimization`,
`07_receta_entrenamiento` y `08_metricas_clasificacion` (revisados el semestre anterior), aplicando
todo el flujo de trabajo a un dataset de imágenes propio con un modelo **MLP puro (sin capas
convolucionales)**.

## Nota importante sobre el dataset

Este cuadernillo se diseñó originalmente pensando en **Caltech-256** (257 categorías, 30 607
imágenes). Sin embargo, al ejecutar la exploración de datos sobre la carpeta real del Drive
(`IA2 Pacheco/256_ObjectCategories`) se comprobó que el contenido corresponde en realidad a
**Caltech-101**:

- **Número de clases:** 101
- **Número total de imágenes:** 11 036

Esto **no afecta el cumplimiento de los requisitos del laboratorio** (se explica en la siguiente
sección), pero es importante dejarlo documentado con honestidad: el nombre de la carpeta
(`256_ObjectCategories`) no coincide con el contenido real descargado.

## 1. Cumplimiento de los criterios del laboratorio

| Criterio exigido | ¿Se cumple? | Evidencia |
|---|---|---|
| Dataset ≥ 200×200 px | Sí (mayoría de las imágenes) | Sobre una muestra de 505 imágenes: ancho mínimo 102 px, promedio 383.7 px, máximo 2592 px; alto mínimo 103 px, promedio 333.5 px, máximo 1944 px. El 78.0 % de la muestra cumple ambas dimensiones ≥ 200 px de forma nativa. El 22 % restante se lleva a 200×200 mediante `Resize`, práctica estándar de preprocesamiento — el entrenamiento en sí usa 200×200 para el 100 % de las imágenes. |
| Dataset ≥ 10 000 imágenes | Sí | 11 036 imágenes totales |
| Modelo MLP (sin convolucionales) | Sí | `nn.Flatten()` + capas `nn.Linear` únicamente, sin `nn.Conv2d` en ninguna parte |
| Criterios de optimización y buenas prácticas | Sí | Ver Sección 5 |
| Despliegue de todas las métricas correspondientes | Sí | Ver Sección 7 |

## 2. Exploración de datos

- **Clases:** 101 categorías de objetos (formato `NNN.nombre-clase`, heredado de la convención
  Caltech).
- **Balance de clases:** el dataset **no está balanceado**.
  - Clase con más imágenes: `096.hammock` (285 imágenes)
  - Clase con menos imágenes: `101.head-phones` (61 imágenes)
  - Media de imágenes por clase: 109.3
  - Desviación estándar: 36.7
- **Estadísticos de normalización** (calculados sobre una muestra, imágenes redimensionadas a
  200×200):
  - Media por canal (RGB): `[0.5456, 0.5231, 0.4935]`
  - Desviación estándar por canal (RGB): `[0.3237, 0.3197, 0.3321]`
- **Partición del dataset** (estratificada 70/15/15):
  - Entrenamiento: 7 725 imágenes
  - Validación: 1 655 imágenes
  - Prueba (test): 1 656 imágenes

**Estrategia de balanceo:** dado el desbalance de clases, se aplicó un `WeightedRandomSampler` en el
`DataLoader` de entrenamiento, asignando a cada imagen un peso inversamente proporcional a la
frecuencia de su clase — así, en promedio, todas las clases se muestrean con la misma frecuencia
durante el entrenamiento, sin necesidad de duplicar archivos físicamente.

## 3. Preprocesamiento

- Redimensionamiento de todas las imágenes a **200×200 px** (`transforms.Resize`).
- Normalización con media/desviación estándar calculadas sobre el propio dataset (no con los
  valores genéricos de ImageNet).
- **Data augmentation** (solo en el conjunto de entrenamiento): `RandomHorizontalFlip`,
  `RandomRotation(10°)`, `ColorJitter` (brillo, contraste, saturación).
- Carga *lazy* de imágenes con `torchvision.datasets.ImageFolder` (no se cargan las 11 036 imágenes
  en RAM de una sola vez, a diferencia del cuadernillo 06 con CIFAR-10, por ser un dataset de mayor
  resolución y peso).
- Para acelerar la lectura en Google Colab, el dataset se copió desde Google Drive al
  almacenamiento local de la sesión (`/content/`) antes del entrenamiento — la copia tomó
  **15.9 minutos** y redujo drásticamente el tiempo por época (la lectura vía Drive montado es el
  principal cuello de botella de este tipo de proyectos en Colab, no el cómputo de la GPU).

## 4. Arquitectura del modelo (MLP)

Se definió una función `build_model` **parametrizable** en número de capas ocultas, unidades por
capa y tasa de dropout. Cada capa oculta incluye:

`Linear → BatchNorm1d → ReLU → Dropout`

**¿Por qué MLP y no una red convolucional (CNN)?** No es una decisión de diseño libre: el enunciado
del laboratorio exige explícitamente un modelo MLP, sin capas convolucionales. En un escenario sin
esa restricción, una CNN sería la opción natural para clasificación de imágenes (explota la
estructura espacial local con muchos menos parámetros). El MLP, al aplanar la imagen en un vector de
200×200×3 = **120 000 valores de entrada**, pierde toda relación espacial entre píxeles vecinos —
esto explica en gran medida el techo de desempeño obtenido (ver Sección 7).

**¿Por qué esta arquitectura específica?** No se fijó a priori: se determinó empíricamente mediante
una búsqueda de hiperparámetros (*random search*, Sección 6) sobre un subconjunto de datos, evaluando
distintas combinaciones de learning rate, batch size y número/tamaño de capas ocultas.

## 5. Validación de la arquitectura (sanity checks)

Antes de entrenar con el dataset completo, se verificó que el diseño del modelo fuera correcto:

1. **Verificación de dimensiones:** entrada `[64, 3, 200, 200]` → salida `[64, 101]` (coincide con
   `[batch, n_clases]`), confirmado.
2. **Fit de una muestra** (memorizar 1 imagen, duplicada para satisfacer el requisito de
   `BatchNorm1d` de batch_size > 1): el loss bajó de 0.0485 (época 5) a 0.00000 (época 20) —
   el modelo puede memorizar perfectamente una sola imagen, sanity check superado.
3. **Fit de un batch** (memorizar 64 imágenes): la accuracy subió de 0.6875 (época 15) a 0.8906
   (época 45), estabilizándose cerca de 0.875-0.89 — el modelo puede memorizar un batch pequeño,
   sanity check superado.

## 6. Criterios de optimización aplicados

Todos los siguientes experimentos se corrieron sobre un **subconjunto estratificado de 4 000
imágenes** (para iterar rápido), antes de decidir la configuración final:

- **Comparación de optimizadores:** SGD, SGD+Momentum, RMSProp, Adam (15 épocas cada uno).
- **Learning rate scheduling:** `StepLR` vs. `CyclicLR`.
- **Batch Normalization:** con vs. sin `BatchNorm1d` en las capas ocultas.
- **Regularización:** sin regularización vs. Dropout (0.4) + weight decay (1e-4).
- **Data augmentation:** con vs. sin transformaciones de aumento de datos.
- **Búsqueda de batch size:** 32, 64, 128.
- **Random search de hiperparámetros** (6 combinaciones aleatorias de learning rate, batch size y
  arquitectura):

| Trial | Learning rate | Batch size | Arquitectura | val_acc |
|---|---|---|---|---|
| 5 | 0.0010 | 128 | (256, 128) | **0.1506** ← mejor |
| 3 | 0.0005 | 64 | (256, 128) | 0.1499 |
| 1 | 0.0050 | 64 | (256, 128) | 0.1482 |
| 6 | 0.0010 | 32 | (512, 256, 128) | 0.1427 |
| 2 | 0.0050 | 64 | (256, 128) | 0.1422 |
| 4 | 0.0100 | 64 | (256, 128) | 0.1390 |

**Configuración final seleccionada:** learning rate = 0.001, batch size = 128, arquitectura de capas
ocultas = (256, 128).

## 7. Entrenamiento final (dataset completo)

Con la mejor configuración encontrada, se entrenó el modelo definitivo sobre las 7 725 imágenes de
entrenamiento (con `WeightedRandomSampler`, data augmentation, BatchNorm, dropout, weight decay
1e-5, y un scheduler `StepLR` que redujo el learning rate cada 10 épocas), durante **50 épocas**
(sin activarse el early stopping, configurado con paciencia de 8 épocas).

**Progreso del entrenamiento** (resumen):

| Época | train_loss | train_acc | val_loss | val_acc | learning rate |
|---|---|---|---|---|---|
| 1 | 4.5197 | 0.0299 | 4.3586 | 0.0536 | 0.001000 |
| 10 | 3.7724 | 0.1399 | 3.7502 | 0.1342 | 0.001000 |
| 20 | 3.5359 | 0.1799 | 3.6328 | 0.1729 | 0.000500 |
| 30 | 3.3862 | 0.2015 | 3.5630 | 0.1986 | 0.000250 |
| 40 | 3.2890 | 0.2244 | 3.5222 | 0.2074 | 0.000125 |
| 50 | 3.2239 | 0.2341 | 3.5057 | 0.2138 | 0.000063 |

El modelo entrenado se guardó en
`/content/drive/MyDrive/IA2 Pacheco /modelos/mlp_caltech256.pt`.

## 8. Métricas de clasificación (conjunto de test, 1 656 imágenes)

- **Accuracy (top-1):** 0.1920 (19.20 %)
- **Top-5 accuracy:** 0.4173 (41.73 %)
- **Precision / Recall / F1 — promedio macro:** 0.1601 / 0.1896 / 0.1589
- **Precision / Recall / F1 — promedio weighted:** 0.1713 / 0.1920 / 0.1662
- **AUC macro-promedio (One-vs-Rest):** 0.8319
- **AUC weighted (One-vs-Rest):** 0.8332

**Mejores clases (mayor F1-score):** `032.cartman` (0.524), `054.diamond-ring` (0.474),
`091.grand-piano-101` (0.462), `005.baseball-glove` (0.400), `044.comet` (0.400).

**Peores clases (F1-score = 0, el modelo nunca acertó):** `007.bat`, `009.bear`, `010.beer-mug`,
`027.calculator`, `061.dumb-bell`, `060.duck`, `059.drinking-straw`, `058.doorknob`, entre otras.

**Pares de clases más confundidos** (matriz de confusión):

| Real | Predicho | Casos |
|---|---|---|
| `096.hammock` | `085.goat` | 5 |
| `008.bathtub` | `034.centipede` | 5 |
| `038.chimp` | `090.gorilla` | 5 |
| `092.grapes` | `031.car-tire` | 5 |
| `012.binoculars` | `003.backpack` | 5 |
| `069.fighter-jet` | `086.golden-gate-bridge` | 5 |
| `093.grasshopper` | `068.fern` | 4 |
| `036.chandelier-101` | `066.ewer-101` | 4 |
| `062.eiffel-tower` | `073.fireworks` | 4 |
| `082.galaxy` | `044.comet` | 4 |

## 9. Análisis y discusión de resultados

- **19.2 % de accuracy top-1** sobre 101 clases (frente al ~1 % de un modelo aleatorio) muestra que
  el modelo **sí aprendió patrones reales**, muy por encima del azar, pero está lejos del desempeño
  que lograría una CNN sobre el mismo dataset.
- El **AUC macro de 0.83** (bastante más alto que la accuracy) indica que el modelo ordena
  razonablemente bien las probabilidades entre clases, aunque no siempre acierta la clase exacta en
  el top-1 — coherente con un top-5 accuracy del 41.7 %, mucho más alto que el top-1.
- Las clases con F1 = 0 tienden a ser objetos con **formas muy variables o poca textura
  distintiva** una vez aplanados en un vector de píxeles (p. ej. `duck`, `bear`, `bat`), justo el
  tipo de información que una CNN captura y un MLP no.
- Las confusiones más frecuentes ocurren entre clases con **composición visual global similar**
  (fondo, paleta de colores, disposición), lo cual es esperable en un MLP: al no tener noción de
  bordes o formas locales, tiende a confundir imágenes que "se parecen en conjunto" aunque el
  objeto central sea distinto.
- La brecha entre `train_acc` (0.234) y `val_acc` (0.214) al final del entrenamiento es moderada —
  el modelo no muestra *overfitting* severo, gracias a BatchNorm, dropout, weight decay y data
  augmentation.

**Conclusión general:** el laboratorio cumple el objetivo pedagógico de aplicar el ciclo completo de
buenas prácticas de entrenamiento y optimización (validación de arquitectura, comparación de
optimizadores, LR scheduling, regularización, balanceo de clases, búsqueda de hiperparámetros,
early stopping, y despliegue completo de métricas), documentando honestamente que el techo de
desempeño de un MLP puro en un problema de 101 clases es limitado por diseño, no por un error de
implementación.

## 10. Estructura del proyecto

```
laboratorio 1.ipynb   # Cuadernillo completo (Google Colab)
README laboratorio 1 .md                     
```

## 11. Cómo ejecutar

1. Abrir el cuadernillo en Google Colab.
2. Montar Google Drive (celda de la Sección 1).
3. Ajustar `DATA_ROOT` / `IMAGES_SUBDIR` a la ruta real del dataset en el Drive.
4. Ejecutar las celdas en orden. Se recomienda copiar el dataset a almacenamiento local de la sesión
   (`/content/`) antes de la Sección 3, para acelerar la lectura de imágenes.
5. Tiempo total de ejecución aproximado: 1.5 a 3 horas en GPU T4 de Colab con el dataset ya copiado
   localmente (ver Sección 3 del cuadernillo).

## 12. Cuadernillos base

- `06_optimization.ipynb` — optimizadores, learning rate scheduling, batch normalization,
  regularización, batch size.
- `07_receta_entrenamiento.ipynb` — receta general de entrenamiento: exploración de datos,
  validación de arquitectura (dimensiones, fit de una muestra, fit de un batch), random search de
  hiperparámetros.
- `08_metricas_clasificacion.ipynb` — métricas de clasificación (adaptadas aquí de un problema
  binario a uno multiclase de 101 categorías).
