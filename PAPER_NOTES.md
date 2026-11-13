# PAPER_NOTES — Modelos generativos de canales loopback IBFD medidos

Notas técnicas para redactar el artículo (destino: revista). Todo lo que aparece aquí está sacado del
código, de las configuraciones o de los ficheros de resultados del repositorio, salvo lo marcado como
**[NO VERIFICADO]** (dato que no consta en el repositorio o que se dedujo sin comprobarlo) o
**[AD HOC]** (calculado con un script puntual durante el desarrollo que no está en el repositorio;
las cifras son las que se obtuvieron, pero habría que regenerarlas con un script versionado antes de
publicarlas).

Rutas relativas a la raíz del repositorio. La ronda de resultados **final** es
`eval_results/sodupe_180_split_g200/` (checkpoints en `checkpoints/sodupe_180_split_g200/`). Esa ronda
se repitió con **5 semillas de entrenamiento** (42 en la carpeta raíz; 1–4 en
`eval_results/sodupe_180_split_g200/seeds/seed{1..4}/`, checkpoints en
`checkpoints/sodupe_180_split_g200/seed{1..4}/`) y se añadió un **bootstrap por bloques** (§4.2, §6.10).
**Las cifras de §6.1–6.8 son de la semilla 42; para el artículo deben usarse las de §6.10**
(media ± desviación entre semillas, IC95 y comparación emparejada con el gaussiano).

---

## 1. Objetivo y pregunta de investigación

**Objetivo.** Entrenar y comparar cinco modelos generativos profundos (GAN, DCGAN, WGAN-GP, DDPM y
DDIM) que generen respuestas al impulso de canal (CIR) sintéticas estadísticamente equivalentes a
canales *loopback* medidos en un sistema full-duplex en banda (IBFD), y compararlos con dos modelos
estadísticos clásicos ajustados a los mismos datos.

**Pregunta de investigación (tal como ha quedado tras el análisis).** ¿Reproducen los modelos
generativos la estadística de los canales loopback medidos, *más allá* de lo que reproduce un modelo
estadístico clásico de segundo orden? En concreto:
1. ¿Qué modelo generativo reproduce mejor la estadística de los canales medidos sobre un conjunto de
   test independiente?
2. ¿Aportan algo sobre un modelo gaussiano con la covarianza completa entre taps (que reproduce por
   construcción toda la estadística de segundo orden: PDP, correlación en frecuencia, ancho de banda
   de coherencia)?
3. ¿Se traslada esa diferencia a métricas de enlace (BER OFDM con ecualización ZF)?

**Mensaje que soportan los resultados** (ver §6): en las métricas de segundo orden el gaussiano es el
techo y los generativos no lo superan (como es de esperar); en las propiedades **por canal en el dominio
de la frecuencia** (nulos, rizado, fracción de banda en *fade*, eficiencia espectral, ganancia de ruido
ZF) y en la **BER**, los generativos superan claramente a los dos modelos clásicos y se acercan a la
referencia "real contra real". En la estructura de retardos (ventana/intervalo de retardo ITU, taps
significativos, retardo medio) el gaussiano empata o gana.

---

## 2. Datos

| Elemento | Valor |
|---|---|
| Fichero de medida | `data/medidas/ch_sodupe_180_time.npy` (no está en GitHub por tamaño; lo aporta el autor) |
| Contenido | Matriz compleja de forma (1024 taps, 14218 canales): taps en el eje 0, cada columna es una CIR estimada |
| Escenario | "sodupe_180": **[NO VERIFICADO]** lugar / configuración exacta de la medida (antenas, separación, potencia, entorno). El autor debe describirlo |
| Tipo de canal | Loopback (canal de autointerferencia) de un sistema IBFD. Solo se usan datos de loopback |
| Ancho de banda | 8 MHz (DVB-T2). Frecuencia de muestreo fs = 8/7 · 8 MHz = 9.142857 MHz → **periodo de tap 109.375 ns** (`sampling.bandwidth_hz` en `evaluation_config.json`; fórmula en `evaluation.sampling_period`) |
| Taps conservados | L = 200 (21.875 µs) |
| Intervalo temporal entre CIR consecutivas | **7.616 ms** según el artículo de la campaña (Bilbao et al., *Measurement* 234, 2024; PP2, Dy = 2); **no consta en el fichero** (`recordings.py --cir-period`). Duración total del fichero ≈ 108 s. El fichero **no es continuo**: 109 tramos (§2.3) |
| Frecuencia portadora, equipo de medida, método de estimación de la CIR | **[NO VERIFICADO]** no constan |

En el repositorio hay un PDF `GANs for channel data generation.pdf` (trabajo previo del autor, **[NO
VERIFICADO]**: no se ha revisado su contenido).

### 2.1 Preprocesado (`preprocess_dataset.py`, función `load_loopback_data`)

Orden exacto de las operaciones:
1. Carga del `.npy` complejo y transposición a (canales, taps).
2. Comprobación: tap de máxima energía media y fracción de energía en los primeros L taps (aviso si
   < 50 %).
3. Recorte a los primeros **L = 200 taps**.
4. **Filtrado de canales anómalos**: se descartan los canales cuya energía (en los 200 taps) está más
   de **8 dB por debajo de la mediana** (`--min-energy-db -8`). Se descartan **19 de 14218** → quedan
   **14199**. Motivo: son caídas de señal que la normalización por energía amplificaría.
5. **Normalización de energía por canal**: cada canal se divide por su norma (energía unidad).
6. Se apilan [Re, Im] → (N, 2, L) y se divide todo el conjunto por el máximo valor absoluto global
   (rango final [-0.711, 1.0]). Consecuencia: **todos los canales tienen exactamente la misma energía**
   (1.10767 en las unidades del fichero; comprobado). Esto se tiene en cuenta en los modelos clásicos
   (§3.6).
7. Salida: `data/datos_loopback/ch_sodupe_180_time_L200.npy`, float32 (14199, 2, 200).
8. Figura de verificación del preprocesado (2×2) generada por `verify_preprocessing` (desactivable con
   `--no-plots`).

Estadística básica del dataset preprocesado [AD HOC]: el PDP medio tiene su máximo en el **tap 0**
(acoplamiento directo) y cae **~54 dB** hasta el final de la ventana de 200 taps. La componente fija
(media compleja de todos los canales) está **2.36 dB por debajo** de la potencia total media (calculado
sobre el `.npy` original, 14218 canales, 200 taps).

### 2.2 Separación entrenamiento / validación / test (`make_block_split`)

- Los canales consecutivos están muy correlados (correlación ≈ 0.99 entre vecinos), así que una
  separación aleatoria por canal filtraría información. Se reparten **bloques de 500 canales
  consecutivos** al azar (semilla 42) en proporción **80/10/10** y se descartan **200 canales en cada
  frontera** entre bloques de conjuntos distintos (100 a cada lado; `--split-block 500 --split-gap 200`).
- Resultado (fichero `data/datos_loopback/ch_sodupe_180_time_L200_split.npz`):
  **train 10399, val 1100, test 900** (se pierde ~13 % de los canales).
- Matiz: los índices se refieren al dataset **ya filtrado** (14199 canales); los 19 canales descartados
  hacen que la distancia temporal real entre algunos canales "consecutivos" sea mayor. Efecto
  despreciable para los huecos, pero conviene saberlo.
- Separación anterior (huecos de 100) conservada en `..._split_gap100.npz` (train 10799, val 1300,
  test 1200); sus resultados están en `eval_results/sodupe_180_split/` y han quedado **sustituidos**.

**Justificación del hueco de 200 — coherencia temporal** (`evaluation_models/recordings.py`,
`recordings_coherence.csv`; sustituye a la tabla [AD HOC] anterior, que mezclaba pares de tramos
distintos, §2.3). Correlación de la parte variable v = x − media global entre la CIR *n* y la *n+k*
(autocorrelación por tap promediada con la potencia de cada tap; canales conservados, 200 taps), usando
**solo pares del mismo tramo continuo**; entre paréntesis, con todos los pares (cálculo antiguo):

| k (CIR) | 1 | 5 | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t (s) | 0.008 | 0.038 | 0.076 | 0.152 | 0.228 | 0.381 | 0.571 | 0.762 | 1.142 | 1.523 | 2.285 |
| ρ mismo tramo | 0.989 | 0.982 | 0.975 | 0.952 | 0.921 | 0.846 | 0.751 | 0.664 | 0.503 | 0.295 | 0.345 |
| ρ todos los pares | 0.985 | 0.941 | 0.890 | 0.784 | 0.677 | 0.478 | 0.303 | 0.208 | 0.071 | 0.018 | 0.009 |
| pares mismo tramo | 14087 | 13648 | 13103 | 12025 | 10949 | 8890 | 6709 | 5082 | 2575 | 1082 | 227 |

| Umbral | Mismo tramo | Todos los pares (antiguo) |
|---|---|---|
| ρ = 0.9 | k = 36 (0.27 s) | k = 9 (0.07 s) |
| ρ = 0.5 | k = 151 (1.15 s) | k = 48 (0.36 s) |
| ρ = 0.1 (descorrelación) | **no se alcanza** (los tramos son más cortos: mediana 127 CIR) | k = 137 (1.04 s) |

- La coherencia **dentro** de un tramo es ~3–4 veces mayor que la estimada antes: la caída rápida de la
  tabla antigua se debía sobre todo a los saltos entre tramos, no a la evolución del canal.
- Para k ≥ 200 hay pocos pares (pocos tramos tan largos) y la estimación es ruidosa (sube a 0.35 en k =
  300 y 0.69 en k = 400 con 52 pares).
- Restar la media de cada tramo (columna "media local" del CSV) da una coherencia sesgada a la baja
  (k = 4 para 0.9) porque la media local absorbe la variación lenta; no se usa.
- **Consecuencia para el hueco de 200**: dentro de un tramo, ρ ≈ 0.3 a 200 CIR, así que el hueco **no**
  descorrela por sí solo. La independencia efectiva entre conjuntos viene de los cortes: de 109 tramos,
  solo **2 se reparten entre train y test** (0 train–val, 0 val–test; §2.3).

**Por qué bloques de 500 y no más pequeños** [AD HOC]. Simulación con 20 semillas de reparto, hueco 200:

| Bloque | Canales perdidos | Desviación media \|RMS delay medio de test − total\| |
|---|---|---|
| 300 | 25 % | 39 ns |
| 500 | 13 % | 32 ns |
| 1000 | 10 % | 32 ns |

El RMS delay medio por bloque de 500 (sin umbral) varía entre 509 y 746 ns a lo largo de la medida.
Con 109 tramos de estadística distinta (§2.3), test (900 canales) contiene solo **8 tramos** (3 completos)
y su media se desvía de la global, sea cual sea el bloque. Intercalar más fino solo pierde datos.

### 2.3 Estructura en tramos continuos (`evaluation_models/recordings.py`)

Ficheros en `eval_results/sodupe_180_split_g200/`: `recordings.txt` (resumen), `recordings_segments.csv`
(un tramo por fila: inicio/fin en el `.npy` original, longitud, canales conservados, instante, conjunto,
retardos medios), `recordings.npz` (índice de tramo de cada CIR del `.npy` original, `segment_raw`, y del
dataset filtrado, `segment_filtered`; cortes; cociente r), figura `fig_recordings.{pdf,png}`.

**Detección.** Para cada posición *n* se calcula r = mediana del cambio entre CIR a través de *n*
(1 − |corr| entre CIR a ambos lados, ventana w = 3) / mediana del cambio entre CIR consecutivas a cada
lado. log10 r es **bimodal**: la gran mayoría en ~0.5 (r ≈ 3) y un grupo separado en ~3 (r ≈ 10³).
- Corte si r > 100 → 121 cortes.
- Los tramos de < 10 CIR (transitorios) se unen al siguiente → **108 cortes, 109 tramos**.
- Robustez al umbral: r > 30: 123 cortes; > 100: 121; > 300: 108; > 1000: 96. Las fronteras son claras.
- La parte variable está correlada entre los extremos de un mismo tramo (mediana 0.47) y anticorrelada a
  través de un corte (mediana −0.20): son discontinuidades reales del canal, no ruido.

**Longitud de los tramos** (CIR): mínimo 11, p25 65, **mediana 127**, p75 191, máximo 452, media 130.4
(≈ 0.99 s con 7.616 ms; máximo 3.44 s).

**No coinciden con la hipótesis de grabaciones de 5 s** (Bilbao et al.: 5 s cada minuto durante una
hora → ~657 CIR por grabación, ~21.7 grabaciones en el fichero):
- ningún intervalo entre cortes es múltiplo de 656 CIR (0 de 107);
- no hay periodo fijo (estadístico de Rayleigh máximo 0.265, por debajo del percentil 95 con cortes al
  azar, 0.284);
- 34 de 107 intervalos están a ±2 CIR de un múltiplo de 64 (al azar, ~8), lo que sugiere un origen en el
  procesado/almacenamiento por bloques.
- Origen de los cortes **[NO VERIFICADO]**: hay que preguntarlo a los autores de la campaña. Hasta
  entonces, describir el fichero como "109 tramos continuos de ~1 s" y no como "N grabaciones".

**Tramos por conjunto** (separación `..._split.npz`):

| Conjunto | Canales | Tramos distintos | Completos | Parciales |
|---|---|---|---|---|
| train | 10399 | 85 | 77 | 8 |
| val | 1100 | 12 | 8 | 4 |
| test | 900 | **8** | 3 | 5 |

Tramos compartidos: train–val 0, **train–test 2**, val–test 0.

**Variación entre tramos** (medias por tramo, ns; eta² = fracción de la varianza por canal explicada por
el tramo):

| Magnitud | Global | Mín | Mediana | Máx | Desv. entre tramos | eta² |
|---|---|---|---|---|---|---|
| RMS delay, sin umbral | 602.6 | 400.5 | 596.3 | 902.4 | 101.4 | 0.60 |
| RMS delay, 20 dB | 209.4 | 107.1 | 202.9 | 288.5 | 38.3 | 0.81 |
| Retardo medio, sin umbral | 200.4 | 96.1 | 197.8 | 334.7 | 51.0 | 0.67 |

Con todos los umbrales eta² = 0.50–0.81: **la mayor parte de la variabilidad del canal es entre tramos**.
Medias por conjunto (RMS delay sin umbral / 20 dB / retardo medio, ns): train 606.7 / 210.4 / 199.6; val
614.7 / 207.3 / 205.4; test 547.1 / 206.2 / 210.5.

**Implicaciones.**
- Las unidades estadísticamente independientes son los ~109 tramos, no las 14199 CIR; test tiene 8.
- El bootstrap por bloques debe compararse con el bootstrap por tramo (§6.10).
- Doppler / correlación temporal: solo dentro de un tramo (resolución ~1/(127 · 7.616 ms) ≈ 1 Hz).
- Memorización (§6.7): la referencia train–train debería excluir los vecinos del mismo tramo
  (pendiente).

---

## 3. Modelos

Código en `train_models/`. Lanzador: `python train_models/train.py --model <gan|dcgan|wgan|ddpm|ddim>`.
Configuraciones: `train_models/models_configs/gans_config.json` (GAN, DCGAN, WGAN) y `dm_config.json`
(DDPM, DDIM). Cada modelo guarda en `<paths.save_dir>/<GAN_Conv1D|DCGAN_Conv1D|WGAN_Conv1D|DDPM|DDIM>/`.

### 3.1 Normalización de entrada común: por tap (`train_models/normalization.py`, `TapScaler`)

Para cada tap *l* y cada componente *c* ∈ {Re, Im}:
z = clip( (x − media[c,l]) / std[c,l] / s , −1, 1 ),
con media y desviación típica calculadas **solo con train**, y *s* = percentil **99.99** de
|(x − media)/std| sobre train (s ≈ **4.78–4.81** según la separación; se recorta el **0.0100 %** de los
valores). El `scaler.npz` se guarda junto a cada checkpoint y la evaluación deshace la transformación
(`inverse`). Rango [-1, 1] compatible con la salida tanh de las GAN y con el *clamp* de los muestreadores
de difusión.

### 3.2 GAN (`train_GAN_Conv1D.py`)

- **Generador** (249 762 parámetros): z ∈ R⁶⁴ ~ N(0, I) → Linear(64, 128·25) → reshape (128, 25) →
  BN + ReLU → ConvT1d(128→64, k4, s2, p1) + BN + ReLU → ConvT1d(64→32, k4, s2, p1) + BN + ReLU →
  ConvT1d(32→2, k4, s2, p1) → tanh. Salida (2, 200).
- **Discriminador**: spectral_norm en todas las capas; Conv1d(2→32, k4 s2 p1) + LeakyReLU(0.2) →
  Conv1d(32→64) + LReLU → Conv1d(64→128) + LReLU → Linear(128·25 → 1).
- Pérdida: BCEWithLogits (GAN estándar no saturante: G maximiza log D(G(z))). D: 0.5·(L_real + L_fake).
- Adam lr **5e-5**, betas (0.5, 0.999); batch 64 (drop_last); pesos iniciales N(0, 0.02).

### 3.3 DCGAN (`train_DCGAN_Conv1D.py`)

- **Generador** (315 618 parámetros): Linear(64, 128·25) → BN + ReLU → ConvT1d(128→128, k4 s2 p1) +
  BN + ReLU → ConvT1d(128→64) + BN + ReLU → ConvT1d(64→32) + BN + ReLU → ConvT1d(32→2, k3, p1) → tanh.
- **Discriminador**: igual que el de la GAN (spectral_norm).
- Pérdida: **mínimos cuadrados** (MSE, LSGAN) con etiqueta real suavizada a **0.9** y falsa 0.0 (el
  comentario del código dice 0.1, pero el código usa 0.0). Se suma ruido N(0, 0.001²) a las muestras
  reales que ve D.
- Adam lr 5e-5, betas (0.5, 0.999); batch 64.

### 3.4 WGAN-GP (`train_WGAN.py`)

- **Generador** (592 578 parámetros): Linear(64, 256·25) → BN(256) + ReLU → ConvT1d(256→128, k4 s2 p1) +
  BN + ReLU → ConvT1d(128→64) + BN + ReLU → ConvT1d(64→32) + BN + ReLU → Conv1d(32→32, k3 p1) + BN +
  ReLU → Conv1d(32→2, k3 p1) → tanh. Longitud exacta L.
- **Crítico**: Conv1d(2→64, k4 s2 p1) + LReLU(0.2) → 64→128 → 128→256 → 256→512 (cada una con LReLU) →
  Linear(512·12 → 1). Sin normalización (WGAN-GP).
- Pérdida: Wasserstein con penalización de gradiente **λ_gp = 10**, **n_critic = 5**. El generador
  añade una pérdida espectral: L_G = −E[C(G(z))] + **λ_spec = 0.1** · MSE( log|rfft(real)|²,
  log|rfft(fake)|² ) calculada a lo largo de los taps, por componente Re/Im, en el espacio normalizado.
  Ojo: la MSE espectral compara muestras reales y generadas **emparejadas por posición en el batch**
  (no emparejadas en sentido estadístico); es lo que hace el código.
- Adam lr **1e-4**, betas (0.0, 0.9); batch 64.

### 3.5 DDPM y DDIM (`train_DDPM.py`, `train_DDIM.py`) — misma red

- **U-Net 1D** (647 682 parámetros): ConvBlock = [Conv1d k3 p1 + GroupNorm(min(8,C)) + SiLU] × 2.
  down1: 2→64; MaxPool(2); down2: 64→128; MaxPool(2); mid: 128→256; up1: interp + concat(x2) →
  ConvBlock(384→128); up2: interp + concat(x1) → ConvBlock(192→64); final Conv1d(64→2, k1).
  Embedding temporal sinusoidal de dimensión 64 → Linear(64, 256) + SiLU, **sumado solo en el
  cuello de botella** (mid).
- Difusión: **T = 1000**, schedule **coseno** (s = 0.008, betas recortadas a [1e-4, 0.999]); la red
  predice el ruido ε.
- **DDPM**: pérdida MSE(ε̂, ε) estándar. (Las claves `use_tap_weights` y `lambda_psd` de la
  configuración **no** las usa el DDPM.)
- **DDIM**: pérdida = MSE ponderada por tap, w_l = 1 + 2·PDP_norm(l) (PDP de los datos de
  entrenamiento **ya normalizados por tap**, por lo que los pesos quedan casi uniformes) + **λ_psd =
  0.05** · MSE( log1p(PSD normalizada de x̂0), log1p(PSD normalizada de x0) ), con x̂0 estimado y
  recortado a [-1, 1] y la PSD = |rfft|² a lo largo de los taps normalizada a suma 1.
- Ambos: Adam lr **2e-4**, batch 64, *gradient clipping* a norma **1.0**.
- **Muestreo DDPM** (`evaluation.generate_signals`): muestreador estándar *clip_denoised*: en cada paso se
  estima x̂0, se recorta a [-1, 1] y se muestrea de la posterior q(x_{t−1} | x_t, x̂0) con su media y
  varianza (β_t(1−ᾱ_{t−1})/(1−ᾱ_t)). 1000 pasos.
- **Muestreo DDIM**: **250 pasos, η = 0** (determinista), con x̂0 recortado a [-1, 1] en cada paso.
  Elegido con el barrido de §5.3.

### 3.6 Modelos clásicos de referencia (`evaluation_models/baselines.py`), ajustados con train

- **TDL-Rice** ("tdl"): línea de retardos con dispersión incorrelada (WSSUS). Cada tap independiente,
  h_l = μ_l + CN(0, σ_l²), con μ_l = media compleja del tap y σ_l² la potencia de la parte difusa.
  Reproduce exactamente el PDP y el factor K de Rice de cada tap, no la correlación entre taps.
- **Gaussian** ("gauss"): vector gaussiano real de dimensión 2L = 400 con la media y la **covarianza
  completa** de [Re(h), Im(h)] (incluye correlación entre taps e impropiedad). Es el mejor modelo posible
  con estadística de hasta segundo orden. Muestreo con la descomposición en autovalores de la covarianza
  (autovalores negativos recortados a 0).
- Las muestras de ambos se **reescalan a la energía constante de los canales del dataset** (igual que
  los reales tras el preprocesado). Es una operación no lineal; se aplica para que la comparación sea
  justa (todas las fuentes cumplen la misma restricción de energía).

### 3.7 Entrenamiento, selección del modelo y early stopping

- Con separación (`dataset.split_file`), se entrena **solo con train**; el scaler se ajusta con train.
- **GAN / DCGAN / WGAN**: el mejor checkpoint y el early stopping se deciden con una métrica de calidad
  de los canales generados (`train_models/quality.py`), no con la pérdida del generador:
  score = PDP_MAE_dB + 10 · KS(RMS delay spread), sobre 2000 canales generados con **ruido fijo**
  (semilla 1234) en cada época, frente a **val** (en escala original, tras deshacer el scaler). PDP_MAE_dB
  sobre los taps por encima de −60 dB del pico del PDP de referencia; RMS delay en unidades de tap, sin
  umbral. Paciencia **40 épocas**, min_delta **0.01**, máximo **300 épocas**.
- **DDPM / DDIM**: early stopping con la **pérdida de validación** (misma pérdida de entrenamiento sobre
  val con timesteps y ruido fijos, semilla 0, sin alterar el estado aleatorio del entrenamiento;
  `diffusion_val_loss`). Paciencia **30**, min_delta **5e-6**, máximo **500 épocas**.
- DDPM, DDIM y WGAN guardan `last.pth` cada época (reanudación tras interrupciones; el WGAN final se
  interrumpió y reanudó una vez).
- Semilla: en la ejecución original (semilla 42) los scripts de GAN, DCGAN y WGAN hacen
  `torch.manual_seed(42)` (`experiment.seed`), pero los de **DDPM y DDIM no fijaban ninguna semilla**
  (esa ejecución de DDPM/DDIM no es reproducible bit a bit). Después se añadió `--seed N` a
  `train_models/train.py`: fija `experiment.seed = N`, siembra `random`, `numpy`, `torch` y `cuda` antes de
  llamar al script del modelo y guarda en `<save_dir>/seed{N}/`. Las semillas 1–4 se entrenaron así
  (todos los modelos sembrados). La semilla también fija el ruido de generación (3000 canales) y el
  muestreo de los clásicos. No se ha comprobado el determinismo bit a bit.
- **5 entrenamientos por modelo** (semillas 42, 1, 2, 3, 4), mismos hiperparámetros, mismos datos y
  misma separación. Logs de las semillas 1–4: `logs/seeds/train_<modelo>_seed{N}.log`. Épocas hasta el
  early stopping en las semillas 1–4 (de los logs): GAN 64/68/71/74; DCGAN 63/53/84/83; WGAN 83/89/61/63.
  Los tiempos de entrenamiento de las semillas 1–4 **no son fiables** (varios entrenamientos se
  interrumpieron por el límite de 2 h de las tareas y se reanudaron desde `last.pth`); para el coste se
  usan los de la semilla 42 (tabla siguiente).
- Hardware: CPU Intel Xeon @ 2.10 GHz, 4 hilos, PyTorch 2.15.0 (cu132, ejecutado en CPU).

Épocas y tiempos de la ronda final (`eval_results/sodupe_180_split_g200/training_log.json`, tomados
de `logs/train_<modelo>.log`):

| Modelo | Épocas hasta early stopping | Mejor época | Tiempo de entrenamiento |
|---|---|---|---|
| GAN | 75 | 35 | 6.6 min |
| DCGAN | 81 | 41 | 9.0 min |
| WGAN | 63 | 23 | 100.9 min (20.0 + 80.9, reanudado) |
| DDPM | 74 | 44 | 23.7 min |
| DDIM | 67 | 37 | 21.3 min |

Los tiempos de las GAN incluyen la evaluación de calidad de cada época (2000 canales generados + métricas).

### 3.8 Hiperparámetros: cómo se eligieron

**No se hizo búsqueda de hiperparámetros sistemática.** Existe la herramienta
(`train_models/hparam_search.py` y `train_models/hparam_specs/*.json`, reanudable, evalúa contra val),
pero no se ejecutó: tras introducir la normalización por tap todos los modelos quedaron al nivel de la
referencia real y se decidió no hacerla. Los hiperparámetros son los de la configuración heredada del
proyecto, con estos cambios justificados durante el desarrollo: z_dim 64; lr del WGAN 1e-4; épocas
máximas y paciencias de early stopping. Lo único que se barrió fueron los parámetros de **muestreo** del
DDIM (§5.3), que no requieren reentrenar. Amenaza a la validez: los modelos no están ajustados por igual
(ver §8).

---

## 4. Evaluación

Código: `evaluation_models/evaluation.py` (configuración `evaluation_models/evaluation_config.json`).

- **Referencia real: test (900 canales)**. Cada modelo genera **3000 canales** (semilla 42), que se
  pasan a escala original con el inverso del scaler.
- Distancias entre distribuciones: estadístico de **Kolmogorov–Smirnov** de dos muestras (900 reales vs
  3000 generados). 0 = idénticas, 1 = disjuntas.
- **Referencia "real contra real"** (`baseline_real_vs_test_full.txt`): 3000 canales de train (al azar,
  semilla 42) y los 1100 de val evaluados contra test con las mismas funciones, como si fueran
  "generados". Es el error que tendría un modelo que reprodujera exactamente la distribución de
  train/val; necesaria porque test es un tramo distinto de la medida (deriva lenta del canal, §2.2).

### 4.1 Métricas (definiciones exactas)

h = CIR compleja de 200 taps; p_l = |h_l|²; τ_l = l · 109.375 ns.

- **PDP** = E[p_l]. *PDP MAE (dB)*: media de |PDP_real,dB − PDP_gen,dB| sobre los taps por encima de
  −60 dB del máximo del PDP real. También MAE/MSE lineales.
- **Energía**: 10·log10(E[energía generada]/E[energía real]).
- **Retardo medio** τ̄ = Σ p τ / Σ p y **RMS delay spread** σ_τ = sqrt(Σ p (τ−τ̄)² / Σ p), por canal.
  Retardos medidos desde el tap 0 (no desde la primera llegada). Se calculan **sin umbral** y con
  umbrales **40, 30, 25 y 20 dB** bajo el pico **de cada canal** (los taps por debajo se ponen a 0).
  Se dan medias, MAE, KS y t-test de Welch.
- **|h|**: KS entre las distribuciones de |h_l| agrupando todos los taps (submuestreo de 50 000 valores);
  KDE y CDF (lineal y en dB).
- **Respuesta en frecuencia**: H = FFT de 200 puntos a lo largo de los taps. *E|H(f)|² MAE (dB)* referido
  al máximo real.
- **Correlación en frecuencia** R(Δf) = |DFT del PDP medio| normalizada (Wiener–Khinchin); MAE.
  **Ancho de banda de coherencia** Bc90 y Bc50: primer Δf con R < 0.9 / 0.5 (interpolado), Δf = fs/200
  = 45.71 kHz.
- **Nulos** (|H(f)|² de cada canal normalizado a su media en la banda): *profundidad del nulo* =
  min_f |H|² (dB); *ganancia de ruido ZF* = 10·log10 E_f{1/|H(f)|²}. Mediana y KS.
- **Curtosis en exceso de la parte difusa** (h − media del tap), media de Re e Im. Tap del pico aparte;
  para el resto (taps a menos de 30 dB del pico del PDP real), mediana y MAE por tap.
- **Parámetros por canal** (`per_channel_params`; mediana y KS):
  - *Rizado* max_f |H|² / min_f |H|² (dB).
  - *Fracción de banda en fade*: fracción de subportadoras con |H|² < media − 10 dB.
  - *Eficiencia espectral* a 20 dB: E_f{log2(1 + 100·|H|²)} (b/s/Hz), y su percentil 10 (*outage* 10 %).
  - *Bc90 por canal*: igual que Bc90 pero con la DFT del perfil |h|² del propio canal.
  - *Ventana de retardo 90 %* (ITU-R P.1407): anchura del intervalo central que contiene el 90 % de la
    energía (del 5 % al 95 % de la energía acumulada), en ns.
  - *Intervalo de retardo* (ITU-R P.1407) a 40/30/25/20 dB: del primer al último tap por encima de
    pico − X dB, en ns.
  - *Taps con el 90 % de la energía*: número mínimo de taps que la reúnen.
- **BER** (`evaluation_models/BER.py`): OFDM de **200 subportadoras** (FFT de la CIR de 200 taps; se
  supone prefijo cíclico suficiente, sin ISI), **CSI perfecta**, **ecualización ZF**, cada canal
  normalizado a energía unidad, 1 símbolo OFDM por canal, **QPSK y 16-QAM** (mapeo Gray según el código
  de las curvas teóricas), SNR = Es/N0 de 0 a 30 dB en pasos de 2. Se usan **900 canales** por fuente
  (el número de canales de test; para los modelos, los 900 primeros generados) y **los mismos bits y
  el mismo ruido** para todas las fuentes. 360 000 bits por punto en QPSK y 720 000 en 16-QAM. Métrica
  resumen: |Δlog10 BER| medio sobre los puntos de SNR. Se dibujan también las curvas teóricas AWGN y
  Rayleigh.
- **Memorización** (`evaluation_models/memorization.py`): distancia de cada canal generado al canal de
  train más cercano (vecino más próximo), en dos espacios: *norm* (espacio normalizado por tap,
  euclídea / sqrt(2L)) y *rel* (espacio original, ||g − t|| / ||t||). Referencia: cada canal de train
  (2000 al azar, semilla 0) frente al resto de train **excluyendo sus vecinos temporales ±200 medidas**.
  *ratio* = mediana del modelo / mediana de la referencia; *copias %* = % de canales con distancia menor
  que el percentil 1 de la referencia (la referencia da 1 % por definición). Test se da como segunda
  referencia.
- **Coste** (`evaluation_models/model_costs.py`): parámetros contados en el checkpoint (generador o
  U-Net), tiempo de generación medido con 500 canales tras un calentamiento (CPU, 4 hilos; los clásicos
  incluyen el ajuste).

### 4.2 Variabilidad: semillas y bootstrap por bloques (`evaluation_models/seed_stats.py`)

Dos fuentes de incertidumbre, tratadas por separado:

1. **Entrenamiento** (semilla): 5 semillas por modelo (§3.7). `seeds_metrics_summary.txt/.csv`: media y
   desviación típica (ddof = 1) entre las 5 ejecuciones de **todas** las métricas de `metrics.pkl`
   (las mismas que en §6.1–6.6, calculadas por `evaluation.py` en cada semilla).
2. **Muestreo del test** (pocos tramos independientes, §2.2): **bootstrap por bloques**, 1000 réplicas.
   En cada réplica:
   - test (900 canales, en orden temporal) se remuestrea en **bloques de 50 canales consecutivos**
     (18 bloques con reemplazo), para respetar la correlación temporal (elegido antes de conocer los tramos
     de §2.3; dentro de un tramo la coherencia es mayor, ver la sensibilidad al bloque más abajo);
   - train y val (referencias reales) se remuestrean igual, por bloques de 50;
   - los canales generados (i.i.d.) se remuestrean canal a canal, de forma independiente para cada semilla;
   - se recalculan las métricas frente al test remuestreado y se promedian entre las 5 semillas.
   El generador aleatorio de la réplica b es `default_rng([42, b])` (resultado independiente del número de
   procesos; las réplicas se pueden reanudar).
   - **IC95**: percentiles 2.5 y 97.5 de la media entre semillas.
   - **P(< Gaussian)**: fracción de réplicas en que la fuente tiene una métrica menor (mejor) que el
     gaussiano **en la misma réplica** (comparación emparejada: el mismo test remuestreado para ambos).
     P > 0.975 o < 0.025 = diferencia significativa al 5 % (bilateral).
   Métricas del bootstrap (todas "menor es mejor"): PDP MAE (dB, taps por encima de −60 dB del pico del
   PDP de test completo); KS de |h| (sobre un histograma de |h|² en dB con resolución 0.1 dB entre −120 y
   20 dB, todos los taps); KS de cada rasgo por canal (retardo medio y RMS con todos los umbrales, nulo,
   ganancia ZF, parámetros por canal de §4.1); |Δlog10 BER| medio sobre 0–30 dB con **BER por canal**
   (`per_channel_ber`, réplica de `BER.ber_curve` canal a canal; aquí se usan **todos** los canales de
   cada fuente, no los 900 primeros, por lo que las cifras de BER difieren algo de `ber_table.txt`).
   Ficheros: `bootstrap_ci.txt/.csv`. Los rasgos por canal se calculan una vez por fuente.
   **Sensibilidad al bloque** (`seed_stats.py --block 25 | 100 | 500 | segment`, ficheros
   `bootstrap_ci_block<B>.txt/.csv`; comparación con `bootstrap_sensitivity.py` →
   `bootstrap_sensitivity.txt/.csv`). Con `--block segment` cada bloque es un tramo continuo de §2.3 (test,
   train y val se remuestrean por tramos; el tamaño del conjunto remuestreado varía). Resultados en §6.10.

**Sesgo conocido**: KS y |Δ| son distancias (≥ 0); al remuestrear, el ruido las aumenta, de modo que la
distribución bootstrap está desplazada hacia arriba y el valor puntual (datos completos) puede quedar
cerca o por debajo del límite inferior del IC95. Por eso **la comparación entre modelos debe basarse en
P(< Gaussian)** (emparejada), no en el solapamiento de los IC95: la mayor parte de la anchura de los IC95
es la variabilidad del test, común a todas las fuentes.

---

## 5. Decisiones de diseño, alternativas descartadas e historia del desarrollo

Rondas de entrenamiento (todas en el repositorio):

| Ronda | Normalización | Referencia | Carpeta de resultados | Estado |
|---|---|---|---|---|
| 1 | Global (energía por canal + máximo global) | todo el dataset (incluye train) | `eval_results/sodupe_180/` | descartada |
| 2 | Por tap | todo el dataset (incluye train) | `eval_results/sodupe_180_tapnorm/` | superada |
| 3 | Por tap + separación, huecos de 100 | test (1200) | `eval_results/sodupe_180_split/` | sustituida |
| **4 (final)** | **Por tap + separación, huecos de 200** | **test (900)** | **`eval_results/sodupe_180_split_g200/`** | **final** |

### 5.1 Normalización por tap (decisión clave)

- **Problema (ronda 1)**: el PDP abarca ~54 dB. Con normalización global, los taps de la cola tienen
  amplitudes ~1e-3 y el error residual de los modelos de difusión queda por encima: aparece un **suelo
  de ruido en el PDP generado a −45/−50 dB**.
- Errores del PDP en la ronda 1 (frente a todo el dataset): GAN **1.732 dB**, DCGAN **3.370**, WGAN
  **5.645**, DDPM **4.693**, DDIM **4.155**. KS del RMS delay: 0.101 / 0.112 / 0.989 / 0.570 / 0.292.
  KS de |h|: 0.036 / 0.040 / 0.296 / 0.205 / 0.178.
- **Solución**: TapScaler (§3.1). Ronda 2 (frente a todo el dataset, **dentro de muestra**): PDP
  0.312 / 0.428 / 0.383 / 0.500 / 0.721 dB; KS RMS delay 0.051 / 0.030 / 0.017 / 0.152 / 0.208;
  KS |h| 0.012 / 0.009 / 0.020 / 0.022 / 0.037 (GAN / DCGAN / WGAN / DDPM / DDIM, DDIM con 100 pasos).
- **Alternativa descartada**: escalar por el **máximo** en lugar del percentil 99.99. La escala salía de
  12 a 25 (hay valores aislados a ~25σ, p. ej. en la parte imaginaria del tap 0) y se desperdiciaba casi
  todo el rango [-1, 1].

### 5.2 Muestreador del DDPM

- **Alternativa descartada**: el muestreador original, μ = (x_t − β_t/√(1−ᾱ_t) ε̂)/√(1−β_t) con
  varianza β_t, sin recortar x̂0. Con el schedule coseno, β_T ≈ 0.999 y el primer paso divide por
  √(1−β) ≈ 0.03, lo que amplifica el error de la red. Con normalización por tap daba un error del PDP de
  **~10.7 dB** [AD HOC].
- **Solución**: muestreador estándar *clip_denoised* (§3.5): **0.54 dB** con 300 muestras [AD HOC] y
  0.500 dB en la evaluación de la ronda 2 (3000 muestras).

### 5.3 Parámetros de muestreo del DDIM (`evaluation_models/sweep_sampling.py`)

Barrido sobre el checkpoint de la **ronda 2** (`checkpoints/sodupe_180_tapnorm/DDIM`), 1000 canales,
referencia = todo el dataset. Fichero: `eval_results/sodupe_180_tapnorm/sweep_sampling_ddim.txt`.
selection_score = PDP_MAE_dB + 10·(KS_rms + KS_|h|).

| Pasos | η | Score | PDP MAE (dB) | KS RMS | KS \|h\| | Tiempo (s, 1000 canales) |
|---|---|---|---|---|---|---|
| 25 | 0 | 3.583 | 0.815 | 0.246 | 0.031 | 19.6 |
| 50 | 0 | 3.458 | 0.764 | 0.239 | 0.031 | 27.0 |
| 100 | 0 | 2.962 | 0.638 | 0.208 | 0.024 | 52.3 |
| **250** | **0** | **1.784** | **0.333** | **0.133** | **0.012** | 132.5 |
| 500 | 0 | 2.835 | 0.712 | 0.166 | 0.046 | 294.7 |
| 1000 | 0 | 9.829 | 2.920 | 0.529 | 0.162 | 541.9 |
| 250 | 0.5 | 3.385 | 0.805 | 0.225 | 0.033 | 133.4 |
| 250 | 1 | 3.965 | 1.007 | 0.256 | 0.040 | 133.2 |

(También se probaron 25/50/100 con η = 0.5 y 1: todas peores que η = 0.) Se eligió **250 pasos, η = 0**.
La degradación con 1000 pasos no se ha investigado; hipótesis **[NO VERIFICADO]**: acumulación del error
de la red paso a paso. **Limitación**: el barrido se hizo con el modelo de la ronda 2 y contra todo el
dataset, no se repitió con el modelo final contra val.

### 5.4 Otras correcciones relevantes

- **Selección del checkpoint de las GAN**: el código original elegía el mejor modelo por la pérdida del
  generador, que no mide la calidad de los canales (depende de la fuerza del discriminador). Sustituido
  por la métrica de calidad de §3.7.
- **WGAN**: (i) el script reinicializaba los pesos después de cargar un checkpoint; (ii) el generador
  original producía un artefacto en el borde de la ventana (ronda 1: KS del RMS delay 0.989). Se
  rediseñó el generador para que produzca exactamente L muestras (init_len = L/8 y tres ConvT ×2), §3.4.
- **BER**: el calculador original hacía la FFT a lo largo del eje de canales (eje 0) en lugar del de los
  taps y usaba una SNR fija. Reescrito (§4.1).
- **Evaluación**: varias correcciones (importaba un módulo con nombre erróneo, z_dim desalineado entre
  configuración y checkpoint —ahora se lee del checkpoint—, la correlación en frecuencia se dibujaba
  frente al retardo, la CDF usaba una clave inexistente y la unidad de retardo era errónea).
- Cada modelo guarda en su propia subcarpeta (antes las tres GAN compartían carpeta y DDPM/DDIM otra, y
  se sobrescribían).

### 5.5 Umbral del delay spread

- Sin umbral, el delay spread está dominado por la cola de la ventana de 200 taps: los taps de la cola
  están a ~22 µs y su peso τ² es grande. RMS delay medio (test): **547 ns sin umbral**, 422 ns a 40 dB,
  218 ns a 30 dB, 211 ns a 25 dB, 206 ns a 20 dB.
- El ranking de los modelos **depende del umbral** (§6.2). Decisión: dar todos los umbrales (sin umbral,
  40, 30, 25 y 20 dB) en tablas y en la figura de CDF lado a lado.
- La deriva entre tramos de la medida existe con cualquier umbral (variación de ±9–12 % del RMS medio
  por bloque de 500) [AD HOC]: no es un efecto de la cola.

### 5.6 Modelos clásicos y métricas de orden superior

- Se añadió el modelo gaussiano con covarianza completa porque es la comparación natural que pediría un
  revisor (¿qué aportan los generativos sobre ajustar la estadística de segundo orden?). Resultó un rival
  fuerte en las métricas de segundo orden, lo que motivó añadir métricas que **no** dependen solo de la
  covarianza (nulos, rizado, *fade*, eficiencia espectral, Bc por canal, parámetros ITU).
- **Alternativas probadas que no discriminan**:
  - *Curtosis por tap de la parte difusa*: el tap del pico tiene curtosis enorme (93.7 en test, 64.2 en
    train, 41.2 en val [AD HOC]) por unos pocos valores atípicos esporádicos que ningún modelo reproduce;
    en el resto de taps la curtosis está entre −1 y +1 y difiere tanto entre train y test (mediana +0.32
    frente a −0.54 [AD HOC]) como entre modelos. El gaussiano queda igual que los mejores. Conclusión:
    el canal es **casi gaussiano tap a tap**; lo no gaussiano está en la estructura conjunta entre taps,
    que se manifiesta en los nulos en frecuencia.
  - *m de Nakagami por tap* (momentos de cuarto orden) [AD HOC, no incluido en evaluation.py]: MAE por
    tap frente a test: real train 0.886, GAN 0.943, DCGAN 0.919, WGAN 1.070, DDPM 1.187, DDIM 1.096,
    TDL-Rice 1.818, Gaussian 1.036. No discrimina entre generativos y gaussiano.
  - *Tasa de cruces por nivel en frecuencia* (−10 dB) [AD HOC, no incluido]: KS real train 0.132, GAN
    0.096, DCGAN 0.173, WGAN 0.139, DDPM 0.259, DDIM 0.226, TDL 0.148, Gaussian 0.091. No discrimina.

### 5.7 Lo que se decidió no hacer

- **Modelado Doppler / evolución temporal**: los modelos generan canales i.i.d. (el entrenamiento baraja
  los canales). El Doppler lo mostrará el autor en el postprocesado de las medidas, sin incluirlo en los
  modelos.
- **Caso de uso** (p. ej. entrenar un estimador o cancelador con canales sintéticos y probarlo con
  reales): descartado por extensión del artículo. Se ampliará la sección de BER en su lugar.
- **Varias semillas / validación cruzada**: no se han hecho (§8).

---

## 6. Resultados finales (ronda 4, `eval_results/sodupe_180_split_g200/`)

Referencia: test (900 canales). 3000 canales generados por modelo. Ficheros: `metrics_summary.txt`
(todas las métricas), `ber_table.txt`, `baseline_real_vs_test_full.txt` (referencia real),
`memorization.txt`, `model_costs.txt`.

### 6.1 Métricas generales

| Métrica | Real train vs test | Real val vs test | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|---|
| PDP MAE (dB) | 1.557 | 1.523 | 1.537 | 1.582 | 1.516 | 2.071 | **1.056** | 1.740 | 1.503 |
| Energía gen./real (dB) | — | — | 0.013 | −0.019 | −0.278 | −0.207 | −0.257 | 0.000 | 0.000 |
| KS \|h\| | 0.079 | 0.081 | 0.084 | 0.074 | 0.079 | 0.113 | **0.040** | 0.072 | 0.067 |
| E\|H(f)\|² MAE (dB) | — | — | 1.434 | 1.144 | 1.172 | 1.073 | **1.003** | 1.132 | 1.068 |
| Corr. en frecuencia MAE | 0.0720 | 0.0475 | **0.0541** | 0.0733 | 0.1104 | 0.1358 | 0.1155 | 0.0868 | 0.0872 |
| Error Bc90 (MHz) (real 0.3173) | 0.0123 | 0.0007 | 0.0080 | 0.0060 | **0.0024** | 0.0131 | 0.0217 | 0.0187 | 0.0155 |
| Error Bc50 (MHz) (real 1.1685) | 0.3665 | 0.0991 | **0.3019** | 0.5069 | 1.2607 | 1.6673 | 1.3719 | 0.4948 | 0.4978 |
| Retardo medio: MAE (ns) (real 210.46) | 9.97 | 5.04 | 12.17 | 16.49 | 24.05 | 30.09 | 37.19 | **8.26** | 11.97 |
| Retardo medio: KS | 0.294 | 0.351 | 0.326 | 0.342 | 0.351 | 0.395 | 0.429 | 0.350 | **0.311** |

(— = no calculado en la referencia real.)

### 6.2 RMS delay spread según el umbral (KS frente a test)

| Umbral | Real train | Real val | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|---|
| Sin umbral (real 547.09 ns) | 0.194 | 0.248 | 0.267 | 0.268 | 0.255 | 0.279 | **0.086** | 0.440 | 0.192 |
| 40 dB (real 422.07 ns) | 0.124 | 0.288 | 0.132 | 0.123 | **0.113** | 0.184 | 0.244 | 0.299 | 0.126 |
| 30 dB (real 217.77 ns) | 0.223 | 0.276 | 0.263 | 0.270 | 0.195 | 0.160 | **0.159** | 0.317 | 0.285 |
| 25 dB (real 211.15 ns) | 0.178 | 0.227 | 0.226 | 0.235 | **0.140** | 0.199 | 0.224 | 0.275 | 0.262 |
| 20 dB (real 206.23 ns) | 0.107 | 0.199 | **0.131** | 0.155 | 0.146 | 0.269 | 0.261 | 0.181 | 0.166 |

RMS medio generado (ns), sin umbral: GAN 616.72, DCGAN 618.65, WGAN 617.30, DDPM 634.43, DDIM 567.73,
TDL 633.75, Gaussian 607.67. A 20 dB: 208.93 / 206.76 / 202.41 / 191.67 / 192.36 / 217.32 / 214.20.
Las tablas completas (retardo medio y RMS con MAE y t-test para cada umbral) están en
`metrics_summary.txt`.

### 6.3 Propiedades por canal en frecuencia y nulos (KS frente a test; mediana real entre paréntesis)

| Parámetro | Real train | Real val | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|---|
| Profundidad del nulo (real −16.66 dB) | 0.186 | 0.266 | 0.294 | 0.245 | 0.250 | **0.188** | 0.221 | 0.368 | 0.325 |
| Ganancia de ruido ZF (real 4.46 dB) | 0.139 | 0.385 | 0.268 | 0.188 | 0.223 | **0.147** | 0.197 | 0.342 | 0.300 |
| Rizado \|H(f)\| (real 21.25 dB) | 0.195 | 0.242 | 0.320 | 0.262 | 0.280 | **0.242** | 0.252 | 0.360 | 0.341 |
| Fracción de banda en fade −10 dB (real 0.045) | 0.088 | 0.369 | 0.170 | **0.085** | 0.136 | 0.120 | 0.109 | 0.233 | 0.192 |
| Eficiencia espectral @20 dB (real 6.148 b/s/Hz) | 0.118 | 0.402 | 0.156 | **0.106** | 0.142 | 0.161 | 0.108 | 0.251 | 0.205 |
| Bc90 por canal (real 0.3259 MHz) | 0.202 | 0.255 | 0.230 | 0.236 | 0.158 | **0.144** | 0.199 | 0.246 | 0.234 |

Medianas generadas: nulo GAN −20.22, DCGAN −19.31, WGAN −19.44, DDPM −18.11, DDIM −18.74, TDL −21.51,
Gaussian −20.75 dB. Ganancia ZF: 5.79 / 5.17 / 5.47 / 4.87 / 5.25 / 6.33 / 6.04 dB. Rizado: 25.91 /
24.85 / 25.10 / 24.53 / 24.63 / 26.65 / 26.51 dB. Eficiencia espectral *outage* 10 % (real 5.826):
5.819 / 5.873 / 5.849 / 5.939 / 5.874 / 5.824 / 5.845 b/s/Hz.

**Todos los modelos, también los clásicos, generan respuestas en frecuencia más selectivas que las
reales** (nulos más profundos, más rizado); los generativos son los que menos se desvían.

### 6.4 Estructura de retardos ITU-R P.1407 (KS frente a test)

| Parámetro | Real train | Real val | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|---|
| Ventana de retardo 90 % (real 765.6 ns) | 0.129 | 0.419 | 0.158 | 0.154 | 0.169 | 0.305 | 0.329 | 0.193 | **0.141** |
| Taps con el 90 % de la energía (real 5) | 0.059 | 0.124 | 0.052 | 0.059 | 0.090 | 0.145 | 0.178 | 0.039 | **0.035** |
| Intervalo de retardo @40 dB (real 10171.9 ns) | 0.132 | 0.124 | 0.224 | 0.192 | 0.243 | 0.322 | 0.397 | **0.126** | 0.188 |
| Intervalo de retardo @30 dB (real 1312.5 ns) | 0.120 | 0.185 | 0.156 | 0.193 | 0.216 | 0.192 | **0.079** | 0.531 | 0.178 |
| Intervalo de retardo @25 dB (real 1093.8 ns) | 0.176 | 0.201 | 0.198 | 0.186 | 0.166 | **0.146** | 0.163 | 0.301 | 0.205 |
| Intervalo de retardo @20 dB (real 1093.8 ns) | 0.195 | 0.085 | 0.232 | 0.232 | 0.279 | 0.357 | 0.364 | **0.202** | 0.225 |

Los valores de ventana e intervalo están cuantizados en múltiplos de 109.375 ns (resolución del tap).

### 6.5 Curtosis de la parte difusa

| | Real (test) | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|
| Tap del pico | 93.73 | −0.49 | −0.32 | −0.36 | 1.33 | 2.71 | −0.07 | 0.75 |
| Mediana del resto de taps | −0.606 | 0.036 | −0.063 | −0.080 | 0.408 | 0.187 | 0.030 | −0.098 |
| MAE por tap | — | 0.625 | 0.688 | **0.534** | 1.290 | 0.835 | 0.557 | 0.556 |

Referencia real: MAE train vs test 0.833, val vs test 0.879 (los modelos quedan por debajo, porque
también difieren train y test). Conclusión en §5.6.

### 6.6 BER (OFDM ZF, `ber_table.txt`; 900 canales por fuente)

Resumen |Δlog10 BER| (media sobre 0–30 dB):

| | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|
| QPSK | 0.084 | 0.053 | 0.045 | 0.036 | **0.033** | 0.088 | 0.085 |
| 16-QAM | 0.050 | 0.025 | 0.019 | 0.019 | **0.014** | 0.047 | 0.045 |

BER en puntos concretos:

| Modulación, SNR | Real | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian | AWGN teór. | Rayleigh teór. |
|---|---|---|---|---|---|---|---|---|---|---|
| QPSK, 10 dB | 3.090e-2 | 3.295e-2 | 2.943e-2 | 3.127e-2 | 2.844e-2 | 3.029e-2 | 3.344e-2 | 3.318e-2 | 7.827e-4 | 4.356e-2 |
| QPSK, 20 dB | 2.611e-3 | 3.506e-3 | 2.961e-3 | 3.042e-3 | 2.531e-3 | 2.700e-3 | 3.292e-3 | 3.494e-3 | 7.620e-24 | 4.926e-3 |
| QPSK, 30 dB | 2.417e-4 | 3.417e-4 | 3.028e-4 | 2.889e-4 | 2.611e-4 | 2.944e-4 | 3.694e-4 | 3.611e-4 | ≈0 | 4.993e-4 |
| 16-QAM, 10 dB | 1.030e-1 | 1.046e-1 | 1.014e-1 | 1.035e-1 | 1.001e-1 | 1.015e-1 | 1.062e-1 | 1.050e-1 | 5.899e-2 | 1.098e-1 |
| 16-QAM, 20 dB | 1.198e-2 | 1.339e-2 | 1.173e-2 | 1.228e-2 | 1.078e-2 | 1.160e-2 | 1.340e-2 | 1.335e-2 | 2.904e-6 | 1.745e-2 |
| 16-QAM, 30 dB | 9.458e-4 | 1.410e-3 | 1.236e-3 | 1.144e-3 | 1.014e-3 | 1.089e-3 | 1.321e-3 | 1.335e-3 | ≈0 | 1.861e-3 |

Los dos modelos clásicos son **pesimistas** (más BER que el real: +34 % en QPSK a 20 dB para el
gaussiano), coherente con sus nulos más profundos (§6.3). DDPM y DDIM son los más cercanos. No se ha
calculado la BER de la referencia "real train vs test".

### 6.7 Memorización (`memorization.txt`, figura `fig_memorization_nn`)

| Fuente | norm: mediana | norm: ratio | norm: copias % | rel: mediana | rel: ratio | rel: copias % |
|---|---|---|---|---|---|---|
| Real train (excl. ±200), referencia | 0.1299 | 1.00 | 1.00 | 0.3812 | 1.00 | 1.00 |
| Real test | 0.1325 | 1.02 | 0.89 | 0.3133 | 0.82 | 0.00 |
| GAN | 0.1659 | 1.28 | 0.00 | 0.4025 | 1.06 | 0.13 |
| DCGAN | 0.1626 | 1.25 | 0.00 | 0.3837 | 1.01 | 0.30 |
| WGAN | 0.1526 | 1.17 | 0.00 | 0.3933 | 1.03 | 0.07 |
| DDPM | 0.1443 | 1.11 | 0.00 | 0.2527 | **0.66** | **21.27** |
| DDIM | 0.1413 | 1.09 | 0.00 | 0.3069 | 0.81 | **6.77** |
| TDL-Rice | 0.2365 | 1.82 | 0.00 | 0.4988 | 1.31 | 0.00 |
| Gaussian | 0.1465 | 1.13 | 0.00 | 0.4320 | 1.13 | 0.03 |

Lectura: en el espacio normalizado por tap ningún modelo copia. En el espacio original (dominado por los
taps fuertes), **el DDPM muestra memorización parcial** (21 % de sus canales más cerca de train que el
99 % de los pares reales independientes) y el DDIM algo (6.8 %); las GAN y los clásicos, no. Diversidad
[AD HOC]: canales de train distintos que son el vecino más próximo de los 3000 generados: GAN 1687,
WGAN 1575, DDPM 1601, DDIM 1882, Gaussian 1834 (no hay colapso a unos pocos canales; repeticiones
máximas: 37, 31, 61, 24, 15). Matiz no explicado: la mediana *rel* de test (0.313) es menor que la de la
referencia train-train (0.381) **[NO VERIFICADO el motivo]**.

### 6.8 Coste computacional (`model_costs.txt`; CPU Xeon 2.10 GHz, 4 hilos)

| Modelo | Parámetros | Generación (ms/canal) | Entrenamiento |
|---|---|---|---|
| GAN | 249 762 | 0.07 | 6.6 min |
| DCGAN | 315 618 | 0.13 | 9.0 min |
| WGAN | 592 578 | 0.17 | 100.9 min |
| DDPM (1000 pasos) | 647 682 | 586.03 | 23.7 min |
| DDIM (250 pasos) | 647 682 | 141.86 | 21.3 min |
| TDL-Rice | — | 0.13 (incl. ajuste) | — |
| Gaussian | — | 0.22 (incl. ajuste) | — |

### 6.9 Resumen de conclusiones (semilla 42; ver §6.10 para la versión con significación)

1. **Segundo orden** (PDP, correlación en frecuencia, Bc): el gaussiano de covarianza completa es el
   techo teórico; los generativos lo igualan, no lo superan (excepto el DDIM en PDP: 1.06 frente a
   1.50 dB). Todos quedan al nivel de la referencia real (PDP real train vs test 1.56 dB).
2. **Por canal en frecuencia y BER**: los generativos superan a los dos clásicos de forma consistente
   y los mejores (DDPM, DCGAN, DDIM) se acercan a la referencia real (p. ej. nulo: DDPM 0.188 vs real
   0.186; ganancia ZF: DDPM 0.147 vs real 0.139; eficiencia espectral: DCGAN 0.106 vs real 0.118).
   La BER de los generativos (excepto GAN) tiene la mitad o menos de error que la de los clásicos.
3. **Estructura de retardos**: el gaussiano (y el TDL a 40 dB) empata o gana; los de difusión son los
   peores en ventana de retardo y número de taps. **Todos los modelos subestiman el retardo medio**
   (12–37 ns, de 0.1 a 0.34 taps); el KS del retardo medio de los generativos (0.33–0.43) supera al de la
   referencia real (0.29).
4. **Por modelo**: DDIM = mejor global (PDP, |h|, BER, RMS sin umbral y a 30 dB), pero lento
   (142 ms/canal) y con indicios leves de memorización. DDPM = mejor en nulos / ganancia ZF / BER cercana
   al real, pero el más flojo en PDP (2.07 dB), con memorización parcial y el más lento. WGAN = mejor
   equilibrio (buena BER, RMS con umbral al nivel de la referencia real, sin memorización, generación
   instantánea; entrenamiento más largo). GAN básica = la más débil de los generativos en BER.

### 6.10 Resultados con 5 semillas y bootstrap por bloques (**cifras para el artículo**)

Ficheros: `seeds_metrics_summary.txt/.csv` (todas las métricas, media ± desviación entre semillas) y
`bootstrap_ci.txt/.csv` (1000 réplicas). Referencia: test. Real train / Real val: un único valor (no hay
semillas). En las tablas A y C las cifras de BER usan todos los canales de cada fuente (§4.2).

**Tabla A** — media ± desviación típica entre las 5 semillas (Real train/val: valor único)

| Métrica | Real train | Real val | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|---|
| PDP MAE (dB) | 1.555 | 1.523 | 1.551 ± 0.090 | 1.546 ± 0.120 | 1.409 ± 0.083 | 1.644 ± 0.444 | 1.024 ± 0.125 | 1.724 ± 0.015 | 1.575 ± 0.046 |
| KS \|h\| | 0.079 | 0.080 | 0.080 ± 0.008 | 0.076 ± 0.012 | 0.073 ± 0.007 | 0.082 ± 0.030 | 0.035 ± 0.012 | 0.071 ± 0.001 | 0.071 ± 0.004 |
| KS retardo medio | 0.289 | 0.351 | 0.294 ± 0.036 | 0.309 ± 0.059 | 0.366 ± 0.024 | 0.365 ± 0.044 | 0.414 ± 0.011 | 0.354 ± 0.007 | 0.304 ± 0.005 |
| KS RMS delay, sin umbral | 0.191 | 0.248 | 0.267 ± 0.008 | 0.270 ± 0.015 | 0.246 ± 0.016 | 0.212 ± 0.067 | 0.138 ± 0.054 | 0.436 ± 0.005 | 0.204 ± 0.009 |
| KS RMS delay, 40 dB | 0.117 | 0.288 | 0.154 ± 0.015 | 0.148 ± 0.015 | 0.102 ± 0.012 | 0.211 ± 0.078 | 0.190 ± 0.099 | 0.297 ± 0.003 | 0.135 ± 0.006 |
| KS RMS delay, 30 dB | 0.211 | 0.276 | 0.227 ± 0.034 | 0.218 ± 0.048 | 0.187 ± 0.016 | 0.164 ± 0.022 | 0.168 ± 0.026 | 0.317 ± 0.004 | 0.285 ± 0.004 |
| KS RMS delay, 25 dB | 0.169 | 0.227 | 0.187 ± 0.043 | 0.171 ± 0.062 | 0.130 ± 0.018 | 0.191 ± 0.047 | 0.173 ± 0.036 | 0.277 ± 0.005 | 0.264 ± 0.003 |
| KS RMS delay, 20 dB | 0.101 | 0.198 | 0.134 ± 0.011 | 0.149 ± 0.031 | 0.166 ± 0.031 | 0.221 ± 0.081 | 0.202 ± 0.046 | 0.187 ± 0.006 | 0.169 ± 0.005 |
| KS profundidad del nulo | 0.193 | 0.266 | 0.279 ± 0.053 | 0.271 ± 0.046 | 0.260 ± 0.019 | 0.188 ± 0.027 | 0.226 ± 0.013 | 0.372 ± 0.007 | 0.321 ± 0.005 |
| KS ganancia de ruido ZF | 0.154 | 0.385 | 0.251 ± 0.058 | 0.226 ± 0.052 | 0.229 ± 0.019 | 0.204 ± 0.048 | 0.194 ± 0.012 | 0.346 ± 0.010 | 0.293 ± 0.008 |
| KS rizado | 0.205 | 0.242 | 0.292 ± 0.055 | 0.278 ± 0.043 | 0.280 ± 0.020 | 0.203 ± 0.042 | 0.237 ± 0.015 | 0.369 ± 0.010 | 0.344 ± 0.005 |
| KS fracción en fade | 0.087 | 0.369 | 0.169 ± 0.029 | 0.139 ± 0.039 | 0.133 ± 0.009 | 0.157 ± 0.055 | 0.101 ± 0.012 | 0.240 ± 0.010 | 0.194 ± 0.008 |
| KS eficiencia espectral | 0.123 | 0.402 | 0.161 ± 0.038 | 0.124 ± 0.031 | 0.125 ± 0.017 | 0.182 ± 0.046 | 0.105 ± 0.013 | 0.257 ± 0.012 | 0.199 ± 0.008 |
| KS Bc90 por canal | 0.198 | 0.255 | 0.192 ± 0.036 | 0.173 ± 0.061 | 0.150 ± 0.017 | 0.173 ± 0.042 | 0.170 ± 0.023 | 0.245 ± 0.004 | 0.235 ± 0.002 |
| KS ventana de retardo 90 % | 0.128 | 0.419 | 0.164 ± 0.037 | 0.145 ± 0.028 | 0.172 ± 0.017 | 0.262 ± 0.087 | 0.241 ± 0.066 | 0.187 ± 0.009 | 0.146 ± 0.004 |
| KS taps con el 90 % de la energía | 0.062 | 0.124 | 0.075 ± 0.015 | 0.090 ± 0.031 | 0.120 ± 0.026 | 0.147 ± 0.063 | 0.156 ± 0.021 | 0.038 ± 0.003 | 0.042 ± 0.005 |
| KS intervalo de retardo 40 dB | 0.138 | 0.124 | 0.195 ± 0.036 | 0.189 ± 0.054 | 0.253 ± 0.024 | 0.351 ± 0.083 | 0.350 ± 0.073 | 0.124 ± 0.006 | 0.195 ± 0.009 |
| KS intervalo de retardo 30 dB | 0.115 | 0.185 | 0.183 ± 0.022 | 0.190 ± 0.016 | 0.161 ± 0.038 | 0.148 ± 0.026 | 0.129 ± 0.033 | 0.527 ± 0.005 | 0.177 ± 0.001 |
| KS intervalo de retardo 25 dB | 0.178 | 0.201 | 0.187 ± 0.009 | 0.194 ± 0.032 | 0.166 ± 0.008 | 0.171 ± 0.021 | 0.171 ± 0.011 | 0.277 ± 0.015 | 0.206 ± 0.002 |
| KS intervalo de retardo 20 dB | 0.198 | 0.085 | 0.212 ± 0.015 | 0.213 ± 0.026 | 0.296 ± 0.031 | 0.323 ± 0.073 | 0.316 ± 0.043 | 0.206 ± 0.007 | 0.216 ± 0.014 |
| BER QPSK \|Δlog10\| | 0.020 | 0.056 | 0.060 ± 0.018 | 0.055 ± 0.017 | 0.042 ± 0.006 | 0.049 ± 0.026 | 0.039 ± 0.004 | 0.090 ± 0.006 | 0.072 ± 0.007 |
| BER 16-QAM \|Δlog10\| | 0.011 | 0.039 | 0.036 ± 0.011 | 0.030 ± 0.012 | 0.021 ± 0.003 | 0.036 ± 0.022 | 0.018 ± 0.002 | 0.058 ± 0.002 | 0.044 ± 0.004 |

**Tabla B** — IC95 del bootstrap por bloques y P(< Gaussian) emparejada (** > 0.975: mejor que el gaussiano; †† < 0.025: peor; * > 0.95 o † < 0.05 solo en una cola)

| Métrica | Real train | Real val | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian IC95 |
|---|---|---|---|---|---|---|---|---|---|
| PDP MAE (dB) | 0.556 | 0.502 | 0.860 | 0.863 | 0.999** | 0.040† | 0.985** | 0.000†† | [1.065, 2.530] |
| KS \|h\| | 0.252 | 0.351 | 0.043† | 0.081 | 0.418 | 0.000†† | 0.954* | 0.567 | [0.031, 0.127] |
| KS retardo medio | 0.743 | 0.124 | 0.956* | 0.313 | 0.124 | 0.126 | 0.076 | 0.000†† | [0.192, 0.530] |
| KS RMS delay, sin umbral | 0.625 | 0.225 | 0.007†† | 0.010†† | 0.009†† | 0.490 | 0.870 | 0.000†† | [0.118, 0.409] |
| KS RMS delay, 40 dB | 0.560 | 0.068 | 0.075 | 0.170 | 0.812 | 0.189 | 0.317 | 0.057 | [0.122, 0.320] |
| KS RMS delay, 30 dB | 0.936 | 0.509 | 0.932 | 0.913 | 0.888 | 0.814 | 0.786 | 0.019†† | [0.162, 0.482] |
| KS RMS delay, 25 dB | 0.900 | 0.521 | 0.816 | 0.799 | 0.761 | 0.657 | 0.679 | 0.075 | [0.147, 0.475] |
| KS RMS delay, 20 dB | 0.745 | 0.252 | 0.580 | 0.530 | 0.487 | 0.379 | 0.428 | 0.263 | [0.117, 0.382] |
| KS profundidad del nulo | 0.991** | 0.658 | 0.995** | 0.994** | 0.995** | 0.977** | 0.992** | 0.000†† | [0.195, 0.451] |
| KS ganancia de ruido ZF | 0.953* | 0.096 | 0.947 | 0.953* | 0.963* | 0.884 | 0.956* | 0.006†† | [0.141, 0.452] |
| KS rizado | 0.996** | 0.819 | 0.998** | 0.998** | 0.998** | 0.992** | 0.995** | 0.000†† | [0.210, 0.491] |
| KS fracción en fade | 0.756 | 0.004†† | 0.763 | 0.791 | 0.798 | 0.598 | 0.761 | 0.090 | [0.112, 0.354] |
| KS eficiencia espectral | 0.749 | 0.002†† | 0.718 | 0.709 | 0.730 | 0.525 | 0.691 | 0.115 | [0.130, 0.396] |
| KS Bc90 por canal | 0.861 | 0.290 | 0.783 | 0.752 | 0.758 | 0.621 | 0.607 | 0.083 | [0.139, 0.440] |
| KS ventana de retardo 90 % | 0.463 | 0.110 | 0.375 | 0.467 | 0.430 | 0.248 | 0.332 | 0.242 | [0.115, 0.367] |
| KS taps con el 90 % de la energía | 0.394 | 0.166 | 0.197 | 0.229 | 0.284 | 0.193 | 0.231 | 0.548 | [0.036, 0.230] |
| KS intervalo de retardo 40 dB | 0.884 | 0.629 | 0.305 | 0.516 | 0.041† | 0.003†† | 0.002†† | 0.693 | [0.127, 0.292] |
| KS intervalo de retardo 30 dB | 0.752 | 0.430 | 0.390 | 0.311 | 0.523 | 0.659 | 0.671 | 0.000†† | [0.122, 0.361] |
| KS intervalo de retardo 25 dB | 0.907 | 0.586 | 0.780 | 0.850 | 0.771 | 0.749 | 0.710 | 0.126 | [0.150, 0.409] |
| KS intervalo de retardo 20 dB | 0.750 | 0.779 | 0.794 | 0.777 | 0.164 | 0.111 | 0.143 | 0.607 | [0.133, 0.423] |
| BER QPSK \|Δlog10\| | 0.707 | 0.558 | 0.672 | 0.701 | 0.757 | 0.592 | 0.732 | 0.160 | [0.020, 0.266] |
| BER 16-QAM \|Δlog10\| | 0.688 | 0.537 | 0.644 | 0.674 | 0.717 | 0.559 | 0.689 | 0.152 | [0.011, 0.150] |

**Tabla C** — IC95 de cada fuente (bootstrap de la media entre semillas):

| Métrica | Real train | Real val | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|---|
| PDP MAE (dB) | [1.006, 2.526] | [1.090, 2.655] | [1.055, 2.509] | [1.066, 2.516] | [0.997, 2.341] | [1.194, 2.556] | [0.860, 1.828] | [1.170, 2.699] | [1.065, 2.530] |
| KS \|h\| | [0.027, 0.137] | [0.037, 0.162] | [0.033, 0.137] | [0.032, 0.133] | [0.027, 0.130] | [0.041, 0.139] | [0.021, 0.093] | [0.032, 0.125] | [0.031, 0.127] |
| KS retardo medio | [0.171, 0.518] | [0.255, 0.619] | [0.184, 0.519] | [0.194, 0.534] | [0.201, 0.591] | [0.202, 0.590] | [0.223, 0.634] | [0.253, 0.580] | [0.192, 0.530] |
| KS RMS delay, sin umbral | [0.109, 0.398] | [0.127, 0.527] | [0.151, 0.469] | [0.150, 0.475] | [0.140, 0.446] | [0.166, 0.372] | [0.126, 0.304] | [0.276, 0.635] | [0.118, 0.409] |
| KS RMS delay, 40 dB | [0.097, 0.336] | [0.136, 0.521] | [0.134, 0.332] | [0.121, 0.342] | [0.100, 0.293] | [0.185, 0.334] | [0.151, 0.342] | [0.134, 0.500] | [0.122, 0.320] |
| KS RMS delay, 30 dB | [0.131, 0.410] | [0.182, 0.478] | [0.156, 0.421] | [0.156, 0.409] | [0.134, 0.379] | [0.156, 0.337] | [0.146, 0.376] | [0.215, 0.551] | [0.162, 0.482] |
| KS RMS delay, 25 dB | [0.130, 0.380] | [0.186, 0.455] | [0.147, 0.397] | [0.157, 0.376] | [0.123, 0.363] | [0.162, 0.402] | [0.145, 0.398] | [0.149, 0.492] | [0.147, 0.475] |
| KS RMS delay, 20 dB | [0.108, 0.331] | [0.168, 0.451] | [0.129, 0.355] | [0.143, 0.359] | [0.111, 0.391] | [0.158, 0.447] | [0.128, 0.427] | [0.118, 0.395] | [0.117, 0.382] |
| KS profundidad del nulo | [0.095, 0.331] | [0.136, 0.494] | [0.171, 0.406] | [0.160, 0.398] | [0.146, 0.386] | [0.133, 0.296] | [0.122, 0.353] | [0.245, 0.504] | [0.195, 0.451] |
| KS ganancia de ruido ZF | [0.108, 0.320] | [0.200, 0.628] | [0.146, 0.406] | [0.141, 0.382] | [0.127, 0.384] | [0.164, 0.312] | [0.112, 0.350] | [0.188, 0.503] | [0.141, 0.452] |
| KS rizado | [0.091, 0.361] | [0.123, 0.477] | [0.167, 0.440] | [0.158, 0.427] | [0.150, 0.426] | [0.128, 0.346] | [0.117, 0.386] | [0.233, 0.519] | [0.210, 0.491] |
| KS fracción en fade | [0.085, 0.268] | [0.242, 0.606] | [0.122, 0.315] | [0.115, 0.280] | [0.103, 0.291] | [0.138, 0.290] | [0.095, 0.264] | [0.134, 0.400] | [0.112, 0.354] |
| KS eficiencia espectral | [0.113, 0.337] | [0.282, 0.642] | [0.141, 0.343] | [0.135, 0.314] | [0.124, 0.331] | [0.174, 0.336] | [0.122, 0.328] | [0.146, 0.448] | [0.130, 0.396] |
| KS Bc90 por canal | [0.122, 0.389] | [0.181, 0.471] | [0.144, 0.384] | [0.158, 0.363] | [0.127, 0.354] | [0.157, 0.384] | [0.145, 0.406] | [0.143, 0.455] | [0.139, 0.440] |
| KS ventana de retardo 90 % | [0.116, 0.356] | [0.228, 0.662] | [0.141, 0.383] | [0.135, 0.354] | [0.113, 0.391] | [0.160, 0.480] | [0.151, 0.461] | [0.130, 0.409] | [0.115, 0.367] |
| KS taps con el 90 % de la energía | [0.041, 0.244] | [0.069, 0.315] | [0.061, 0.250] | [0.065, 0.260] | [0.052, 0.320] | [0.078, 0.348] | [0.049, 0.357] | [0.036, 0.236] | [0.036, 0.230] |
| KS intervalo de retardo 40 dB | [0.096, 0.255] | [0.119, 0.295] | [0.135, 0.296] | [0.146, 0.288] | [0.153, 0.351] | [0.247, 0.449] | [0.245, 0.447] | [0.094, 0.282] | [0.127, 0.292] |
| KS intervalo de retardo 30 dB | [0.105, 0.330] | [0.129, 0.366] | [0.148, 0.356] | [0.137, 0.371] | [0.123, 0.352] | [0.136, 0.308] | [0.121, 0.295] | [0.399, 0.705] | [0.122, 0.361] |
| KS intervalo de retardo 25 dB | [0.135, 0.377] | [0.137, 0.420] | [0.136, 0.368] | [0.145, 0.374] | [0.131, 0.352] | [0.153, 0.335] | [0.140, 0.358] | [0.170, 0.469] | [0.150, 0.409] |
| KS intervalo de retardo 20 dB | [0.114, 0.399] | [0.065, 0.352] | [0.123, 0.415] | [0.121, 0.418] | [0.141, 0.497] | [0.192, 0.527] | [0.170, 0.516] | [0.151, 0.416] | [0.133, 0.423] |
| BER QPSK \|Δlog10\| | [0.016, 0.205] | [0.017, 0.263] | [0.033, 0.241] | [0.029, 0.238] | [0.022, 0.227] | [0.044, 0.192] | [0.023, 0.217] | [0.018, 0.285] | [0.020, 0.266] |
| BER 16-QAM \|Δlog10\| | [0.009, 0.114] | [0.010, 0.149] | [0.022, 0.133] | [0.020, 0.129] | [0.013, 0.121] | [0.030, 0.111] | [0.012, 0.116] | [0.010, 0.163] | [0.011, 0.150] |


Notas:
- DDPM entre semillas es **bimodal** en PDP MAE (dB): 2.071 (semilla 42, sin sembrar), 1.298, 2.184,
  1.371, 1.298. Las demás arquitecturas son estables (p. ej. DDIM 1.056 / 1.066 / 0.884 / 0.921 / 1.195;
  WGAN 1.516 / 1.377 / 1.346 / 1.478 / 1.330).
- La desviación entre semillas de los modelos generativos (0.01–0.09 en KS) es del orden de muchas de
  las diferencias entre modelos; la de los clásicos es casi nula (solo cambia su muestreo).
- Los IC95 son anchos (p. ej. PDP MAE del gaussiano [1.07, 2.53] dB; KS del nulo [0.20, 0.45]): con un
  solo test, la variabilidad de la referencia es del mismo orden que las diferencias entre modelos.

**Lectura de la significación (P(< Gaussian), bilateral al 5 %)**:

1. **Significativamente mejores que el gaussiano (los 5 generativos)**: profundidad del nulo
   (P = 0.977–0.995) y rizado de |H(f)| (P = 0.992–0.998). También la referencia real train
   (0.991 y 0.996): el gaussiano es claramente demasiado selectivo en frecuencia.
2. **Ganancia de ruido ZF**: todos los generativos mejores en media (0.194–0.251 frente a 0.293) pero solo
   al límite (P = 0.88–0.96; significativo al 5 % unilateral para DCGAN, WGAN y DDIM, no bilateral).
3. **PDP MAE**: WGAN (P = 0.999) y DDIM (P = 0.985) significativamente mejores; DDPM peor al límite
   (P = 0.040); TDL-Rice significativamente peor (P = 0.000).
4. **KS de |h|**: DDIM mejor al límite (P = 0.954); DDPM significativamente peor (P = 0.000).
5. **RMS delay sin umbral**: GAN, DCGAN y WGAN **significativamente peores** que el gaussiano
   (P = 0.007–0.010); DDIM mejor en media (0.138 frente a 0.204) pero no significativo (P = 0.870). Con
   umbrales (40–20 dB) ninguna diferencia generativo–gaussiano es significativa (P entre 0.07 y 0.93).
6. **Estructura de retardos**: DDPM y DDIM significativamente peores en intervalo de retardo a 40 dB
   (P = 0.003 y 0.002); en ventana de retardo y taps con el 90 % de la energía nadie difiere
   significativamente del gaussiano. Retardo medio: solo GAN mejor (0.980 a 40 dB, 0.979 a 30 dB); el
   resto, no significativo.
7. **BER** (QPSK y 16-QAM): todos los generativos tienen menor |Δlog10 BER| medio que el gaussiano
   (QPSK 0.039–0.060 frente a 0.072; 16-QAM 0.018–0.036 frente a 0.044), pero **ninguna diferencia es
   significativa** (P = 0.56–0.76). Ni siquiera la referencia real train lo es (P = 0.71): a esta escala
   de test, la BER media no discrimina entre fuentes.
8. **TDL-Rice** es significativamente peor que el gaussiano en casi todo (PDP, nulos, rizado, ZF, RMS,
   intervalo a 30 dB); solo empata en las métricas de energía concentrada.

**Sensibilidad al tamaño de bloque del bootstrap** (`bootstrap_sensitivity.txt/.csv`; 1000 réplicas por
bloque; "tramo" = un bloque por tramo continuo, §2.3). Número de modelos generativos (de 5)
significativamente mejores / peores que el gaussiano, y anchura del IC95 del gaussiano relativa a la del
bloque 50:

| Métrica | Bloque 25 | 50 | 100 | 500 (no fiable) | **Tramo** |
|---|---|---|---|---|---|
| KS profundidad del nulo | 5/0 ×0.79 | 5/0 ×1.00 | 4/0 ×1.15 | 5/0 ×0.19 | **4/0 ×1.47** |
| KS rizado | 5/0 ×0.79 | 5/0 ×1.00 | 5/0 ×1.15 | 5/0 ×0.17 | **4/0 ×1.46** |
| KS ganancia ZF | 4/0 ×0.77 | 0/0 ×1.00 | 0/0 ×1.15 | 5/0 ×0.19 | 0/0 ×1.34 |
| PDP MAE | 2/1 ×0.73 | 2/0 ×1.00 | 0/0 ×1.42 | 2/0 ×0.13 | 0/0 ×1.43 |
| BER QPSK | 0/0 ×0.87 | 0/0 ×1.00 | 0/0 ×1.08 | 5/0 ×0.06 | 0/0 ×1.38 |
| BER 16-QAM | 0/0 ×0.84 | 0/0 ×1.00 | 0/0 ×1.16 | 3/0 ×0.20 | 0/0 ×1.46 |
| KS \|h\| | 1/2 ×0.75 | 0/1 ×1.00 | 0/1 ×1.31 | 1/3 ×0.15 | 0/1 ×1.33 |
| KS RMS delay, sin umbral | 0/3 ×0.79 | 0/3 ×1.00 | 0/0 ×1.24 | 1/3 ×0.64 | 0/0 ×1.49 |
| KS RMS delay, 20 dB | 0/0 ×0.83 | 0/0 ×1.00 | 0/0 ×1.38 | 0/2 ×0.53 | 0/0 ×1.83 |
| KS fracción en fade | 0/0 ×0.85 | 0/0 ×1.00 | 0/0 ×1.16 | 4/0 ×0.44 | 0/0 ×1.44 |
| KS eficiencia espectral | 0/0 ×0.82 | 0/0 ×1.00 | 0/0 ×1.21 | 5/0 ×0.39 | 0/0 ×1.61 |

P(< Gaussian) por modelo en las métricas clave (bloque 50 → tramo):

| Métrica | GAN | DCGAN | WGAN | DDPM | DDIM | Real train |
|---|---|---|---|---|---|---|
| Nulo | 0.995 → 0.980 | 0.994 → 0.977 | 0.995 → 0.982 | 0.977 → 0.927 | 0.992 → 0.979 | 0.991 → 0.974 |
| Rizado | 0.998 → 0.989 | 0.998 → 0.987 | 0.998 → 0.992 | 0.992 → 0.968 | 0.995 → 0.983 | 0.996 → 0.979 |
| Ganancia ZF | 0.947 → 0.863 | 0.953 → 0.873 | 0.963 → 0.905 | 0.884 → 0.756 | 0.956 → 0.884 | 0.953 → 0.870 |
| PDP MAE | 0.860 → 0.818 | 0.863 → 0.806 | 0.999 → 0.972 | 0.040 → 0.147 | 0.985 → 0.934 | 0.556 → 0.495 |
| BER QPSK | 0.672 → 0.633 | 0.701 → 0.647 | 0.757 → 0.685 | 0.592 → 0.560 | 0.732 → 0.666 | 0.707 → 0.655 |
| RMS delay sin umbral | 0.007 → 0.048 | 0.010 → 0.048 | 0.009 → 0.050 | 0.490 → 0.505 | 0.870 → 0.762 | 0.625 → 0.526 |

Lectura:
- **Bloque 500 no es fiable** y no se usa: test (900 canales) solo tiene 2 bloques de 500, así que el
  bootstrap apenas varía la referencia y infravalora mucho la varianza (IC95 ×0.06–0.64 más estrechos;
  casi todo sale significativo).
- Al pasar de bloques de 50 a tramos (8 en test), los IC95 se ensanchan ×1.3–1.8, coherente con que la
  coherencia real dentro de un tramo es mayor que 50 CIR (§2.2).
- **Robusto**: nulo y rizado siguen siendo significativos con bloque = tramo para GAN, DCGAN, WGAN y DDIM
  (P = 0.977–0.992); DDPM pasa a no significativo (0.927 y 0.968). TDL-Rice sigue siendo peor.
- **No robusto**:
  - PDP de WGAN (0.972) y DDIM (0.934) queda al límite, ya no significativo al 5 % bilateral (WGAN sí
    unilateral).
  - RMS delay sin umbral: el "peor" de GAN/DCGAN/WGAN pasa a P ≈ 0.05.
  - Ganancia ZF: solo significativa con bloque 25.
- **BER**: no significativa con ningún bloque fiable.
- **Cifra recomendada para el artículo**: bloque = tramo (es la unidad independiente, §2.3), con bloque 50
  como referencia. Con 8 tramos en test el bootstrap es tosco; la validación cruzada por tramos daría más
  potencia (§8).

---

### 6.11 Validación cruzada por tramos (prueba piloto, `eval_results/sodupe_180_segcv/`)

Diseño: `make_segment_folds.py` reparte los 109 tramos (§2.3) en 5 particiones equilibradas (test ≈ 2840
canales / 21–22 tramos; val ≈ 1400–1570 canales / 8–14 tramos enteros; train el resto, ≈ 9800–9930). Ningún
tramo se reparte entre conjuntos. En cada partición se entrenan los 5 modelos (**una sola semilla, 42**), se
ajustan TDL-Rice y gaussiano con su train y se evalúa frente a su test (`evaluation.py`). `cv_stats.py`
promedia las métricas de seed_stats sobre las 5 particiones y hace un bootstrap estratificado por tramos
(1000 réplicas; test, train y val remuestreados por tramos dentro de cada partición; generados canal a
canal). Ficheros: `folds.txt`, `cv_folds.txt/.csv` (por partición), `cv_bootstrap.txt/.csv`.

Media sobre particiones / P(< Gaussian) (** > 0.975, †† < 0.025):

| Métrica | Real train | GAN | DCGAN | WGAN | DDPM | DDIM | TDL-Rice | Gaussian |
|---|---|---|---|---|---|---|---|---|
| KS profundidad del nulo | 0.097 / 0.82 | 0.118 / 0.82 | 0.157 / 0.00†† | 0.099 / 0.89 | 0.108 / 0.74 | 0.103 / 0.82 | 0.198 / 0.00†† | 0.141 |
| KS rizado | 0.087 / 0.83 | 0.109 / 0.88 | 0.143 / 0.03 | 0.095 / 0.88 | 0.103 / 0.78 | 0.097 / 0.84 | 0.181 / 0.00†† | 0.137 |
| KS ganancia ZF | 0.096 / 0.82 | 0.119 / 0.80 | 0.173 / 0.00†† | 0.107 / 0.84 | 0.118 / 0.68 | 0.107 / 0.81 | 0.204 / 0.00†† | 0.142 |
| KS fracción en fade | 0.119 / 0.64 | 0.126 / 0.65 | 0.172 / 0.00†† | 0.113 / 0.85 | 0.123 / 0.68 | 0.129 / 0.64 | 0.196 / 0.00†† | 0.138 |
| PDP MAE (dB) | 0.598 / 0.20 | 0.757 / 0.01†† | 0.780 / 0.00†† | 0.764 / 0.11 | 0.923 / 0.04 | 0.992 / 0.10 | 0.640 / 0.15 | 0.595 |
| KS RMS delay, sin umbral | 0.101 / 0.15 | 0.159 / 0.01†† | 0.156 / 0.01†† | 0.138 / 0.01†† | 0.187 / 0.02†† | 0.142 / 0.17 | 0.259 / 0.00†† | 0.098 |
| KS retardo medio | 0.129 / 0.22 | 0.141 / 0.45 | 0.178 / 0.11 | 0.256 / 0.00†† | 0.267 / 0.00†† | 0.227 / 0.01†† | 0.168 / 0.16 | 0.131 |
| KS intervalo de retardo 40 dB | 0.102 / 0.63 | 0.146 / 0.16 | 0.157 / 0.08 | 0.199 / 0.00†† | 0.351 / 0.00†† | 0.297 / 0.00†† | 0.137 / 0.42 | 0.134 |
| KS \|h\| | 0.030 / 0.24 | 0.036 / 0.19 | 0.036 / 0.05 | 0.033 / 0.22 | 0.038 / 0.24 | 0.046 / 0.17 | 0.031 / 0.27 | 0.030 |
| BER QPSK \|Δlog10\| | 0.073 / 0.69 | 0.080 / 0.63 | 0.106 / 0.00†† | 0.075 / 0.73 | 0.086 / 0.61 | 0.082 / 0.62 | 0.107 / 0.01†† | 0.089 |
| BER 16-QAM \|Δlog10\| | 0.040 / 0.73 | 0.045 / 0.64 | 0.064 / 0.00†† | 0.041 / 0.81 | 0.051 / 0.60 | 0.049 / 0.60 | 0.071 / 0.00†† | 0.051 |

Lectura:
- **Ningún modelo generativo es significativamente mejor que el gaussiano en ninguna métrica.** En nulo,
  rizado, ZF, fade y BER, WGAN, DDIM, GAN y DDPM son mejores en media (WGAN ≈ real train) pero con P =
  0.64–0.89. Por partición, WGAN mejora al gaussiano en nulo y rizado en 4 de 5 particiones (no en la 1).
- **Techo de la evaluación**: tampoco real train (datos reales independientes) es significativamente mejor
  que el gaussiano (P = 0.82 en nulo y ZF, 0.20 en PDP). Con la heterogeneidad entre tramos (eta² 0.5–0.8),
  estas métricas no separan "datos reales" de "gaussiano de covarianza completa"; ningún modelo puede
  superar ese techo. Más semillas no lo cambian.
- **PDP**: el gaussiano (0.595 dB) iguala a real train (0.598), como se esperaba (reproduce la covarianza
  de train por construcción); GAN y DCGAN son significativamente peores. La ventaja de DDIM/WGAN en PDP de
  §6.10 no se mantiene: era propia del test único.
- **Estructura de retardos**: los generativos son **significativamente peores** que el gaussiano en RMS
  delay sin umbral (GAN, DCGAN, WGAN, DDPM), retardo medio e intervalo de retardo a 40 dB (WGAN, DDPM,
  DDIM).
- DCGAN falla en la partición 4 (KS del nulo 0.37, ZF 0.49): posible colapso; con una semilla no se puede
  saber si es la partición o el entrenamiento.
- Conclusión frente a §6.10: la significación de nulo y rizado con el test único **no se confirma** con
  validación cruzada. Las cifras del artículo deben ser las de esta sección (o de su ampliación).

## 7. Figuras (PDF y PNG) en `eval_results/sodupe_180_split_g200/`

| Figura | Contenido |
|---|---|
| `fig_pdp_linear`, `fig_pdp_db` | PDP medio real vs modelos (lineal y dB) |
| `fig_kde_comparison`, `fig_cdf_comparison`, `fig_cdf_comparison_db` | Distribución de \|h\| (KDE, CDF con zoom, CDF en dB) |
| `fig_freq_response_db` | E\|H(f)\|² referida al máximo real |
| `fig_frequency_correlation` | \|R_H(Δf)\| normalizada, con líneas en 0.9 y 0.5 |
| `fig_rms_delay`, `fig_mean_delay` | KDE + diagrama de cajas, sin umbral |
| `fig_rms_delay_th{40,30,25,20}dB`, `fig_mean_delay_th{40,30,25,20}dB` | Ídem con cada umbral |
| `fig_rms_delay_cdf_thresholds` | CDF del RMS delay con todos los umbrales lado a lado (candidata para el artículo) |
| `fig_null_depth` | CDF de la profundidad del nulo y de la ganancia de ruido ZF |
| `fig_kurtosis` | Curtosis de la parte difusa por tap (taps a < 30 dB del pico, sin el tap del pico) |
| `fig_channel_params` | CDF de los 10 parámetros por canal (rizado, fade, eficiencia espectral, Bc por canal, ventana de retardo, taps 90 %, intervalo de retardo a 40/30/25/20 dB) |
| `fig_ber_qpsk`, `fig_ber_16qam` | Curvas de BER con teóricas AWGN y Rayleigh |
| `fig_memorization_nn` | CDF de la distancia al vecino más próximo de train (espacios norm y rel) |
| `examples_<modelo>` | 4 canales reales y 4 generados (\|h\| en dB) por modelo, incluidos `tdl` y `gauss` |

Figuras del preprocesado (las genera `preprocess_dataset.py` sin `--no-plots`), en `data/datos_loopback/`:
`verificacion_preprocesado.png` (verificación 2×2), `mean_profile.png`, `heatmap_magnitud.png`,
`histogram_magnitud.png`, `histogram_real_imag.png`, `examples_real_imag.png`. **[NO VERIFICADO]** que
estén actualizadas con la última ejecución (la última se lanzó con `--no-plots`).

Rondas anteriores (para una posible figura de ablación de la normalización): `eval_results/sodupe_180/`
(normalización global) y `eval_results/sodupe_180_tapnorm/` (por tap, dentro de muestra), con las mismas
figuras base.

---

## 8. Limitaciones y amenazas a la validez

1. **Un solo test.** Test tiene 900 canales repartidos en solo **8 tramos continuos** (3 completos;
   §2.3). La propia referencia real varía mucho según qué datos reales se comparen
   con test (p. ej. KS de la ganancia ZF: train 0.139, val 0.385; fracción en fade: 0.088 vs 0.369).
   Se cuantificó con 5 semillas y bootstrap por bloques (§4.2, §6.10): los IC95 son anchos y solo
   algunas diferencias son significativas (nulos, rizado, PDP de WGAN/DDIM; BER no). No se hizo
   validación cruzada rotando los bloques de test (daría más potencia estadística, pero exige
   reentrenar 5 modelos × k particiones).
   - Sensibilidad al bloque del bootstrap (§6.10): con bloque = tramo siguen siendo significativos nulo y
     rizado (4 de 5 generativos; DDPM no); el PDP de WGAN/DDIM queda al límite (P = 0.972 / 0.934) y el
     RMS sin umbral deja de ser significativo. Bloque 500 no es fiable (2 bloques en test). El número de
     réplicas (1000) es elección propia.
   - La separación se hizo por bloques de 500 sin conocer los tramos: 2 tramos se reparten entre train y
     test y, dentro de un tramo, ρ ≈ 0.3 a 200 CIR (§2.2), así que el hueco de 200 no descorrela del todo
     en esos 2 tramos.
   - 5 semillas dan una estimación tosca de la desviación entre semillas (y DDPM es bimodal).
2. **Deriva del canal (no estacionariedad).** Test es un tramo de la medida con estadística algo distinta
   (RMS delay sin umbral: test 547.1 ns, train 606.7, val 614.7; `recordings.py`). La estadística cambia
   mucho entre tramos continuos (eta² = 0.50–0.81 en los retardos, §2.3). Por eso hay que leer los resultados frente a la referencia
   "real contra real" y no frente a cero.
3. **Una sola medida/escenario** (sodupe_180). No se ha comprobado la generalización a otros escenarios
   (el autor tiene más medidas en OneDrive).
4. **Hiperparámetros no ajustados por igual** (§3.8). El DDPM, en particular, podría mejorar; el DDIM
   usa pérdidas adicionales (ponderación por tap y PSD) que el DDPM no usa, así que la comparación DDPM
   vs DDIM mezcla muestreador y función de pérdida.
5. **El barrido de muestreo del DDIM** se hizo sobre el modelo de la ronda 2 y contra todo el dataset
   (§5.3), no sobre el modelo final contra val.
6. **Memorización parcial del DDPM** (y leve del DDIM) en el espacio original: parte de su buena BER
   podría deberse a reproducir de cerca los taps fuertes de canales concretos de entrenamiento.
7. **Subestimación sistemática del retardo medio** por todos los modelos (12–37 ns). Causa no
   investigada **[NO VERIFICADO]**.
8. **Bc50 no es fiable**: la correlación en frecuencia real se aplana cerca de 0.45–0.5, así que el
   cruce por 0.5 es muy sensible.
9. **BER idealizada**: CSI perfecta, ZF, sin ISI (prefijo cíclico supuesto), 1 símbolo OFDM por canal y
   energía unidad por canal. Extensiones previstas para el artículo (no hechas): MMSE, 64-QAM,
   estimación de canal imperfecta (LS con pilotos), distribución de la BER por canal / *outage*.
10. **Restricción de energía constante**: el preprocesado normaliza todos los canales a la misma energía,
    así que no se modela la variación de la potencia del acoplamiento (pérdida de aislamiento) entre
    medidas. Los modelos clásicos también se reescalan a esa energía.
11. **Los modelos son i.i.d.**: no generan secuencias temporales (sin Doppler ni correlación temporal).
    El fichero no es continuo (109 tramos de ~1 s, §2.3): cualquier análisis temporal o Doppler debe
    hacerse dentro de cada tramo. El origen de los cortes **[NO VERIFICADO]** (no son las grabaciones de
    5 s de la campaña).
15. **Referencia de memorización**: la distancia train–train al vecino más próximo (§6.7) incluye vecinos
    del mismo tramo, muy correlados; una referencia que excluya el mismo tramo sería más justa (pendiente).
12. **Componente del tap 0 con valores atípicos** (curtosis ~94; valores aislados a ~25σ) que ningún
    modelo reproduce; su origen físico (¿errores de estimación?, ¿eventos reales?) **[NO VERIFICADO]**.
13. **Resolución en retardo**: 109.375 ns por tap; la ventana y el intervalo de retardo están
    cuantizados a esa resolución.
14. **Determinismo**: no se ha comprobado que relanzar el entrenamiento reproduzca exactamente los mismos
    checkpoints. En la semilla 42, DDPM y DDIM no fijaban semilla (corregido para las semillas 1–4 con
    `train.py --seed`).

---

## 9. Reproducción (comandos)

```bash
python preprocess_dataset.py                          # dataset + separación (bloques 500, huecos 200)
python train_models/train.py --model gan dcgan wgan ddpm ddim
python evaluation_models/evaluation.py                # genera 3000 canales/modelo y evalúa sobre test
python evaluation_models/memorization.py
# Semillas adicionales (por semilla N y modelo): entrenar, generar y evaluar en su propia carpeta
python train_models/train.py --model gan dcgan wgan ddpm ddim --seed N   # → checkpoints/.../seed{N}/
python evaluation_models/evaluation.py --config eval_results/sodupe_180_split_g200/seeds/seedN/evaluation_config.json
python evaluation_models/seed_stats.py --cache <carpeta>   # tabla entre semillas + bootstrap (≈45 min, 4 procesos)
python evaluation_models/recordings.py                     # tramos continuos, coherencia por tramo, variación entre tramos
for b in 25 100 500 segment; do python evaluation_models/seed_stats.py --block $b --cache <carpeta>; done
python evaluation_models/bootstrap_sensitivity.py           # comparación de IC95 y P entre bloques
python evaluation_models/make_segment_folds.py              # 5 particiones por tramos
# por partición k: train.py --model <m> --config checkpoints/sodupe_180_segcv/fold<k>/<gans|dm>_config.json,
# generate_from_config.py y evaluation.py --config eval_results/sodupe_180_segcv/fold<k>/evaluation_config.json --reuse-generated
python evaluation_models/cv_stats.py --cache <carpeta>      # media sobre particiones + bootstrap por tramos
python evaluation_models/model_costs.py               # lee training_log.json de paths.save_dir
python evaluation_models/sweep_sampling.py --steps 25 50 100 250 500 1000 --eta 0 0.5 1   # muestreo DDIM
```

La referencia "real contra real" completa (`baseline_real_vs_test_full.txt`) se generó con un script
puntual que llama a `evaluation.channel_stats` y `evaluation.compare` con 3000 canales de train (semilla
42) y los 1100 de val como "generados"; **no está versionado como script** (habría que añadirlo si se
quiere regenerar automáticamente).

Detalles de implementación útiles:
- `evaluation.py --reuse-generated` rehace métricas y figuras con los canales de `generated/` (los
  clásicos se reajustan siempre; son deterministas con la semilla).
- `--subset {train,val,test,all}` cambia la referencia real.
- Las claves de la configuración que controlan las métricas: `evaluation.delay_thresholds_db`,
  `evaluation.channel_params` (umbral de fade, SNR de la eficiencia espectral, % de la ventana de retardo,
  umbral de Bc por canal), `evaluation.coherence_thresholds`, `ber.*`.
- `metrics.pkl` contiene todas las métricas y arrays (PDP, distribuciones por canal) para rehacer
  figuras sin recalcular.
