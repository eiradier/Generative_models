# PAPER_NOTES — Modelos generativos de canales loopback IBFD medidos

Notas técnicas para redactar el artículo (destino: revista). Todo lo que aparece aquí está sacado del
código, de las configuraciones o de los ficheros de resultados del repositorio, salvo lo marcado como
**[NO VERIFICADO]** (dato que no consta en el repositorio o que se dedujo sin comprobarlo) o
**[AD HOC]** (calculado con un script puntual durante el desarrollo que no está en el repositorio;
las cifras son las que se obtuvieron, pero habría que regenerarlas con un script versionado antes de
publicarlas).

Rutas relativas a la raíz del repositorio. La ronda de resultados **final** es
`eval_results/sodupe_180_split_g200/` (checkpoints en `checkpoints/sodupe_180_split_g200/`).

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
| Intervalo temporal entre CIR consecutivas | **[NO VERIFICADO]** no consta en el repositorio. Necesario para expresar tiempos de coherencia en segundos y para el análisis Doppler (que el autor hará aparte; los modelos no lo modelan) |
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

**Justificación del hueco de 200 — tiempo de coherencia en muestras** [AD HOC]. Correlación entre
la parte variable (sin la media) del canal en la medida *n* y en la *n+k*, autocorrelación por tap
promediada con la potencia de cada tap (`.npy` original, 14218 canales, 200 taps):

| Separación k (medidas) | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Correlación por tap (parte variable) | 0.984 | 0.972 | 0.960 | 0.941 | 0.889 | 0.783 | 0.676 | 0.479 | 0.307 | 0.214 | 0.079 | 0.024 | 0.017 | 0.066 | 0.009 |

→ coherencia 0.9 ≈ 10 medidas, 0.5 ≈ 50 medidas, descorrelación ≈ 150–200 medidas. Con hueco de 100
la correlación residual entre el último canal de train y el primero de test era ~0.2; con 200, ~0.02.
Con la correlación vectorial completa (incluida la parte fija) nunca baja de ~0.6, por la componente
estática.

**Por qué bloques de 500 y no más pequeños** [AD HOC]. Simulación con 20 semillas de reparto, hueco 200:

| Bloque | Canales perdidos | Desviación media \|RMS delay medio de test − total\| |
|---|---|---|
| 300 | 25 % | 39 ns |
| 500 | 13 % | 32 ns |
| 1000 | 10 % | 32 ns |

El RMS delay medio por bloque de 500 (sin umbral) varía entre 509 y 746 ns a lo largo de la medida:
el canal deriva lentamente. Con ~70 tramos independientes en toda la medida, test (10 %) contiene
~7 y siempre se desvía ~30 ns de la media, sea cual sea el bloque. Intercalar más fino solo pierde
datos.

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
- Semilla: los scripts de GAN, DCGAN y WGAN hacen `torch.manual_seed(42)` (`experiment.seed`); los de
  **DDPM y DDIM no fijan ninguna semilla** (comprobado: no aparece `seed` en esos ficheros). No se ha
  comprobado el determinismo de ningún entrenamiento. **Un único entrenamiento por modelo** (sin
  repeticiones con otras semillas).
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

### 6.9 Resumen de conclusiones

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

---

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

1. **Un solo entrenamiento por modelo y un solo test.** Sin repeticiones con otras semillas ni
   intervalos de confianza. Test tiene 900 canales ≈ 5–7 tramos independientes del canal (tiempo de
   coherencia ~200 medidas). La propia referencia real varía mucho según qué datos reales se comparen
   con test (p. ej. KS de la ganancia ZF: train 0.139, val 0.385; fracción en fade: 0.088 vs 0.369).
   **Diferencias de KS de ~0.05 entre modelos probablemente no son significativas.** Para una revista
   convendría: varias semillas por modelo y/o validación cruzada rotando los bloques de test, o
   *bootstrap* por bloques para los intervalos de confianza.
2. **Deriva del canal (no estacionariedad).** Test es un tramo de la medida con estadística algo distinta
   (RMS delay sin umbral 547 ns en test frente a ~609 ns en train y val [AD HOC: train 608.6, val
   609.3 con la separación de huecos 100]). Por eso hay que leer los resultados frente a la referencia
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
12. **Componente del tap 0 con valores atípicos** (curtosis ~94; valores aislados a ~25σ) que ningún
    modelo reproduce; su origen físico (¿errores de estimación?, ¿eventos reales?) **[NO VERIFICADO]**.
13. **Resolución en retardo**: 109.375 ns por tap; la ventana y el intervalo de retardo están
    cuantizados a esa resolución.
14. **Determinismo**: no se ha comprobado que relanzar el entrenamiento reproduzca exactamente los mismos
    checkpoints; DDPM y DDIM ni siquiera fijan semilla.

---

## 9. Reproducción (comandos)

```bash
python preprocess_dataset.py                          # dataset + separación (bloques 500, huecos 200)
python train_models/train.py --model gan dcgan wgan ddpm ddim
python evaluation_models/evaluation.py                # genera 3000 canales/modelo y evalúa sobre test
python evaluation_models/memorization.py
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
