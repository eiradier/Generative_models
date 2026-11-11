# Generative models for loopback channels

Useful data in https://upvehueus-my.sharepoint.com/:f:/g/personal/eneko_iradier_ehu_eus/IgAV40P1cFX1QZthYQBXohh8AUePFnWHq3kJoqNYDHAIs3U?e=kUYsvc

Modelos generativos (GAN, DCGAN, WGAN-GP, DDPM y DDIM) de canales loopback medidos. Todos los
comandos se lanzan desde la raíz del repositorio.

## 1. Preprocesado

Copia la medida (p. ej. `ch_sodupe_180_time.npy`) a `data/medidas/` y ejecuta:

```bash
python preprocess_dataset.py
```

- Recorta cada canal a L = 200 taps, descarta los canales con energía anómalamente baja
  (`--min-energy-db`, por defecto 8 dB bajo la mediana) y normaliza.
- Genera `data/datos_loopback/ch_sodupe_180_time_L200.npy` (N, 2, L) y la figura de verificación.
- Genera `..._split.npz` con la separación entrenamiento / validación / test (80/10/10). Los canales
  consecutivos están muy correlados, así que se reparten **bloques** de 500 canales y se descartan
  200 canales en cada frontera entre conjuntos (`--split-block`, `--split-gap`): el canal tarda
  ~150-200 medidas en descorrelarse, así que no hay fuga entre conjuntos.

## 2. Entrenamiento

```bash
python train_models/train.py --model ddim
python train_models/train.py --model gan dcgan wgan ddpm ddim
```

- Configuración: `train_models/models_configs/gans_config.json` y `dm_config.json`.
- `dataset.normalization: "per_tap"` normaliza cada tap (real/imag) por su media y desviación típica
  (el PDP abarca >50 dB; sin esto los modelos de difusión dejan un suelo de ruido en la cola).
  El `scaler.npz` se guarda junto al checkpoint y la evaluación lo deshace.
- `dataset.split_file`: si se indica, se entrena solo con `train` y el early stopping usa `val`
  (pérdida de validación en difusión; métrica de calidad de canales en las GAN).
- Cada modelo guarda en `<paths.save_dir>/<MODELO>/`. DDPM, DDIM y WGAN guardan `last.pth` en cada
  época: si el entrenamiento se interrumpe, al relanzarlo continúa desde ahí.

## 3. Búsqueda de hiperparámetros

```bash
python train_models/hparam_search.py train_models/hparam_specs/ddim.json --dry-run   # ver pruebas
python train_models/hparam_search.py train_models/hparam_specs/ddim.json
```

- Las especificaciones están en `train_models/hparam_specs/` (una por modelo): parámetros a barrer
  (rejilla o aleatorio), épocas máximas y cuántos canales generar.
- Cada prueba entrena, genera canales y los compara con **validación**. Resultados en
  `hpsearch/<modelo>/results.csv` y `summary.txt`, ordenados por `selection_score`
  (= error del PDP en dB + 10 · (KS del RMS delay spread + KS de |h|); menor es mejor).
- Es reanudable: relanzar el mismo comando salta las pruebas terminadas.

Parámetros de muestreo del DDIM (sin reentrenar):

```bash
python evaluation_models/sweep_sampling.py --steps 25 50 100 250 --eta 0 0.5 1
```

## 4. Evaluación

```bash
python evaluation_models/evaluation.py                     # todos los modelos de la configuración
python evaluation_models/evaluation.py --models ddim gan --subset test
python evaluation_models/evaluation.py --reuse-generated   # rehace figuras con los canales ya generados
```

- Configuración: `evaluation_models/evaluation_config.json` (checkpoints, referencia real
  `data.subset`, parámetros de muestreo, SNR de la BER).
- Para resultados finales usa `--subset test` (los hiperparámetros se eligen con `val`).
- El retardo medio y el RMS delay spread se calculan sin umbral y con los umbrales de
  `evaluation.delay_thresholds_db` (dB bajo el pico de cada canal; por defecto 40, 30, 25 y 20 dB).
  Tabla: filas `@X dB`; figuras `fig_rms_delay_th<X>dB`, `fig_mean_delay_th<X>dB` y
  `fig_rms_delay_cdf_thresholds` (CDF con todos los umbrales).
- Métricas de orden superior (no dependen solo de la covarianza): profundidad del nulo y ganancia de
  ruido ZF (`fig_null_depth`), curtosis de la parte difusa por tap (`fig_kurtosis`) y parámetros por
  canal con su CDF (`fig_channel_params`): rizado de |H(f)|, fracción de banda en fade, eficiencia
  espectral (media y outage 10%), Bc por canal, ventana e intervalo de retardo (ITU-R P.1407) y taps
  con el 90% de la energía. Umbrales en `evaluation.channel_params`.
- Además de los modelos generativos se evalúan dos modelos estadísticos clásicos de referencia
  (`evaluation_models/baselines.py`), ajustados con `train`: **TDL-Rice** (taps independientes,
  Rice con la media y la potencia de cada tap; WSSUS) y **Gaussian** (vector gaussiano con la
  covarianza completa entre taps). Se activan con `tdl` y `gauss` en `evaluation.models`.
- Salidas en `paths.save_dir`: canales generados (`generated/`), `metrics_summary.txt`,
  `ber_table.txt`, `metrics.pkl` y figuras (PDF y PNG).

Análisis complementarios (después de `evaluation.py`, usan los canales de `generated/`):

```bash
python evaluation_models/memorization.py   # ¿copian los modelos el entrenamiento? (vecino más próximo)
python evaluation_models/model_costs.py    # parámetros, tiempo de generación y de entrenamiento
```

- `memorization.py`: distancia de cada canal generado al canal de train más cercano, comparada con
  la de los canales reales de test (`memorization.txt`, `fig_memorization_nn`).
- `model_costs.py`: lee los tiempos de entrenamiento de `training_log.json` en `paths.save_dir`
  (`model_costs.txt`).
