"""
Estructura temporal de la medida: discontinuidades entre CIR consecutivas, coherencia temporal dentro de
cada tramo continuo y variación del canal entre tramos.

1. Discontinuidades. Para cada posible corte entre las CIR i-1 e i (CIR normalizadas a energía unidad,
   primeros L taps) se compara el cambio "a través" del corte con el cambio "a cada lado":
       cross  = mediana de 1 - |<x_a, x_b>| con a en [i-w, i), b en [i, i+w)
       within = mediana de 1 - |<x_j, x_j+1>| con j, j+1 en el mismo lado
       r = cross / within
   Dentro de un tramo continuo r ≈ 3 (los pares cruzados están más separados en el tiempo); en una
   discontinuidad r ~ 10^3 (la CIR cambia en 1 medida lo que normalmente cambia en cientos). La
   distribución de log10(r) es bimodal; se toma como corte cada máximo local con r > --threshold
   (por defecto 100, en el hueco entre los dos modos) y se agrupan los candidatos a <= w medidas.
   Los tramos de menos de --min-len CIR (transitorios al inicio de un tramo nuevo) se unen al siguiente.
   Se informa también del número de cortes con otros umbrales (robustez) y de si las longitudes de
   los tramos siguen un periodo fijo (prueba de Rayleigh frente a cortes al azar).

2. Coherencia temporal dentro de cada tramo. rho(k) = sum Re(v_n · v_{n+k}^*) / sqrt(sum |v_n|^2 ·
   sum |v_{n+k}|^2), con v = x - media global (parte variable), sumando sobre los taps y sobre los pares
   (n, n+k) de canales conservados (filtro de energía del preprocesado) del MISMO tramo. Se da también
   la versión sin restricción (todos los pares, como en el análisis original) para comparar.
   k en medidas y en segundos (--cir-period, 7.616 ms según la campaña de medida; no consta en el
   fichero).

3. Variación entre tramos: retardo medio y RMS delay spread medios por tramo (sin umbral y con los
   umbrales de evaluation_config.json) y fracción de la varianza por canal explicada por el tramo (eta^2).

Salidas en <paths.save_dir>: recordings.txt, recordings_segments.csv (un tramo por fila),
recordings_coherence.csv, recordings.npz (índice de tramo de cada canal del .npy original y del dataset
filtrado, índice original de cada canal filtrado, cortes y r), fig_recordings.{pdf,png}.

Uso (desde la raíz del repositorio):
    python evaluation_models/recordings.py
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))
import evaluation as ev  # noqa: E402


def load_raw(path, L):
    """CIR originales (N, L) normalizadas a energía unidad, y la máscara del filtro de energía del
    preprocesado (preprocess_dataset.load_loopback_data: energía >= mediana - 8 dB)."""
    H = np.nan_to_num(np.load(path)).T[:, :L]
    e = np.sum(np.abs(H) ** 2, axis=1)
    edb = 10 * np.log10(e + 1e-30)
    keep = edb - np.median(edb) >= -8.0
    return H / np.sqrt(e[:, None] + 1e-30), keep


def jump_ratio(x, w):
    n = len(x)
    r = np.full(n, np.nan)
    d1 = 1 - np.abs(np.sum(x[1:] * np.conj(x[:-1]), axis=1))  # cambio entre medidas consecutivas
    for i in range(w, n - w + 1):
        A, B = x[i - w:i], x[i:i + w]
        cross = np.median(1 - np.abs(A @ B.conj().T))
        within = np.median(np.r_[d1[i - w:i - 1], d1[i:i + w - 1]])
        r[i] = cross / max(within, 1e-12)
    return r


def find_cuts(r, threshold, w):
    cand = np.where(np.nan_to_num(r) > threshold)[0]
    groups = []
    for i in cand:
        if groups and i - groups[-1][-1] <= w:
            groups[-1].append(i)
        else:
            groups.append([i])
    return np.array([g[int(np.argmax(r[g]))] for g in groups], dtype=int)


def merge_short(cuts, n, min_len):
    """Une los tramos de menos de min_len CIR al tramo siguiente (quita el corte que los cierra)."""
    cuts = list(cuts)
    changed = True
    while changed:
        changed = False
        b = [0] + cuts + [n]
        for j in range(len(b) - 1):
            if b[j + 1] - b[j] < min_len and j < len(cuts):
                del cuts[j]  # el tramo [b[j], b[j+1]) pasa a formar parte del siguiente
                changed = True
                break
    return np.array(cuts, dtype=int)


def periodicity(cuts, n, periods, n_null=100, seed=0):
    """Máximo del estadístico de Rayleigh |mean exp(2πi c/P)| y su percentil 95 con cortes al azar."""
    def zmax(c):
        z = np.abs(np.exp(2j * np.pi * c[:, None] / periods[None]).mean(0))
        return z.max(), periods[z.argmax()]
    z, P = zmax(cuts)
    rng = np.random.default_rng(seed)
    null = [zmax(np.sort(rng.choice(n, len(cuts), replace=False)))[0] for _ in range(n_null)]
    return z, P, float(np.percentile(null, 95))


def coherence(v, pos, seg, kmax, same_segment=True):
    """rho(k) de la parte variable para k = 1..kmax (pares de canales conservados; pos = índice original)."""
    n = len(v)
    out = np.full(kmax + 1, np.nan)
    npairs = np.zeros(kmax + 1, dtype=int)
    out[0], npairs[0] = 1.0, n
    p = np.sum(np.abs(v) ** 2, axis=1)
    lookup = -np.ones(pos.max() + 1, dtype=int)
    lookup[pos] = np.arange(n)
    for k in range(1, kmax + 1):
        tgt = pos + k
        ok = tgt <= pos.max()
        a = np.where(ok)[0]
        b = lookup[tgt[ok]]
        good = b >= 0
        a, b = a[good], b[good]
        if same_segment:
            s = seg[a] == seg[b]
            a, b = a[s], b[s]
        npairs[k] = len(a)
        if len(a) == 0:
            continue
        num = np.sum(np.real(np.sum(v[a] * np.conj(v[b]), axis=1)))
        out[k] = num / np.sqrt(p[a].sum() * p[b].sum())
    return out, npairs


def first_below(rho, level):
    idx = np.where(rho < level)[0]
    if not len(idx):
        return np.nan
    k = idx[0]
    # interpolación lineal entre k-1 y k
    return float(k - 1 + (rho[k - 1] - level) / (rho[k - 1] - rho[k]))


def main():
    parser = argparse.ArgumentParser(description="Discontinuidades, coherencia y variación entre tramos")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--raw", default="data/medidas/ch_sodupe_180_time.npy")
    parser.add_argument("--window", type=int, default=3, help="w: CIR a cada lado del corte")
    parser.add_argument("--threshold", type=float, default=100.0, help="Umbral de r para un corte")
    parser.add_argument("--min-len", type=int, default=10, help="Tramos más cortos se unen al siguiente")
    parser.add_argument("--cir-period", type=float, default=7.616e-3,
                        help="Segundos entre CIR consecutivas (campaña de medida; no consta en el fichero)")
    parser.add_argument("--kmax", type=int, default=600)
    args = parser.parse_args()

    cfg = ev.load_config(args.config)
    L = cfg["data"].get("L", 200) if isinstance(cfg.get("data"), dict) else 200
    save_dir = ev.repo_path(cfg["paths"]["save_dir"])
    x, keep = load_raw(ev.repo_path(args.raw), L)
    n = len(x)
    dt_s = args.cir_period
    lines = [f"Estructura temporal de {args.raw}: {n} CIR, {L} taps; {keep.sum()} conservadas tras el "
             f"filtro de energía (descartadas: {np.where(~keep)[0].tolist()})",
             f"Periodo entre CIR supuesto: {dt_s * 1e3:.3f} ms (de la campaña de medida; no consta en el "
             f"fichero) -> duración total {n * dt_s:.1f} s", ""]

    # ---------------------------------------------------------------- 1. discontinuidades
    r = jump_ratio(x, args.window)
    lr = np.log10(r[np.isfinite(r)])
    hist, edges = np.histogram(lr, bins=np.arange(-1, 4.75, 0.25))
    lines += [f"1. DISCONTINUIDADES (w = {args.window}; r = cambio a través del corte / cambio a cada lado)",
              "Histograma de log10(r) (todas las posiciones):"]
    lines += [f"   [{a:5.2f}, {a + 0.25:5.2f}) {h:6d}" for a, h in zip(edges[:-1], hist)]
    lines.append(f"   mediana de r: {np.median(r[np.isfinite(r)]):.2f}")
    lines.append("Cortes según el umbral (robustez): " + ", ".join(
        f"r>{t:g}: {len(find_cuts(r, t, args.window))}" for t in (30, 100, 300, 1000)))
    raw_cuts = find_cuts(r, args.threshold, args.window)
    cuts = merge_short(raw_cuts, n, args.min_len)
    seg_raw = np.searchsorted(cuts, np.arange(n), side="right")
    bounds = np.r_[0, cuts, n]
    lens = np.diff(bounds)
    n_seg = len(lens)
    lines += [f"Umbral elegido r > {args.threshold:g}: {len(raw_cuts)} cortes; tras unir los tramos de < "
              f"{args.min_len} CIR (transitorios) al siguiente: {len(cuts)} cortes -> {n_seg} tramos continuos",
              f"Longitud de los tramos (CIR): mín {lens.min()}, p25 {np.percentile(lens, 25):.0f}, mediana "
              f"{np.median(lens):.0f}, p75 {np.percentile(lens, 75):.0f}, máx {lens.max()}, media "
              f"{lens.mean():.1f} ({lens.mean() * dt_s:.2f} s; máx {lens.max() * dt_s:.2f} s)"]
    lines.append("Hipótesis de grabaciones de 5 s: 5 s / periodo = "
                 f"{5 / dt_s:.0f} CIR por grabación -> {n / (5 / dt_s):.1f} grabaciones en el fichero")
    z, P, z95 = periodicity(cuts, n, np.arange(30, 1500, 0.25))
    lines.append(f"Periodicidad de los cortes (Rayleigh, periodos 30–1500 CIR): máx {z:.3f} en P = {P:.1f}; "
                 f"percentil 95 con cortes al azar: {z95:.3f} -> "
                 + ("periodo significativo" if z > z95 else "SIN periodo fijo"))
    d = np.diff(cuts)
    for base in (64, 656):
        m = np.maximum(np.round(d / base), 1)
        hits = int(np.sum(np.abs(d - m * base) <= 2))
        lines.append(f"Intervalos entre cortes a ±2 CIR de un múltiplo de {base}: {hits} de {len(d)} "
                     f"(al azar se esperarían ~{len(d) * 5 / base:.0f})")
    # Comprobación de que el canal sí evoluciona dentro de los tramos (si no, un hueco largo podría no
    # dejar marca): correlación de la parte variable entre la 3ª y la antepenúltima CIR de cada tramo
    v_all = x - x[keep].mean(0)

    def rc(a, b):
        return float(np.real(np.vdot(b, a)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))
    long_idx = np.where(lens >= 100)[0]
    inside = [rc(v_all[bounds[j] + 2], v_all[bounds[j + 1] - 3]) for j in long_idx]
    across = [rc(v_all[c - 3], v_all[c + 2]) for c in cuts if 3 <= c < n - 2]
    lines.append(f"Parte variable: correlación entre extremos de un mismo tramo (tramos >= 100 CIR): mediana "
                 f"{np.median(inside):.3f} (rango {min(inside):.2f}–{max(inside):.2f}); a través de un corte "
                 f"(±3 CIR): mediana {np.median(across):.3f}")
    lines.append("")

    # Conjuntos de entrenamiento: índice de tramo de cada canal del dataset filtrado
    raw_index = np.where(keep)[0]
    seg_f = seg_raw[keep]
    split = np.load(ev.repo_path(cfg["data"]["split_file"]))
    lines.append("Tramos por conjunto (dataset filtrado y separación de " + cfg["data"]["split_file"] + "):")
    sets = {}
    for name in ("train", "val", "test"):
        s = np.unique(seg_f[split[name]])
        sets[name] = set(s.tolist())
        full = [j for j in s if np.all(np.isin(np.where(seg_f == j)[0], split[name]))]
        lines.append(f"   {name:<5}: {len(split[name]):5d} canales, {len(s):3d} tramos distintos "
                     f"({len(full)} completos, {len(s) - len(full)} parciales)")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        lines.append(f"   tramos compartidos {a}–{b}: {len(sets[a] & sets[b])}")
    lines.append("")

    # ---------------------------------------------------------------- 2. coherencia
    v = v_all[keep]
    kmax = min(args.kmax, int(lens.max()))
    rho_s, np_s = coherence(v, raw_index, seg_f, kmax, same_segment=True)
    rho_a, np_a = coherence(v, raw_index, seg_f, kmax, same_segment=False)
    # Variante: restando la media de cada tramo (solo la fluctuación dentro del tramo, sin la deriva entre
    # tramos). Sesgada a la baja para k comparable a la longitud del tramo.
    v_loc = v.copy()
    for j in np.unique(seg_f):
        m = seg_f == j
        v_loc[m] -= v[m].mean(0)
    rho_l, _ = coherence(v_loc, raw_index, seg_f, kmax, same_segment=True)
    lines.append("2. COHERENCIA TEMPORAL de la parte variable (v = x - media global), pares de canales "
                 "conservados")
    lines.append(f"{'k':>6}{'t (s)':>9}{'rho mismo tramo':>17}{'pares':>8}{'rho todos':>11}{'pares':>8}"
                 f"{'rho tramo, media local':>24}")
    for k in [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]:
        if k <= kmax:
            lines.append(f"{k:>6}{k * dt_s:>9.3f}{rho_s[k]:>17.3f}{np_s[k]:>8d}{rho_a[k]:>11.3f}{np_a[k]:>8d}"
                         f"{rho_l[k]:>24.3f}")
    summary = {}
    def fmt(k):
        return "no se alcanza" if np.isnan(k) else f"k = {k:6.1f} ({k * dt_s:.3f} s)"
    for lbl, lev in (("0.9", 0.9), ("0.5", 0.5), ("descorrelación (0.1)", 0.1), ("cero", 0.0)):
        ks, ka, kl = first_below(rho_s, lev), first_below(rho_a, lev), first_below(rho_l, lev)
        summary[lbl] = (ks, ka, kl)
        lines.append(f"   rho < {lbl:<21}: mismo tramo {fmt(ks)} | todos los pares {fmt(ka)} | "
                     f"mismo tramo, media local {fmt(kl)}")
    lines.append(f"   (los valores con k grande se basan en pocos tramos: pares a k = 200: {np_s[200] if kmax >= 200 else 0})")
    lines.append("")
    with open(os.path.join(save_dir, "recordings_coherence.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "t_s", "rho_same_segment", "pairs_same_segment", "rho_all_pairs", "pairs_all",
                     "rho_same_segment_local_mean"])
        for k in range(kmax + 1):
            wr.writerow([k, k * dt_s, rho_s[k], np_s[k], rho_a[k], np_a[k], rho_l[k]])

    # ---------------------------------------------------------------- 3. variación entre tramos
    tau = np.arange(L) * ev.sampling_period(cfg)
    p = np.abs(x[keep]) ** 2
    thrs = cfg["evaluation"].get("delay_thresholds_db", [None])
    feats = {}
    for thr in thrs:
        md, rd = ev.delay_moments(p, tau, thr)
        tag = ev.threshold_tag(thr) or "_sin_umbral"
        feats[f"mean_delay{tag}"] = md * 1e9
        feats[f"rms_delay{tag}"] = rd * 1e9
    segs = np.unique(seg_f)
    cnt = np.array([np.sum(seg_f == j) for j in segs])
    rows = []
    for j, c in zip(segs, cnt):
        m = seg_f == j
        row = {"segment": int(j), "raw_start": int(bounds[j]), "raw_end": int(bounds[j + 1]),
               "length_raw": int(lens[j]), "n_kept": int(c), "t_start_s": bounds[j] * dt_s,
               "set": "/".join(nm for nm in ("train", "val", "test") if np.any(np.isin(np.where(m)[0], split[nm])))
               or "-"}
        for k, val in feats.items():
            row[k] = float(val[m].mean())
        rows.append(row)
    with open(os.path.join(save_dir, "recordings_segments.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines.append(f"3. VARIACIÓN ENTRE TRAMOS ({len(segs)} tramos; estadística de las medias por tramo, "
                 "en ns; eta² = fracción de la varianza por canal explicada por el tramo)")
    lines.append(f"{'':<26}{'global':>9}{'mín':>9}{'p10':>9}{'mediana':>9}{'p90':>9}{'máx':>9}"
                 f"{'std':>8}{'std (>=100)':>13}{'eta²':>7}")
    big = cnt >= 100
    for k, val in feats.items():
        mu = np.array([r_[k] for r_ in rows])
        eta2 = np.sum(cnt * (mu - val.mean()) ** 2) / np.sum((val - val.mean()) ** 2)
        lines.append(f"{k:<26}{val.mean():>9.1f}{mu.min():>9.1f}{np.percentile(mu, 10):>9.1f}"
                     f"{np.median(mu):>9.1f}{np.percentile(mu, 90):>9.1f}{mu.max():>9.1f}{mu.std(ddof=1):>8.1f}"
                     f"{mu[big].std(ddof=1):>13.1f}{eta2:>7.2f}")
    lines.append("")
    lines.append("Por conjunto (media de los canales, ns):")
    for name in ("train", "val", "test"):
        lines.append(f"   {name:<5}: " + ", ".join(
            f"{k} {feats[k][split[name]].mean():.1f}" for k in ("rms_delay_sin_umbral", "rms_delay_th20dB",
                                                                 "mean_delay_sin_umbral") if k in feats))

    np.savez(os.path.join(save_dir, "recordings.npz"), segment_raw=seg_raw, segment_filtered=seg_f,
             raw_index_filtered=raw_index, cuts=cuts, cuts_before_merge=raw_cuts, ratio=r,
             threshold=args.threshold, window=args.window, min_len=args.min_len, cir_period_s=dt_s)
    txt = "\n".join(lines)
    with open(os.path.join(save_dir, "recordings.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].hist(lr, bins=np.arange(-1, 4.75, 0.1), color="gray")
    ax[0, 0].axvline(np.log10(args.threshold), color="red", ls="--", label=f"threshold r = {args.threshold:g}")
    ax[0, 0].set_yscale("log")
    ev._style(ax[0, 0], "log10 r (jump ratio)", "Count", legend=True)
    ax[0, 1].hist(lens, bins=np.arange(0, lens.max() + 20, 16), color="steelblue")
    ev._style(ax[0, 1], "Segment length (CIR)", "Count", legend=False)
    k = np.arange(kmax + 1)
    ax[1, 0].plot(k * dt_s, rho_s, label="Same segment")
    ax[1, 0].plot(k * dt_s, rho_a, ls="--", label="All pairs")
    ax[1, 0].plot(k * dt_s, rho_l, ls=":", label="Same segment, local mean removed")
    ax[1, 0].set_xlim(0, 300 * dt_s)
    for lev in (0.9, 0.5, 0.1):
        ax[1, 0].axhline(lev, color="gray", lw=0.6, ls=":")
    ev._style(ax[1, 0], "Time lag (s)", "Correlation of the variable part", legend=True)
    t = raw_index * dt_s
    ax[1, 1].plot(t, feats["rms_delay_sin_umbral"], ",", color="lightgray")
    mids = np.array([(r_["raw_start"] + r_["raw_end"]) / 2 * dt_s for r_ in rows])
    ax[1, 1].plot(mids, [r_["rms_delay_sin_umbral"] for r_ in rows], "o", ms=3, label="No threshold")
    if "rms_delay_th20dB" in feats:
        ax[1, 1].plot(mids, [r_["rms_delay_th20dB"] for r_ in rows], "s", ms=3, label="20 dB")
    for c in cuts:
        ax[1, 1].axvline(c * dt_s, color="red", lw=0.3, alpha=0.4)
    ev._style(ax[1, 1], "Time (s, assumed CIR period)", "Mean RMS delay per segment (ns)", legend=True)
    ev._save(fig, save_dir, "fig_recordings")


if __name__ == "__main__":
    main()
