"""
Sensibilidad del bootstrap al tamaño de bloque: compara los IC95 y P(< Gaussian) de seed_stats.py
ejecutado con distintos bloques (bootstrap_ci.csv = bloque 50; bootstrap_ci_block<B>.csv = otros).

Uso (desde la raíz del repositorio, tras seed_stats.py --block 25 / 100 / 500 / segment):
    python evaluation_models/bootstrap_sensitivity.py
Resultados en <paths.save_dir>/bootstrap_sensitivity.txt y .csv.
"""
import argparse
import csv
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import evaluation as ev  # noqa: E402

KEY_METRICS = ["null_depth_db_ks", "ripple_db_ks", "zf_gain_db_ks", "pdp_mae_db", "ber_QPSK_dlog10",
               "ber_16QAM_dlog10", "mag_ks", "rms_delay_ks", "rms_delay_th20dB_ks", "fade_frac_ks", "se_ks"]
SOURCES = ["real_train", "real_val", "gan", "dcgan", "wgan", "ddpm", "ddim", "tdl", "gauss"]
LABEL = {"real_train": "Real train", "real_val": "Real val", **ev.MODEL_LABELS}


def main():
    parser = argparse.ArgumentParser(description="Sensibilidad del bootstrap al tamaño de bloque")
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "evaluation_config.json"))
    parser.add_argument("--blocks", nargs="+", default=["25", "50", "100", "500", "segment"])
    args = parser.parse_args()
    save_dir = ev.repo_path(ev.load_config(args.config)["paths"]["save_dir"])

    data = {}
    for b in args.blocks:
        path = os.path.join(save_dir, "bootstrap_ci.csv" if b == "50" else f"bootstrap_ci_block{b}.csv")
        if not os.path.exists(path):
            print(f"Falta {path}, se omite el bloque {b}")
            continue
        data[b] = {(r["metric"], r["source"]): r for r in csv.DictReader(open(path))}
    blocks = list(data)
    name = {b: ("tramo" if b == "segment" else b) for b in blocks}

    rows = []
    lines = ["Sensibilidad del bootstrap al bloque (test, train y val remuestreados por bloques; 'tramo' = un "
             "bloque por tramo continuo de recordings.py)",
             "Para cada métrica: anchura del IC95 de la media entre semillas y P(< Gaussian) emparejada.",
             "** P > 0.975 (mejor que el gaussiano, 5 % bilateral); †† P < 0.025 (peor).", ""]
    for m in KEY_METRICS:
        if (m, "gauss") not in data[blocks[0]]:
            continue
        lines.append(f"== {m}")
        lines.append(f"  {'':<12}" + "".join(f"{'IC95 ' + name[b]:>22}" for b in blocks)
                     + "".join(f"{'P ' + name[b]:>10}" for b in blocks))
        for s in SOURCES:
            if (m, s) not in data[blocks[0]]:
                continue
            ci, ps = [], []
            for b in blocks:
                r = data[b][(m, s)]
                lo, hi, p = float(r["ci_low"]), float(r["ci_high"]), float(r["p_better_than_gauss"])
                ci.append(f"[{lo:.3f}, {hi:.3f}]")
                mark = "" if np.isnan(p) else ("**" if p > 0.975 else "††" if p < 0.025 else "")
                ps.append("" if np.isnan(p) else f"{p:.3f}{mark}")
                rows.append({"metric": m, "source": s, "block": name[b], "mean": float(r["mean"]),
                             "ci_low": lo, "ci_high": hi, "ci_width": hi - lo, "p_better_than_gauss": p})
            lines.append(f"  {LABEL[s]:<12}" + "".join(f"{c:>22}" for c in ci) + "".join(f"{p:>10}" for p in ps))
        lines.append("")

    # Resumen: número de comparaciones significativas por bloque (modelos generativos frente al gaussiano)
    gen = ["gan", "dcgan", "wgan", "ddpm", "ddim"]
    lines.append("Resumen: modelos generativos significativamente mejores (P > 0.975) / peores (P < 0.025) que "
                 "el gaussiano, de 5, y anchura media relativa del IC95 del gaussiano (respecto al bloque 50)")
    lines.append(f"  {'':<22}" + "".join(f"{name[b]:>14}" for b in blocks))
    for m in KEY_METRICS:
        if (m, "gauss") not in data[blocks[0]]:
            continue
        cells = []
        w50 = (float(data["50"][(m, "gauss")]["ci_high"]) - float(data["50"][(m, "gauss")]["ci_low"])) \
            if "50" in data else np.nan
        for b in blocks:
            p = [float(data[b][(m, g)]["p_better_than_gauss"]) for g in gen]
            w = float(data[b][(m, "gauss")]["ci_high"]) - float(data[b][(m, "gauss")]["ci_low"])
            cells.append(f"{sum(x > 0.975 for x in p)}/{sum(x < 0.025 for x in p)} ×{w / w50:.2f}")
        lines.append(f"  {m:<22}" + "".join(f"{c:>14}" for c in cells))

    with open(os.path.join(save_dir, "bootstrap_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = "\n".join(lines)
    with open(os.path.join(save_dir, "bootstrap_sensitivity.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
