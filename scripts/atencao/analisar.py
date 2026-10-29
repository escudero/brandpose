"""Análise de atenção nas gôndolas: vídeo de CCTV + loja em 3D do LojaSim → estatística por face.

Entradas:
  --dados  saída do promo/extract_v2.py (pessoas, olhar do Gaze-LLE, silhuetas)
  --cena   cena do LojaSim (câmera calibrada sem cliques: camera_final.json)
  --loja   loja montada no LojaSim (plano_final.json: móveis, frente, lados)
Saídas em --saida: trajetos.json, olhares.json, atencao.json, mapas.npz

Uso: python analisar.py --dados D --cena C --loja L --saida S
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cena3d
import estatistica
import olhar3d
import trajetos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", type=Path, required=True)
    ap.add_argument("--cena", type=Path, required=True)
    ap.add_argument("--loja", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cena = cena3d.Cena(args.cena, args.loja)
    meta = json.loads((args.dados / "meta.json").read_text())
    frames = json.loads((args.dados / "tracks.json").read_text())
    gaze = json.loads((args.dados / "gaze.json").read_text())
    heatmaps = np.load(args.dados / "gaze_heatmaps.npy", mmap_mode="r")
    masks = np.load(args.dados / "person_masks_packed.npy", mmap_mode="r")
    print(f"{len(cena.itens)} móveis, {len(cena.faces)} faces de exposição, {len(gaze)} olhares")

    pessoas, _ = trajetos.calcular(cena, frames, meta["fps"], meta["width"], meta["height"])
    print(f"{len(pessoas)} trajetos ({sum(p['equipe'] for p in pessoas)} da equipe), {time.time() - t0:.0f}s")
    (args.saida / "trajetos.json").write_text(json.dumps({"fps": meta["fps"], "quadros": meta["frames"], "pessoas": pessoas}))

    amostras = olhar3d.projetar_olhares(cena, pessoas, gaze, heatmaps, masks, meta, frames)
    print(f"{len(amostras)} olhares em 3D, {time.time() - t0:.0f}s")
    (args.saida / "olhares.json").write_text(json.dumps(amostras))

    res, mapas = estatistica.calcular(cena, pessoas, amostras, meta["frames"] / meta["fps"])
    res["faces_geo"] = [{"face": f.id, "c": f.c.round(3).tolist(), "n": f.n.round(3).tolist(), "a1": f.a1.round(3).tolist(),
                         "a2": f.a2.round(3).tolist(), "w1": f.w1, "w2": f.w2} for f in cena.faces]
    res["fontes"] = {"dados": str(args.dados), "cena": str(args.cena), "loja": str(args.loja)}
    (args.saida / "atencao.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    np.savez_compressed(args.saida / "mapas.npz", **{f.id: m for f, m in zip(cena.faces, mapas)})
    r = res["resumo"]
    print(f"clientes {r['clientes']}, olhar estimado {r['olhar_estimado_s']} s, em gôndolas {r['atencao_gondolas_s']} s "
          f"({r['fracao_em_gondolas']:.0%}); 1º lugar segue 1º em {r['chance_lider_continuar_1a']:.0%} dos sorteios")
    print("destino do olhar:", ", ".join(f"{k} {v:.0%}" for k, v in r["destino_do_olhar"].items()))
    for f in res["faces"][:10]:
        print(f"  {f['nome']:<40} {f['segundos']:6.1f} s [{f['ic_segundos'][0]:.1f}–{f['ic_segundos'][1]:.1f}] "
              f"{f['parcela']:5.1%}  {f['pessoas']} pessoas")
    for f in res["faixas"]:
        print(f"  {f['faixa']:<24} {f['parcela']:5.1%} [{f['ic_parcela'][0]:.0%}–{f['ic_parcela'][1]:.0%}]")
    print(f"pronto em {time.time() - t0:.0f}s → {args.saida}")


if __name__ == "__main__":
    main()
