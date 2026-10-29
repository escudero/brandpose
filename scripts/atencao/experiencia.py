"""Experiência 3D da atenção: a loja do LojaSim no navegador, com o mapa de calor nascendo ao vivo.

Monta uma pasta que abre com dois cliques (sem servidor): index.html + app.js + dados.js (loja .glb
embutida, faces, estatística, trajetos e olhares) + video.mp4 (o vídeo original, leve, sincronizado).

Uso: python experiencia.py ANALISE_DIR VIDEO --cena C --loja L
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np

import cena3d
from cena3d import _cam

AQUI = Path(__file__).resolve().parent
CORES = ["#4f8cff", "#3ddc97", "#ffc53d", "#ff5d5d", "#c77dff", "#2ee6e6", "#ff8a3d", "#b8e05a", "#7a8cff", "#ff6fae",
         "#55d6ff", "#ffd86b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analise", type=Path)
    ap.add_argument("video")
    ap.add_argument("--cena", type=Path, required=True)
    ap.add_argument("--loja", type=Path, required=True)
    args = ap.parse_args()

    out = args.analise / "experiencia"
    out.mkdir(exist_ok=True)
    cena = cena3d.Cena(args.cena, args.loja)
    res = json.loads((args.analise / "atencao.json").read_text())
    tj = json.loads((args.analise / "trajetos.json").read_text())
    olh = json.loads((args.analise / "olhares.json").read_text())
    fps = tj["fps"]

    est = {f["face"]: f for f in res["faces"]}
    faces = [{"id": f.id, "item": f.item, "nome": est[f.id]["nome"], "lado": f.lado, "c": f.c.round(3).tolist(),
              "n": f.n.round(3).tolist(), "a1": f.a1.round(3).tolist(), "a2": f.a2.round(3).tolist(),
              "w1": round(f.w1, 3), "w2": round(f.w2, 3), "seg": est[f.id]["segundos"], "ic": est[f.id]["ic_segundos"],
              "parcela": est[f.id]["parcela"], "pessoas": est[f.id]["pessoas"]} for f in cena.faces]
    moveis = [{"id": b["id"], "nome": cena.nome(b["id"]), "tipo": b["tipo"], "x": float(b["c"][0]), "y": float(b["c"][1]),
               "H": b["H"]} for b in cena.caixas]
    ix = {f.id: k for k, f in enumerate(cena.faces)}
    clientes = [p for p in tj["pessoas"] if not p["equipe"]]
    PASSO = 3
    pessoas = [{"id": p["id"], "cor": CORES[i % len(CORES)], "q0": p["q0"], "z": p["z_olhos"],
                "xy": np.round(np.c_[p["x"][::PASSO], p["y"][::PASSO]], 2).ravel().tolist()} for i, p in enumerate(clientes)]
    ids = {p["id"] for p in clientes}
    amostras = []
    for a in olh:
        if a["pessoa"] not in ids:
            continue
        alvos = sorted(a["alvos"], key=lambda t: -t[3])[:16]
        escala = sum(t[3] for t in a["alvos"]) / max(sum(t[3] for t in alvos), 1e-9)  # os 16 maiores levam o peso de todos
        amostras.append({"q": a["q"], "p": a["pessoa"], "O": a["O"], "k": round(a["dt"] * a["inout"], 4),
                         "a": [[t[0], t[1], t[2], round(t[3] * escala, 5), t[4], t[5], t[6]] for t in alvos]})
    M = _cam.cam_matrix(cena.cam.get("yaw", 0.0), cena.cam["pitch"], cena.cam["roll"])
    camera = {"h": cena.cam["h"], "f": cena.cam["f"], "W": cena.cam["W"], "H": cena.cam["H"],
              "direita": M[:, 0].tolist(), "cima": (-M[:, 1]).tolist(), "tras": (-M[:, 2]).tolist()}
    dados = {"fps": fps, "quadros": tj["quadros"], "passo": PASSO, "faces": faces, "moveis": moveis, "pessoas": pessoas,
             "amostras": amostras, "camera": camera, "resumo": res["resumo"], "faixas": res["faixas"],
             "ranking": res["moveis"][:8], "cel": 0.05}
    glb = base64.b64encode((args.loja / "loja_final.glb").read_bytes()).decode()
    (out / "dados.js").write_text("window.DADOS=" + json.dumps(dados, separators=(",", ":")) + ";\nwindow.GLB='" + glb + "';\n")
    app = (AQUI / "web" / "app.js").read_text()
    (out / "index.html").write_text((AQUI / "web" / "index.html").read_text().replace("/*APP*/", app))  # módulo embutido: abre por file://
    if not (out / "video.mp4").exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-vf", "scale=960:-2", "-an", "-c:v", "libx264",
                        "-crf", "26", "-preset", "veryfast", "-movflags", "+faststart", str(out / "video.mp4")], check=True)
    print("experiência", out / "index.html")


if __name__ == "__main__":
    main()
