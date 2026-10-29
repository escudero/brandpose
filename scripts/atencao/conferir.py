"""Vídeo de conferência: o olhar em 3D desenhado sobre o vídeo original da câmera.

Para checar a olho se a análise faz sentido: cada cliente com o pé no piso (círculo), o
trajeto, e linhas dos olhos até os pontos das faces que o olhar atingiu (mais forte = mais
peso). Os móveis do modelo aparecem em contorno: se o olhar cai onde não há móvel, falta
móvel no modelo. No canto, a planta vista de cima.

Uso: python conferir.py VIDEO ANALISE_DIR --cena C --loja L [--escala 0.5] [--ate SEG]
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import cena3d

CORES = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (171, 71, 188), (0, 172, 193),
         (255, 112, 67), (158, 157, 36), (92, 107, 192), (240, 98, 146)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("analise", type=Path)
    ap.add_argument("--cena", type=Path, required=True)
    ap.add_argument("--loja", type=Path, required=True)
    ap.add_argument("--escala", type=float, default=0.5)
    ap.add_argument("--ate", type=float)
    args = ap.parse_args()

    cena = cena3d.Cena(args.cena, args.loja)
    tj = json.loads((args.analise / "trajetos.json").read_text())
    olh = json.loads((args.analise / "olhares.json").read_text())
    pessoas = [p for p in tj["pessoas"] if not p["equipe"]]
    cor = {p["id"]: CORES[i % len(CORES)][::-1] for i, p in enumerate(pessoas)}
    por_q = {}
    for a in olh:
        por_q.setdefault(a["q"], []).append(a)
    W, H = cena.cam["W"], cena.cam["H"]
    sw, sh = int(W * args.escala), int(H * args.escala)

    # contorno dos móveis e das faces de exposição, uma vez só
    base = np.zeros((H, W, 3), np.uint8)
    for f in cena.faces:
        cs = [f.canto(0, f.z0), f.canto(f.w1, f.z0), f.canto(f.w1, f.z0 + f.w2), f.canto(0, f.z0 + f.w2)]
        q = cena.projetar(np.array(cs)).astype(np.int32)
        cv2.polylines(base, [q], True, (0, 220, 255), 2, cv2.LINE_AA)

    # planta de cima
    xs = np.r_[[b["c"][0] for b in cena.caixas], [v for p in pessoas for v in p["x"]]]
    ys = np.r_[[b["c"][1] for b in cena.caixas], [v for p in pessoas for v in p["y"]]]
    x0, x1, y0, y1 = np.percentile(xs, 1) - 1, np.percentile(xs, 99) + 1, np.percentile(ys, 1) - 1, np.percentile(ys, 99) + 1
    PL = 360
    esc = PL / max(x1 - x0, y1 - y0)
    pl = lambda x, y: (int((x - x0) * esc), int(PL - (y - y0) * esc))
    planta = np.full((PL, PL, 3), 30, np.uint8)
    for b in cena.caixas:
        cs = [b["c"] + b["R"] @ np.array([sx * b["L"] / 2, sy * b["d"] / 2]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        cv2.fillPoly(planta, [np.array([pl(*c) for c in cs], np.int32)], (90, 90, 90))
    for f in cena.faces:
        a, b = f.canto(0, 0), f.canto(f.w1, 0)
        cv2.line(planta, pl(*a[:2]), pl(*b[:2]), (0, 220, 255), 2)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    saida = args.analise / "conferencia.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{sw}x{sh}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(saida)],
                          stdin=subprocess.PIPE)
    ultimo = {}
    q = 0
    while True:
        ok, img = cap.read()
        if not ok or (args.ate and q >= args.ate * fps):
            break
        img = cv2.addWeighted(img, 1.0, base, 0.5, 0)
        for a in por_q.get(q, []):
            ultimo[a["pessoa"]] = a
        pl_q = planta.copy()
        for p in pessoas:
            k = q - p["q0"]
            if not 0 <= k < len(p["x"]):
                continue
            c = cor[p["id"]]
            tr = np.c_[p["x"][max(0, k - 90):k + 1], p["y"][max(0, k - 90):k + 1], np.zeros(min(k, 90) + 1)]
            cv2.polylines(img, [cena.projetar(tr).astype(np.int32)], False, c, 2, cv2.LINE_AA)
            pe = cena.projetar(np.array([[p["x"][k], p["y"][k], 0]]))[0].astype(int)
            cv2.circle(img, tuple(pe), 9, c, 2, cv2.LINE_AA)
            cv2.circle(pl_q, pl(p["x"][k], p["y"][k]), 5, c, -1)
            a = ultimo.get(p["id"])
            if a is None or q - a["q"] > 6:
                continue
            O = np.array(a["O"])
            o2 = cena.projetar(O[None])[0].astype(int)
            alvos = sorted(a["alvos"], key=lambda t: -t[3])[:40]
            wmax = max([t[3] for t in alvos], default=1)
            for f, s1, s2, w, *P in alvos:
                p2 = cena.projetar(np.array([P]))[0].astype(int)
                forca = w / wmax
                cv2.line(img, tuple(o2), tuple(p2), c, 1 + int(2 * forca), cv2.LINE_AA)
                cv2.circle(img, tuple(p2), 3 + int(4 * forca), c, -1, cv2.LINE_AA)
                cv2.line(pl_q, pl(*O[:2]), pl(*P[:2]), c, 1)
            if alvos:
                f = cena.faces[alvos[0][0]]
                txt = f"{cena.nome(f.item)} {int(round(a['fracao'] * 100))}%"
                cv2.putText(img, txt, (o2[0] + 12, o2[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, txt, (o2[0] + 12, o2[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        ps = int(PL * args.escala * 1.4)
        img[sh - ps - 8:sh - 8, sw - ps - 8:sw - 8] = cv2.resize(pl_q, (ps, ps))
        ff.stdin.write(img.tobytes())
        q += 1
    ff.stdin.close()
    ff.wait()
    print("vídeo", saida, q, "quadros")


if __name__ == "__main__":
    main()
