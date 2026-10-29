"""Vídeo promocional da visão do cliente, para o LinkedIn: vertical 4:5 (1080x1350), 30 fps.

Feito para tocar sem som: a imagem ocupa o quadro inteiro e o texto grande, por cima
(com degradês para ler), conta a história desde o primeiro quadro. Cenas:
  gancho   a câmera de segurança real, num recorte 4:5 por onde ele entra
  congela  o tempo para em 18,5 s e a imagem se aproxima dele
  revela   uma cortina troca o vídeo pela loja em 3D, do mesmo ponto de vista
  voo      a câmera sai da CCTV e entra nos olhos dele (shots.py)
  visao    o que ele vê (shots.py), com a câmera real em sincronia num canto
  mapa     sem teto: o caminho dele, para onde olhou e o tempo por zona
  fim      a mensagem e os créditos

Uso: python promo.py DATA_DIR RUN_DIR [--lang en] [--stills 0,3,6,10,16,22,27,34]
Grava RUN_DIR/promo_<idioma>_mudo.mp4, promo_<idioma>_timeline.json e promo_<idioma>_thumb.jpg;
o promo_audio.py faz a trilha e junta tudo. Com --stills, só os quadros desses instantes (s).
"""

import argparse
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import calib
import layout
from compose import mark, writer
from trajectories import MERGE
from warp import distort

W, H, FPS = 1080, 1350, 30
CX = 640  # centro do recorte 4:5 na imagem da CCTV (o mesmo do shots.py): ele entra pela esquerda
CW = calib.H * W // H  # 864 px: a largura do recorte, com a altura toda da CCTV
WHITE, LIGHT, GRAY, DIM = (255, 255, 255), (228, 231, 237), (190, 194, 202), (150, 154, 162)
YELLOW = (255, 196, 46)
WHO, FREEZE, F_FAST, F_END = 12, 555, 810, 1050  # quadros do vídeo de origem
SRC0 = 435  # o gancho começa em 14,5 s
VOO_N, MAPA_N = 150, 225
LENS_T = 0.4  # no começo do voo, a lente da CCTV se desfaz (s)
HEAD_Y, SUB_Y = 150, 292  # centro da manchete e da linha de apoio, em cima
TEXTS = {
    "pt": {
        "dec": ",",
        "hook": "E se a câmera da loja mostrasse o que o cliente vê?",
        "hook_sub": "Câmera de segurança real, numa loja de conveniência",
        "enters": "Este cliente acabou de entrar.",
        "freeze": "Vamos parar o tempo.",
        "reveal": "A loja inteira vira 3D…",
        "reveal_sub": "reconstruída a partir dessa única câmera",
        "fly": "…e a câmera entra nos olhos dele.",
        "fly_sub": "tempo congelado em 18,5 s",
        "pov": "O que ele vê, a cada instante.",
        "pov2": "Para onde ele olha vem da IA.",
        "walk": "Ele segue pela loja.",
        "looking": "OLHANDO PARA",
        "real": "câmera real, no mesmo instante",
        "map": "Cada olhar vira dado.",
        "legend": ("caminho dele", "para onde olhou"),
        "stats": "Em {s} s, o olhar dele parou em:",
        "end1": "Toda loja já tem câmeras.",
        "end2": "Agora elas podem mostrar o que o cliente vê.",
        "author": "Claudio Escudero  ·  projeto pessoal, código aberto",
        "credits": ["Imagens: demonstração pública da câmera ACTi E43A (© ACTi Corporation)",
                    "3D: Blender  ·  olhar: Gaze-LLE  ·  pessoas: YOLO11"],
    },
    "en": {
        "dec": ".",
        "hook": "What if the store camera could show what shoppers see?",
        "hook_sub": "A real security camera in a convenience store",
        "enters": "This shopper just walked in.",
        "freeze": "Let's freeze time.",
        "reveal": "The whole store turns 3D…",
        "reveal_sub": "rebuilt from this one camera",
        "fly": "…and the camera steps into his eyes.",
        "fly_sub": "time frozen at 18.5 s",
        "pov": "What he sees, moment by moment.",
        "pov2": "Where he looks comes from AI.",
        "walk": "He keeps walking.",
        "looking": "LOOKING AT",
        "real": "real camera, same moment",
        "map": "Every glance becomes data.",
        "legend": ("his path", "where he looked"),
        "stats": "In {s} s, his gaze landed on:",
        "end1": "Every store already has cameras.",
        "end2": "Now they can show what shoppers see.",
        "author": "Claudio Escudero  ·  personal project, open source",
        "credits": ["Footage: public demo of the ACTi E43A camera (© ACTi Corporation)",
                    "3D: Blender  ·  gaze: Gaze-LLE  ·  people: YOLO11"],
    },
}
LINK = "github.com/escudero/brandpose"
LINKEDIN = "linkedin.com/in/claudioescudero"


def font(size, weight="bold"):
    """Avenir Next, como no vídeo de demonstração; fora do macOS, a fonte padrão do Pillow."""
    try:
        index = {"bold": 0, "demi": 2, "medium": 5, "heavy": 8}[weight]
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size, index=index)
    except OSError:
        return ImageFont.load_default(size)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def fade(t, t0, t1, fin=0.3, fout=0.25):
    """Opacidade de algo visível entre t0 e t1 (fin=0: já entra inteiro)."""
    if t < t0 or t > t1:
        return 0.0
    a = 1.0 if fin == 0 else ease((t - t0) / fin)
    return min(a, ease((t1 - t) / fout) if fout else 1.0)


def wrap(d, s, f, width):
    lines, cur = [], ""
    for word in s.split():
        test = f"{cur} {word}".strip()
        if d.textlength(test, font=f) <= width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def scrim(y0, y1, a, p=1.5):
    """Degradê de preto por cima da imagem: opacidade a na linha y0, zero a partir de y1."""
    u = np.clip((np.arange(H, dtype=np.float32) - y1) / (y0 - y1), 0, 1)
    return a * u ** p


class Layer:
    """Textos e caixas com transparência sobre o quadro (RGBA do Pillow), com sombra suave."""

    def __init__(self):
        self.im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, size, fill=WHITE, a=1.0, anchor="la", weight="bold", rise=0.0, track=0):
        if a <= 0:
            return
        f = font(size, weight)
        x, y = xy[0], xy[1] + (1 - a) * rise
        col = (*fill, int(255 * a))
        if not track:
            self.d.text((x, y), s, font=f, fill=col, anchor=anchor)
            return
        total = sum(self.d.textlength(c, font=f) + track for c in s) - track
        x -= {"m": total / 2, "r": total}.get(anchor[0], 0)
        for c in s:
            self.d.text((x, y), c, font=f, fill=col, anchor="l" + anchor[1])
            x += self.d.textlength(c, font=f) + track

    def block(self, cx, y, s, size, width, a=1.0, fill=WHITE, weight="bold", rise=18, spacing=1.18):
        """Parágrafo centrado em cx, com quebra de linha; y é o centro vertical do bloco."""
        if a <= 0:
            return
        f = font(size, weight)
        n = len(wrap(self.d, s, f, width))
        while n > 1 and width > 200 and len(wrap(self.d, s, f, width - 8)) == n:
            width -= 8  # linhas equilibradas, sem uma palavra sozinha na última
        lines = wrap(self.d, s, f, width)
        y0 = y - (len(lines) - 1) * size * spacing / 2
        for i, line in enumerate(lines):
            self.text((cx, y0 + i * size * spacing), line, size, fill, a, "mm", weight, rise)

    def box(self, xy, fill, a=1.0, r=14):
        if a > 0:  # fill em RGB ou RGBA; a multiplica a opacidade
            self.d.rounded_rectangle(xy, r, fill=(*fill[:3], int((fill[3] if len(fill) == 4 else 255) * a)))

    def onto(self, img):
        alpha = np.array(self.im.getchannel("A"))
        shadow = np.roll(cv2.GaussianBlur(alpha, (0, 0), 6), 3, axis=0).astype(np.float32) / 255 * 0.7
        img = (img * (1 - shadow[..., None])).astype(np.uint8)
        base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        return cv2.cvtColor(np.array(Image.alpha_composite(base, self.im).convert("RGB")), cv2.COLOR_RGB2BGR)


class Video:
    """Leitura em ordem do vídeo de origem, voltando só quando precisa."""

    def __init__(self, path):
        self.cap, self.pos, self.img = cv2.VideoCapture(str(path)), -1, None

    def get(self, f):
        if f != self.pos:
            if f < self.pos or f > self.pos + 90:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                self.pos = f - 1
            while self.pos < f:
                self.img = self.cap.read()[1]
                self.pos += 1
        return self.img.copy()


def fit(img):
    return img if img.shape[1] == W else cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def crop(img, s=1.0, center=(CX, calib.H / 2)):
    """Recorte 4:5 da imagem da CCTV com zoom s em volta de center (pixels dela), no tamanho do vídeo."""
    h, w = img.shape[:2]
    cw, ch = CW / s, h / s
    cx = min(max(center[0], cw / 2), w - cw / 2)
    cy = min(max(center[1], ch / 2), h - ch / 2)
    M = np.array([[W / cw, 0, -(cx - cw / 2) * W / cw], [0, H / ch, -(cy - ch / 2) * H / ch]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC), M


def zone_labels(zones, short=15, ramp=6):
    """Rótulo da zona olhada em cada quadro, (zona, opacidade). Sem piscar: a zona mais frequente em ~0,4 s;
    sem zona por até `short` quadros, fica a anterior; por mais tempo, o rótulo some e volta com a próxima."""
    steady = [Counter(zones[max(0, k - 6): k + 7]).most_common(1)[0][0] for k in range(len(zones))]
    runs = [(z, len(list(g))) for z, g in groupby(steady)]
    out = []
    for i, (z, n) in enumerate(runs):
        prev, nxt = runs[i - 1][0] if i else None, runs[i + 1][0] if i + 1 < len(runs) else None
        if z or (n <= short and (prev or nxt)):
            out += [(z or prev or nxt, 1.0)] * n
            continue
        for j in range(n):
            near = prev if j < n / 2 else nxt
            out.append((near, 1 - min(1.0, min(j + 1, n - j) / ramp) if near else 0.0))
    return out


class Promo:
    def __init__(self, data_dir, run_dir, lang):
        self.tx, self.names = TEXTS[lang], layout.zone_names(lang)
        self.cam = calib.load_camera()
        self.tracks = json.loads((data_dir / "tracks.json").read_text())
        self.video = Video(json.loads((data_dir / "meta.json").read_text())["video"])
        self.shots = run_dir / f"promo_{lang}"
        self.info = json.loads((run_dir / f"pov_{lang}" / "pov_info.json").read_text())
        self.f0 = self.info[0]["frame"]
        zones = [i["zone"] for i in self.info]
        self.label = zone_labels(zones)
        # tempo por zona no trecho todo (o caminho do mapa também é o trecho todo)
        n = Counter(z for z in zones if z)
        self.stats = [(z, c / FPS) for z, c in n.most_common(4)]
        self.span = len(zones) / FPS
        near = self.shots / "voo.json"
        self.near = json.loads(near.read_text())["head_distance"] if near.exists() else []
        me = next(p for p in self.tracks[FREEZE] if MERGE.get(p["id"], p["id"]) == WHO)
        self.head = ((me["head"][0] + me["head"][2]) / 2, (me["head"][1] + me["head"][3]) / 2)
        self.raw = cv2.imread(str(self.shots / "cctv_raw.png"), cv2.IMREAD_UNCHANGED)
        self.cg = self.lens(1.0)
        # degradês: em cima, para a manchete (mais alto quando tem linha de apoio); embaixo, conforme a cena
        self.top, self.top_sub = scrim(0, 430, 0.72), scrim(0, 600, 0.82, 1.1)
        self.low = {"leve": scrim(H, 1130, 0.62), "visao": scrim(H, 800, 0.86), "mapa": scrim(H, 700, 0.92),
                    "fim": scrim(H, 740, 0.9)}
        seg = [("gancho", (FREEZE - SRC0) / FPS), ("congela", 1.2), ("revela", 2.4), ("voo", LENS_T + VOO_N / FPS),
               ("visao", (F_FAST - FREEZE) / FPS + (F_END - F_FAST) / FPS / 3), ("mapa", 7.8), ("fim", 6.0)]
        self.seg, t = {}, 0.0
        for name, d in seg:
            self.seg[name] = (t, t + d)
            t += d
        self.total = t

    def lens(self, w):
        """O quadro 3D da CCTV com a fração w da distorção da lente, recortado como o vídeo real.
        Com w=1 casa com o vídeo; com w=0 é o primeiro quadro do voo."""
        if self.raw is None:
            return self.missing("cctv_raw.png")
        x0 = round((CX - CW / 2) * H / calib.H)
        full = distort(self.raw, dict(self.cam, k1=self.cam["k1"] * w), (calib.W * H // calib.H, H))
        return full[:, x0:x0 + W].copy()

    def missing(self, name):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(img, f"falta {name}", (40, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img

    def shot(self, name, k):
        img = cv2.imread(str(self.shots / name / f"{k + 1:04d}.png"))
        return fit(img) if img is not None else self.missing(f"{name}/{k + 1:04d}.png")

    def pov_frame(self, f):
        img = cv2.imread(str(self.shots / "visao" / f"{f:04d}.png"))
        if img is None:
            return self.missing(f"visao/{f:04d}.png")
        img = fit(img)
        c = (W // 2, H // 2)
        cv2.circle(img, c, 18, YELLOW[::-1], 3, cv2.LINE_AA)
        cv2.circle(img, c, 4, YELLOW[::-1], -1, cv2.LINE_AA)
        return img

    def cctv_marked(self, f, size):
        img = self.video.get(f)
        mark(img, self.info[f - self.f0], self.tracks[f], self.cam)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def visao_source(self, u):
        """Quadro de origem no instante u da cena da visão: 1x até F_FAST, depois 3x."""
        slow = (F_FAST - FREEZE) / FPS
        f = FREEZE + u * FPS if u < slow else F_FAST + (u - slow) * FPS * 3
        return min(int(round(f)), F_END)

    def frame(self, t):
        tx, S = self.tx, self.seg
        L = Layer()
        name = next((n for n, (a, b) in S.items() if a <= t < b), "fim")
        t0, t1 = S[name]
        u = t - t0
        low, inset = self.low["leve"], None
        if name == "gancho":
            f = SRC0 + int(u * FPS)
            img, M = crop(self.video.get(f))
            me = next((p for p in self.tracks[f] if MERGE.get(p["id"], p["id"]) == WHO), None)
            a = fade(t, 2.9, 99, 0.4)
            if me and a > 0:  # ele entra: contorno amarelo
                x0, y0, x1, y1 = me["box"]
                X0, Y0 = M @ np.array([x0, y0, 1.0])
                X1, Y1 = M @ np.array([x1, min(y1, 1075), 1.0])
                L.d.rounded_rectangle((X0 - 8, Y0 - 8, X1 + 8, Y1 + 8), 16, outline=(*YELLOW, int(255 * a)), width=5)
        elif name in ("congela", "revela"):
            src = self.video.get(FREEZE)
            if name == "congela":
                s = 1 + 0.45 * ease(u / 1.2)
            else:
                s = 1 + 0.45 * (1 - ease(u / 0.35))
            k = (s - 1) / 0.45
            center = (CX + (self.head[0] - CX) * 0.85 * k, calib.H / 2 + (self.head[1] - calib.H / 2) * 0.85 * k)
            img, M = crop(src, s, center)
            hx, hy = M @ np.array([self.head[0], self.head[1], 1.0])
            z = M[0, 0]  # pixels do vídeo por pixel da CCTV
            dark = (0.68 * ease(u / 0.6)) if name == "congela" else 0.68 * (1 - ease(u / 0.3))
            if dark > 0:  # luz sobre ele
                yy, xx = np.mgrid[0:H, 0:W]
                r = np.hypot((xx - hx) * 1.3, yy - (hy + 160 * z))
                m = np.clip((r - 213 * z) / (302 * z), 0, 1) * dark
                img = (img * (1 - m[..., None])).astype(np.uint8)
            if name == "revela" and u >= 0.35:
                xw = int(W * ease((u - 0.35) / 1.2))
                img[:, :xw] = self.cg[:, :xw]
                if 0 < xw < W:  # a linha da cortina, com brilho
                    line = np.zeros(img.shape[:2], np.float32)
                    line[:, max(xw - 2, 0):xw + 2] = 1.0
                    glow = cv2.GaussianBlur(line, (0, 0), 10) * 1.5 + line
                    img = np.clip(img + glow[..., None] * 255, 0, 255).astype(np.uint8)
        elif name == "voo" and u < LENS_T:  # a lente da CCTV se desfaz: vira o primeiro quadro do voo
            img = self.lens(1 - ease(u / LENS_T))
        elif name == "voo":
            k = min(int((u - LENS_T) * FPS), VOO_N - 1)
            img = self.shot("voo", k)
            d = self.near[k] if k < len(self.near) else 9.0
            if 0.45 <= d < 1.2:  # perto da cabeça: o corpo some aos poucos
                bare = cv2.imread(str(self.shots / "voo_sem" / f"{k + 1:04d}.png"))
                if bare is not None:
                    a = ease((d - 0.45) / 0.75)
                    img = cv2.addWeighted(img, a, fit(bare), 1 - a, 0)
        elif name == "visao":
            f = self.visao_source(u)
            img = self.pov_frame(f)
            if u < 0.15:
                img = cv2.addWeighted(self.shot("voo", VOO_N - 1), 1 - u / 0.15, img, u / 0.15, 0)
            low = self.low["visao"]
            # num canto: a câmera real e a zona olhada
            pw, ph = 432, 243
            px, py = 40, H - 96 - ph
            inset = (self.cctv_marked(f, (pw, ph)), px, py)
            L.d.rounded_rectangle((px - 2, py - 2, px + pw + 2, py + ph + 2), 10, outline=(*DIM, 255), width=2)
            L.text((px, py + ph + 26), tx["real"], 22, GRAY, 1.0, "ls", "medium")
            z, a = self.label[f - self.f0]
            lx = px + pw + 36
            L.text((lx, py + 22), tx["looking"], 24, GRAY, a, "lm", "demi", track=3)
            if z:
                lines = wrap(L.d, self.names[z], font(46, "bold"), W - lx - 40)[:3]
                L.d.rectangle((lx, py + 50, lx + 6, py + 46 + 52 * len(lines)), fill=(*YELLOW, int(255 * a)))
                for i, line in enumerate(lines):
                    L.text((lx + 24, py + 74 + i * 52), line, 46, WHITE, a, "lm")
            if f > F_FAST:
                L.box((W - 128, 270, W - 24, 320), (20, 20, 20, 210), 1.0, 25)
                L.text((W - 76, 295), "3×", 30, YELLOW, 1.0, "mm")
        elif name == "mapa":
            k = min(int(u * FPS), MAPA_N - 1)
            img = self.shot("mapa", k)
            if u < 0.35:
                img = cv2.addWeighted(self.pov_frame(F_END), 1 - ease(u / 0.35), img, ease(u / 0.35), 0)
            low = self.low["mapa"]
            a = fade(t, t0 + 0.8, t1 + 9, 0.4)  # legenda
            lx, ly = 52, 872
            L.box((lx - 14, ly - 26, lx + 512, ly + 24), (16, 16, 18, 190), a, 22)
            L.d.line((lx + 4, ly, lx + 44, ly), fill=(255, 140, 5, int(255 * a)), width=8)
            L.text((lx + 56, ly), tx["legend"][0], 26, WHITE, a, "lm", "demi")
            x2 = lx + 76 + L.d.textlength(tx["legend"][0], font=font(26, "demi"))
            L.d.ellipse((x2, ly - 9, x2 + 18, ly + 9), fill=(255, 140, 5, int(255 * a)))
            L.text((x2 + 30, ly), tx["legend"][1], 26, WHITE, a, "lm", "demi")
            # tempo por zona
            top = self.stats[0][1]
            y = 950
            a = fade(t, t0 + 1.2, t1 + 9, 0.4)
            L.text((60, y), tx["stats"].format(s=f"{self.span:.1f}".replace(".", tx["dec"])), 34, WHITE, a, "lm",
                   "demi", rise=12)
            for i, (z, sec) in enumerate(self.stats):
                ai = fade(t, t0 + 1.7 + 0.3 * i, t1 + 9, 0.35)
                yy = y + 64 + i * 62
                L.text((60, yy), self.names[z], 32, WHITE, ai, "lm", "medium", rise=10)
                L.text((W - 60, yy), f"{sec:.1f} s".replace(".", tx["dec"]), 32, YELLOW, ai, "rm", "bold", rise=10)
                grow = ease((t - t0 - 1.9 - 0.3 * i) / 0.7)
                L.box((60, yy + 22, W - 60, yy + 30), (255, 255, 255, 40), ai, 4)
                L.box((60, yy + 22, 60 + max(6, (W - 120) * sec / top * grow), yy + 30), (*YELLOW, 255), ai, 4)
        if name == "fim":
            img = self.shot("mapa", MAPA_N - 1)
            img = (img * (1 - 0.4 * ease(u / 1.4))).astype(np.uint8)
            low = self.low["fim"]
            a = fade(t, t0 + 2.2, 99, 0.5)
            y = 1030
            L.text((W // 2, y), tx["author"], 32, WHITE, a, "mm", "demi", rise=12)
            L.text((W // 2, y + 52), LINKEDIN, 32, YELLOW, a, "mm", "demi", rise=12)
            L.text((W // 2, y + 98), LINK, 32, YELLOW, a, "mm", "demi", rise=12)
            for i, c in enumerate(tx["credits"]):
                L.text((W // 2, y + 166 + i * 34), c, 22, GRAY, a, "mm", "medium")

        # manchetes e linha de apoio (em cima), com o degradê
        subs = [
            (tx["hook_sub"], 0.0, S["gancho"][1] - 1.2, 0, 32, LIGHT, "demi"),
            (tx["enters"], 2.9, S["revela"][0], 0.35, 40, WHITE, "bold"),
            (tx["reveal_sub"], S["revela"][0] + 0.6, S["voo"][0], 0.35, 34, WHITE, "demi"),
            (tx["fly_sub"], S["voo"][0] + 0.3, S["visao"][0] - 0.2, 0.35, 32, LIGHT, "demi"),
        ]
        top = self.top_sub if t < S["visao"][0] else self.top
        img = (img * (1 - (top + low)[:, None, None])).astype(np.uint8)
        if inset is not None:
            pic, px, py = inset
            img[py:py + pic.shape[0], px:px + pic.shape[1]] = pic
        heads = [
            (tx["hook"], 0.0, S["gancho"][1], 0),
            (tx["freeze"], S["congela"][0], S["revela"][0], 0.3),
            (tx["reveal"], S["revela"][0], S["voo"][0], 0.3),
            (tx["fly"], S["voo"][0], S["visao"][0], 0.3),
            (tx["pov"], S["visao"][0], S["visao"][0] + 4.4, 0.3),
            (tx["pov2"], S["visao"][0] + 4.4, S["visao"][0] + (F_FAST - FREEZE) / FPS, 0.3),
            (tx["walk"], S["visao"][0] + (F_FAST - FREEZE) / FPS, S["mapa"][0], 0.3),
            (tx["map"], S["mapa"][0], S["fim"][0], 0.3),
            (tx["end1"], S["fim"][0] + 0.2, S["fim"][0] + 1.4, 0.4),
        ]
        for s, a0, a1, fin in heads:
            L.block(W // 2, HEAD_Y, s, 64, 960, fade(t, a0, a1, fin, 0.22))
        a = fade(t, S["fim"][0] + 1.4, 99, 0.5)  # a frase final fica em duas camadas
        if a > 0:
            L.block(W // 2, HEAD_Y - 50, tx["end1"], 46, 960, a, GRAY, "demi", rise=0)
            L.block(W // 2, HEAD_Y + 46, tx["end2"], 58, 980, a)
        for s, a0, a1, fin, size, col, wt in subs:
            L.block(W // 2, SUB_Y, s, size, 960, fade(t, a0, a1, fin, 0.22), col, wt)
        if name != "fim":
            L.text((W // 2, H - 34), LINKEDIN, 24, WHITE, 0.85, "mm", "demi")
        return L.onto(img)

    def timeline(self):
        S = self.seg
        events = {
            "enters": 2.9, "wipe": S["revela"][0] + 0.35, "fly": S["voo"][0], "eyes": S["visao"][0] - 0.25,
            "fast": S["visao"][0] + (F_FAST - FREEZE) / FPS, "map": S["mapa"][0],
            "rows": [S["mapa"][0] + 1.7 + 0.3 * i for i in range(len(self.stats))], "final": S["fim"][0] + 1.5,
            "heads": [0.0, S["congela"][0], S["revela"][0], S["voo"][0], S["visao"][0], S["visao"][0] + 4.4,
                      S["mapa"][0], S["fim"][0] + 0.2],
        }
        return {"total": self.total, "segments": [[n, a, b] for n, (a, b) in S.items()], "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="dados do vídeo (extract_v2.py)")
    ap.add_argument("run_dir", type=Path, help="pasta da rodada, com pov_<idioma>/ e promo_<idioma>/")
    ap.add_argument("--lang", choices=sorted(TEXTS), default="pt")
    ap.add_argument("--stills", default="", help="instantes (s) para gravar como JPG, separados por vírgula")
    args = ap.parse_args()
    p = Promo(args.data_dir, args.run_dir, args.lang)
    print(json.dumps({n: [round(a, 2), round(b, 2)] for n, (a, b) in p.seg.items()}), round(p.total, 2), "s")
    if args.stills:
        out = args.run_dir / f"promo_{args.lang}_stills"
        out.mkdir(parents=True, exist_ok=True)
        for s in args.stills.split(","):
            cv2.imwrite(str(out / f"t{float(s):05.1f}.jpg"), p.frame(float(s)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    (args.run_dir / f"promo_{args.lang}_timeline.json").write_text(json.dumps(p.timeline(), indent=1))
    w = writer(args.run_dir / f"promo_{args.lang}_mudo.mp4", (W, H))
    n = int(round(p.total * FPS))
    for i in range(n):
        w.stdin.write(p.frame(i / FPS).tobytes())
    w.stdin.close()
    w.wait()
    # miniatura: o meio da cortina, metade vídeo e metade 3D
    cv2.imwrite(str(args.run_dir / f"promo_{args.lang}_thumb.jpg"), p.frame(p.seg["revela"][0] + 0.95),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("ok", n, "quadros")


if __name__ == "__main__":
    main()
