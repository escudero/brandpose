"""Reel vertical para o Instagram (1080x1920, 30 fps, em inglês): a etapa 1 como tutorial.

Em ~35 s e no visual cyberpunk do vídeo de demonstração, mostra passo a passo como o
olhar de cada cliente vira atenção por prateleira. O nome do projeto não aparece. Cenas:
  hook    a análise rodando e a promessa do vídeo
  detect  passo 1: volta ao início; uma varredura liga esqueletos e IDs (YOLO11-pose)
  crop    passo 2: tempo congelado; o quadro inteiro espremido em 448x448 contra o
          recorte em volta da cabeça, com zoom nos pixels que o modelo recebe
  gaze    passo 3: o mapa de calor do Gaze-LLE vira um feixe; o tempo volta a correr
  filter  passo 4: olhares grudados na borda ou sobre outra pessoa são descartados
  zones   passo 5: zonas desenhadas à mão somam o tempo; o resto do vídeo, acelerado
  result  o ranking final, numa panorâmica sobre o calor
  next    chamada para a parte 2 (a loja em 3D no Blender)

O texto fica entre y=1000 e y=1480 e longe da borda direita, fora do que a interface
do Instagram cobre. Todos os números saem dos dados (nada é inventado).

Uso:
  python reel.py DATA_DIR OUT_DIR [--part2 RUN_DIR/promo_en] [--stills 0.5,4,9]
Grava OUT_DIR/reel_mudo.mp4, reel_timeline.json e reel_capa.jpg; o reel_audio.py faz a trilha.
"""

import argparse
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_promo as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaze_sim_v2 import crop_window  # noqa: E402

W, H, FPS = 1080, 1920, 30
SW, SH = R.W, R.H  # o vídeo de origem
ZFULL = H / SH  # zoom em que a altura do vídeo de origem preenche o quadro vertical
Z0 = W / SW  # zoom em que a largura inteira cabe (uma faixa 16:9)
BAND_Y = 700  # centro da faixa 16:9 quando a câmera se afasta

# Cores: RGB para o Pillow; bgr() converte para o OpenCV
WHITE, SOFT, DIM = (255, 255, 255), (205, 225, 235), (125, 148, 165)
CYAN, AMBER, MAGENTA, RED = R.RGB_CYAN, R.RGB_AMBER, (255, 60, 200), (255, 64, 88)
INK = (6, 14, 22)  # texto escuro sobre as etiquetas neon
NB = " "  # espaço que não quebra a linha ("16 s")

HERO, CROP_F = 6, 135  # o cliente #05 (camisa branca), congelado em 4,5 s olhando os doces
FILTER_F = 210  # 7,0 s: três olhares que um pipeline ingênuo contaria como "óculos"
HOOK_SRC, DETECT_SRC = 2.6, 0.8  # início do trecho do gancho e do passo 1 (s)
GAZE_LOCK, GAZE_BEAM, GAZE_RUN = 0.75, 1.05, 2.0  # passo 3: a mira trava, o feixe sai, o tempo volta (s)
X_OFF, X_ONP, VANISH, COUNTER = (0.7,), (1.7, 1.85), 2.8, 3.0  # passo 4: os descartes e o placar (s)
ZONES_FF = 0.9  # no passo 5, o avanço rápido começa aqui (s)
P, PY, PX = 440, 470, (70, 570)  # painéis do passo 2: lado, topo e esquerda de cada um

# os cortes caem nos compassos da trilha (120 bpm, compassos de 2 s a partir de 3 s)
SCENES = [("hook", 3.0), ("detect", 4.0), ("crop", 4.0), ("gaze", 4.0), ("filter", 6.0), ("zones", 4.0),
          ("result", 6.0), ("next", 4.0)]
STEPS = {"detect": 1, "crop": 2, "gaze": 3, "filter": 4, "zones": 5}

TX = {
    "hook_tag": "CCTV  //  CONVENIENCE STORE",
    "hook": [("I taught a security camera to see", WHITE), ("where shoppers look", CYAN)],
    "hook_sub": "5 steps  ·  open-source AI  ·  here's how",
    "steps": ["DETECT", "CROP", "GAZE", "FILTER", "ZONES"],
    "chip": "STEP {i}/5  ·  {name}",
    "detect": ("FIND EVERY PERSON", "YOLO11-pose: 17 body points and an ID that follows each shopper."),
    "crop": ("CROP AROUND EACH HEAD", "The model only sees 448×448 px. Cropped, the head is {k}× bigger."),
    "model_sees": "WHAT THE MODEL SEES",
    "panels": ("WHOLE FRAME", "CROP", "HEAD {w}×{h} PX", "CROP = 10× THE HEAD"),
    "gaze": ("PREDICT WHERE THEY LOOK", "Gaze-LLE returns a heatmap. Its peak is the gaze, 10× per second."),
    "heatmap": "GAZE-LLE HEATMAP", "peak": "PEAK",
    "filter": ("THROW OUT FAKE GAZES", ["Stuck on the frame edge = looking off-screen.",
                                        "Landing on another person = not a shelf.",
                                        f"Skip this and sunglasses get {{bad}}{NB}s instead of {{good}}{NB}s."]),
    "off": "OFF-SCREEN", "onp": "ON A PERSON",
    "counter": ("SUNGLASSES ZONE", "NO FILTER", "FILTERED"),
    "zones": ("ADD UP THE SECONDS", "Shelf zones drawn by hand. Every valid gaze adds heat."),
    "result_tag": "RESULT  ·  {s} S OF FOOTAGE",
    "result": "{zone} WINS.", "result_close": "{zone} WINS. BARELY.",
    "result_stats": "{n} shoppers  ·  {g} gazes  ·  {s} s of attention",
    "next_tag": "PART 2",
    "next": ("I REBUILT THIS STORE IN 3D", "…and stepped into a shopper's eyes. Follow for part 2."),
    "credit": "Footage: ACTi E43A public demo  ·  © ACTi Corporation",
    "frozen": "TIME FROZEN", "rewind": "◀◀ REWIND", "raw": "RAW FOOTAGE", "ff": "▶▶ {s}×",
}
GLYPHS = "01<>/\\|#$%&*+=?XZ"


# ---------------------------------------------------------------- utilidades

@lru_cache(maxsize=None)
def font(kind, size):
    path, index = {
        "head": ("Avenir Next Condensed.ttc", 8), "demi": ("Avenir Next.ttc", 2),
        "mono": ("Menlo.ttc", 0), "monob": ("Menlo.ttc", 1), "num": ("Supplemental/DIN Condensed Bold.ttf", 0),
    }[kind]
    return ImageFont.truetype(f"{R.FONTS}/{path}", size, index=index)


def bgr(c, k=1.0):
    return tuple(int(v * k) for v in c[::-1])


def ease(u):
    return R.ease_in_out(u)


def lerp(a, b, k):
    return a + (b - a) * k


def glow(img, sigma=5.0, scale=4):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def scramble(s, u, seed):
    """Texto "decodificando": depois da fração u, os caracteres ainda são sinais aleatórios."""
    if u >= 1:
        return s
    rng = np.random.default_rng(seed)
    k = int(len(s) * max(u, 0.0))
    return s[:k] + "".join(c if c in " " + NB else GLYPHS[rng.integers(len(GLYPHS))] for c in s[k:])


def ip(p):
    return int(round(p[0])), int(round(p[1]))


def head_center(p):
    x1, y1, x2, y2 = p["head"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def paste(img, pic, rect):
    """Cola pic (redimensionada) no retângulo rect=(x0, y0, x1, y1) do quadro, recortando nas bordas."""
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    pic = cv2.resize(pic, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32)
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if bx > ax and by > ay:
        img[ay:by, ax:bx] = pic[ay - y0:by - y0, ax - x0:bx - x0]


class Layer:
    """Textos e caixas (RGBA do Pillow) por cima do quadro, com sombra e brilho neon opcional."""

    def __init__(self):
        self.im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.gl = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # o que ganha brilho
        self.g = ImageDraw.Draw(self.gl)

    def text(self, xy, s, f, fill=WHITE, a=1.0, anchor="la", track=0, glow=0.0):
        """Escreve s e devolve a largura. track: espaço extra entre letras (rótulos em mono)."""
        if a <= 0.01 or not s:
            return 0
        x, y = xy
        if track:
            total = sum(self.d.textlength(c, font=f) + track for c in s) - track
        else:
            total = self.d.textlength(s, font=f)
        x -= {"m": total / 2, "r": total}.get(anchor[0], 0)
        targets = [(self.d, int(255 * a))]
        if glow > 0:
            targets.append((self.g, int(min(255, 255 * a * glow))))
        for d, alpha in targets:
            if track:
                xx = x
                for c in s:
                    d.text((xx, y), c, font=f, fill=(*fill, alpha), anchor="l" + anchor[1])
                    xx += d.textlength(c, font=f) + track
            else:
                d.text((x, y), s, font=f, fill=(*fill, alpha), anchor="l" + anchor[1])
        return total

    def rich(self, x, y, segs, f, width, line_h, a=1.0, glow=0.0, rise=0.0):
        """Parágrafo com trechos coloridos, quebrado em width; y é a linha de base da primeira linha.
        Devolve a linha de base da última."""
        sp = self.d.textlength(" ", font=f)
        lines = self.wrap(segs, f, width)
        while len(lines) > 1 and width > 300 and len(self.wrap(segs, f, width - 10)) == len(lines):
            width -= 10  # linhas equilibradas, sem uma palavra sozinha na última
            lines = self.wrap(segs, f, width)
        y0 = y + (1 - a) * rise
        for i, line in enumerate(lines):
            xx = x
            for w, col, wl in line:
                self.text((xx, y0 + i * line_h), w, f, col, a, "ls", glow=glow)
                xx += wl + sp
        return y + (len(lines) - 1) * line_h

    def wrap(self, segs, f, width):
        sp = self.d.textlength(" ", font=f)
        lines, cur, cw = [], [], 0.0
        for s, col in segs:
            for w in s.split(" "):  # o espaço que não quebra (NB) fica dentro da palavra
                if not w:
                    continue
                wl = self.d.textlength(w, font=f)
                if cur and cw + sp + wl > width:
                    lines.append(cur)
                    cur, cw = [], 0.0
                cw += (sp if cur else 0) + wl
                cur.append((w, col, wl))
        return lines + [cur]

    def box(self, xy, fill=None, outline=None, width=2, r=8, a=1.0, glow=0.0):
        x0, y0, x1, y1 = xy
        if a <= 0 or x1 <= x0 or y1 <= y0:
            return

        def rgba(c, k):
            return None if c is None else (*c[:3], int((c[3] if len(c) == 4 else 255) * k))

        self.d.rounded_rectangle(xy, r, fill=rgba(fill, a), outline=rgba(outline, a), width=width)
        if glow > 0:
            self.g.rounded_rectangle(xy, r, fill=rgba(fill, a * glow), outline=rgba(outline, a * glow), width=width)

    def onto(self, img):
        a = np.asarray(self.im)
        al = a[..., 3]
        out = img.astype(np.float32)
        if al.any():  # sombra suave para ler sobre a imagem
            sh = cv2.GaussianBlur(al, (0, 0), 5).astype(np.float32) / 255 * 0.6
            out *= 1 - sh[..., None]
        g = np.asarray(self.gl)
        if g[..., 3].any():
            gs = cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
            gs = cv2.resize(cv2.GaussianBlur(gs, (0, 0), 3.5), (W, H), interpolation=cv2.INTER_LINEAR)
            out += gs[..., 2::-1] * (gs[..., 3:4] / 255) * 1.8
        k = al[..., None].astype(np.float32) / 255
        out = out * (1 - k) + a[..., 2::-1].astype(np.float32) * k
        return np.clip(out, 0, 255).astype(np.uint8)


class View:
    """Câmera virtual sobre o vídeo de origem: centro (cx, cy) em pixels dele e zoom z.
    Com z >= ZFULL a imagem preenche o quadro; abaixo disso, vira uma faixa (até a 16:9
    inteira, em Z0) centrada perto de BAND_Y, com a própria imagem borrada em volta."""

    def __init__(self, cx, cy, z=ZFULL):
        z = max(z, Z0)
        hw = W / 2 / z
        self.cx = min(max(cx, hw), SW - hw)
        if z >= ZFULL:
            hh = H / 2 / z
            self.cy = min(max(cy, hh), SH - hh)
            ty = H / 2 - z * self.cy
            self.band = None
        else:
            self.cy = SH / 2
            ty = lerp(H / 2, BAND_Y, (ZFULL - z) / (ZFULL - Z0)) - z * SH / 2
            self.band = (ty, ty + z * SH)
        self.z, self.ty = z, ty
        self.M = np.float32([[z, 0, W / 2 - z * self.cx], [0, z, ty]])

    def pt(self, x, y):
        return self.z * (x - self.cx) + W / 2, self.z * y + self.ty

    def warp(self, img):
        out = cv2.warpAffine(img, self.M, (W, H), flags=cv2.INTER_LINEAR)
        if self.band is not None:
            y0, y1 = int(math.ceil(self.band[0])), int(self.band[1])
            amb = ambient(img)
            out[:max(y0, 0)] = amb[:max(y0, 0)]
            out[max(y1, 0):] = amb[max(y1, 0):]
        return out


def ambient(img):
    """Fundo em volta da faixa: a própria imagem cobrindo o quadro, borrada e escura."""
    small = cv2.resize(img, (SW // 8, SH // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    cw = int(SH // 8 * W / H)
    x0 = (small.shape[1] - cw) // 2
    return cv2.resize(small[:, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR) * 0.4


# ---------------------------------------------------------------- desenho (no quadro vertical)

def skeleton(ov, view, p, staff, a=1.0):
    kp, kc = np.array(p["kp"]), np.array(p["kc"])
    col = bgr((130, 130, 130) if staff else (165, 240, 255), a)
    pts = [view.pt(*k) for k in kp]
    th = 3 if view.z > 1 else 2
    for i, j in R.COCO_EDGES:
        if kc[i] > 0.4 and kc[j] > 0.4:
            cv2.line(ov, ip(pts[i]), ip(pts[j]), col, th, cv2.LINE_AA)
    for i in range(5, 17):
        if kc[i] > 0.4:
            cv2.circle(ov, ip(pts[i]), th + 2, col, -1, cv2.LINE_AA)
    if p["head"]:
        r = int(max(8, (p["head"][2] - p["head"][0]) * 0.42 * view.z))
        cv2.circle(ov, ip(view.pt(*head_center(p))), r, col, 2, cv2.LINE_AA)


def tag(L, view, p, label, staff, a=1.0):
    x1, y1, x2, y2 = p["head"]
    cx, top = view.pt((x1 + x2) / 2, y1)
    _, bottom = view.pt(0, y2)
    f = font("monob", 26)
    w = L.d.textlength(label, font=f)
    y = top - 16 if top - 60 > 320 else bottom + 58
    L.box((cx - w / 2 - 10, y - 38, cx + w / 2 + 10, y), fill=(50, 50, 50, 170) if staff else (8, 50, 66, 190),
          outline=(150, 150, 150) if staff else CYAN, width=2, r=5, a=a)
    L.text((cx, y - 19), label, f, (205, 205, 205) if staff else WHITE, a, "mm")


def beam(ov, o, t, a, phase, col=AMBER, core=(255, 235, 175), ring=True):
    """Feixe do olhar, da cabeça (o) ao alvo (t), com a mira no alvo."""
    o, t = np.array(o, np.float32), np.array(t, np.float32)
    d = t - o
    L = float(np.linalg.norm(d))
    if L < 12 or a <= 0:
        return
    n = np.array([-d[1], d[0]], np.float32) / L
    w0, w1 = 4.0, float(np.clip(0.11 * L, 18, 64))
    tmp = np.zeros_like(ov)
    for i in range(12):
        a0, a1 = i / 12, (i + 1) / 12
        p0, p1 = o + d * a0, o + d * a1
        wa, wb = w0 + (w1 - w0) * a0, w0 + (w1 - w0) * a1
        k = a * (0.06 + 0.30 * a1 ** 1.3)
        pts = np.array([p0 + n * wa, p1 + n * wb, p1 - n * wb, p0 - n * wa]).astype(np.int32)
        cv2.fillConvexPoly(tmp, pts, bgr(col, k), cv2.LINE_AA)
    cv2.line(tmp, ip(o), ip(t), bgr(core, a * 0.85), 3, cv2.LINE_AA)
    if ring:
        r = int(26 + 6 * math.sin(phase * 2 * math.pi * 1.3))
        cv2.circle(tmp, ip(t), r, bgr(core, a), 3, cv2.LINE_AA)
        cv2.circle(tmp, ip(t), 6, bgr(core, a), -1, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(tmp, (int(t[0] + dx * (r + 6)), int(t[1] + dy * (r + 6))),
                     (int(t[0] + dx * (r + 18)), int(t[1] + dy * (r + 18))), bgr(core, a), 3, cv2.LINE_AA)
    cv2.add(ov, tmp, ov)


def label(L, xy, s, a, col=AMBER, fg=(255, 228, 175), size=30):
    """Rótulo ao lado do alvo; vira para a esquerda perto da borda."""
    x, y = xy
    f = font("demi", size)
    w = L.d.textlength(s, font=f)
    bx = x + 38 if x + 38 + w + 24 < W - 30 else x - 38 - w - 24
    by = min(max(y - 70, 330), H - 80)
    L.box((bx, by, bx + w + 24, by + size + 16), fill=(*[int(c * 0.16) for c in col], 200), outline=col, r=7, a=a)
    L.text((bx + 12, by + (size + 16) / 2), s, f, fg, a, "lm")


def x_mark(ov, c, a, size=22):
    for s in (1, -1):
        cv2.line(ov, (int(c[0] - size), int(c[1] - s * size)), (int(c[0] + size), int(c[1] + s * size)),
                 bgr(RED, a), 6, cv2.LINE_AA)


def corners(ov, xyxy, col, a, n=28, th=4):
    x0, y0, x1, y1 = (int(v) for v in xyxy)
    for px, py, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(ov, (px, py), (px + dx * n, py), bgr(col, a), th, cv2.LINE_AA)
        cv2.line(ov, (px, py), (px, py + dy * n), bgr(col, a), th, cv2.LINE_AA)


def poly_partial(ov, pts, k, col, th=3):
    """Contorno do polígono desenhado até a fração k do perímetro."""
    pts = [np.array(p, np.float32) for p in pts] + [np.array(pts[0], np.float32)]
    seg = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
    left = k * sum(seg)
    for (a, b), s in zip(zip(pts, pts[1:]), seg):
        if left <= 0:
            break
        e = b if left >= s else a + (b - a) * (left / s)
        cv2.line(ov, ip(a), ip(e), col, th, cv2.LINE_AA)
        left -= s


def cyber(img):
    """Holograma para o 3D da parte 2: fundo azul-marinho e as arestas em ciano."""
    f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 30, 90)
    lum = gray.astype(np.float32)[..., None] / 255
    base = np.array([60, 25, 8], np.float32) * (1 - lum) * 0.8 + \
        np.array([255, 200, 90], np.float32) * lum ** 3 * 0.35 + f * 0.12
    e3 = (e[..., None].astype(np.float32) / 255 * R.CYAN).astype(np.uint8)
    return base + e3 * 0.75 + glow(e3, 3, 2) * 1.0


# ---------------------------------------------------------------- o reel

class Reel:
    def __init__(self, data_dir, part2=None):
        R.TX.update(R.TEXTS["en"])  # nomes das zonas em inglês
        self.d = R.Data(data_dir)
        self.raw = no_rules(data_dir)
        self.S = R.Scene(self.d, self.d.meta["video"])
        self.part2 = Path(part2) if part2 else None
        self.fps = self.d.meta["fps"]
        self.src_end = (self.d.n_frames - 1) / self.fps
        self.seg, t = {}, 0.0
        for name, dur in SCENES:
            self.seg[name] = (t, t + dur)
            t += dur
        self.total = t
        # números do ranking (com e sem as regras)
        self.vals = self.d.cum_zone[-1]
        self.order = [int(i) for i in np.argsort(-self.vals) if self.vals[i] > 0]
        sun = next(i for i, z in enumerate(self.d.zones) if z["short"] == "ÓCULOS")
        self.bad, self.good = float(self.raw.cum_zone[-1][sun]), float(self.vals[sun])
        self._crop_setup()
        self._result_setup()
        # efeitos da imagem no quadro vertical: linhas de varredura, vinheta e degradês do texto
        self.scan = np.where(np.arange(H) % 4 == 0, 0.9, 1.0).astype(np.float32)[:, None, None]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = 0.55 * ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        self.vig = np.clip(1 - 0.32 * r2, 0.45, 1)[..., None]
        y = np.arange(H, dtype=np.float32)
        top = 0.6 * np.clip(1 - y / 400, 0, 1) ** 1.4
        low = 0.92 * np.clip((y - 700) / 900, 0, 1) ** 0.9
        self.scrim = (1 - top - low).clip(0.05, 1)[:, None, None]
        self.ov_mask = (1 - 0.65 * np.clip((y - 990) / 70, 0, 1) - 0.75 * np.clip((330 - y) / 40, 0, 1))[:, None, None]
        self.flagged = None  # passo 4: olhares descartados, calculados uma vez
        self._p2 = {}

    # ---------- preparação

    def _crop_setup(self):
        """Passo 2 e 3: a janela do recorte, as duas entradas do modelo e o mapa do Gaze-LLE."""
        p = self.d.by_frame[CROP_F][HERO]
        self.hero = p
        x1, y1, x2, y2 = p["head"]
        hw, hh = x2 - x1, y2 - y1
        left, top, side = crop_window(p["head"], SW, SH, self.d.meta["crop_scale"], self.d.meta["crop_min"])
        self.win = (left, top, side)
        frame = self.S.src(CROP_F).copy()
        self.still = frame
        # o que vai para o modelo: o quadro inteiro espremido, ou o recorte, em 448x448
        self.inputs = {"full": cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA),
                       "crop": cv2.resize(frame[top:top + side, left:left + side], (448, 448),
                                          interpolation=cv2.INTER_AREA)}
        s_full, s_crop = (448 / SW, 448 / SH), 448 / side
        self.head_px = {"full": (hw * s_full[0], hh * s_full[1]), "crop": (hw * s_crop, hh * s_crop)}
        self.gain = self.head_px["crop"][0] / self.head_px["full"][0]
        # a mesma região em volta da cabeça (3 cabeças de lado), nas coordenadas de cada entrada
        cx, cy = head_center(p)
        r0 = 1.5 * hw
        self.head_in, self.head_rect = {}, {}
        for key, (sx, sy), (ox, oy) in (("full", s_full, (0, 0)), ("crop", (s_crop, s_crop), (left, top))):
            self.head_in[key] = ((cx - ox) * sx, (cy - oy) * sy)
            self.head_rect[key] = ((cx - r0 - ox) * sx, (cy - r0 - oy) * sy, (cx + r0 - ox) * sx, (cy + r0 - oy) * sy)
        # mapa de calor do Gaze-LLE para ele nesse quadro, na resolução do vídeo
        g = next(g for g in self.d.gaze if g["frame"] == CROP_F and g["id"] == HERO)
        hm = self.d.hms[g["hm"]].astype(np.float32)
        big = cv2.resize(hm, (hm.shape[1] * 16, hm.shape[0] * 16), interpolation=cv2.INTER_CUBIC)
        big = np.pad(np.clip(big, 0, None), ((0, SH - big.shape[0]), (0, SW - big.shape[1])))
        v = big / big.max()
        col = (v ** 1.4)[..., None] * np.array(bgr(MAGENTA), np.float32) + \
            (v ** 5)[..., None] * np.array([255, 240, 255], np.float32) * 0.8
        self.hm_img = col + glow(np.clip(col, 0, 255).astype(np.uint8), 8, 4) * 0.8
        # pico (sem a própria cabeça, como no extract_v2.py), refinado pela média em volta
        pick = hm.copy()
        ds = self.d.ds
        pick[int(max(0, cy - hh * 0.75) / ds):int((cy + hh * 0.75) / ds) + 1,
             int(max(0, cx - hw * 0.75) / ds):int((cx + hw * 0.75) / ds) + 1] = 0
        py, px = np.unravel_index(pick.argmax(), pick.shape)
        y0, x0 = max(0, py - 1), max(0, px - 1)
        win = pick[y0:py + 2, x0:px + 2]
        gy, gx = np.mgrid[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
        self.peak = ((gx * win).sum() / win.sum() + 0.5) * ds, ((gy * win).sum() / win.sum() + 0.5) * ds
        self.peak_zone = self.d.zone_for(CROP_F, *self.peak)

    def _result_setup(self):
        plate = self.S.plate
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        base = (0.72 * gray * self.S.tint + 0.28 * plate.astype(np.float32)) * 0.30 + self.S.edges
        v = self.S.heat.at(self.d.n_frames - 1)
        self.final_bg = base + self.S.heat.colorize(v, None, gain=1.15)
        # marcadores do ranking: o centro do calor dentro de cada zona
        self.marks = {}
        hh, ww = v.shape
        gy, gx = np.mgrid[0:hh, 0:ww]
        for zi in self.order[:3]:
            m = cv2.resize(self.d.zone_q[zi], (ww, hh)) * v
            self.marks[zi] = ((gx * m).sum() / m.sum() * 2, (gy * m).sum() / m.sum() * 2)

    # ---------- composição

    def ai(self, f, people=True, heat=True, gain=1.0):
        """O "modo IA" do vídeo de demonstração, no espaço do vídeo de origem (float BGR)."""
        frame = self.S.src(f)
        fr = frame.astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        base = (0.72 * gray * self.S.tint + 0.28 * fr) * 0.36 + self.S.edges
        m = self.S.people_soft(f)
        if people:
            base = base * (1 - m[..., None] * 0.9) + fr * 0.855 * m[..., None]
            rim = cv2.dilate(m, np.ones((7, 7), np.uint8)) - cv2.erode(m, np.ones((5, 5), np.uint8))
            rim_u8 = (np.clip(rim, 0, 1)[..., None] * R.CYAN).astype(np.uint8)
            base += rim_u8.astype(np.float32) * 0.55 + R.glow(rim_u8, 4, 4) * 0.9
        if heat:
            base += self.S.heat.colorize(self.S.heat.at(f), m if people else None, gain)
        return base

    def people(self, ov, L, view, f, t, beams=True, labels=True, tags=True, a=1.0, below=None):
        """Esqueletos, IDs e feixes do olhar (com as regras) das pessoas no quadro f.
        below: só desenha quem tem a cabeça acima dessa linha (a varredura do passo 1)."""
        d = self.d
        for tid, p in d.by_frame[min(f, d.n_frames - 1)].items():
            staff = tid in d.staff
            if p["head"] is None:
                continue
            hc = view.pt(*head_center(p))
            if below is not None and hc[1] > below:
                continue
            skeleton(ov, view, p, staff, a)
            if tags:
                tag(L, view, p, R.TX["staff"] if staff else f"#{d.number.get(d.person.get(tid, tid), 0):02d}", staff, a)
            if staff or not beams:
                continue
            g = d.gaze_at(tid, f)
            if g is None:
                continue
            gx, gy, io = g
            k = R.clamp01((io - 0.45) / 0.15) * a
            if k <= 0:
                continue
            tg = view.pt(gx, gy)
            beam(ov, hc, tg, k, t + tid * 0.37)
            z = d.zone_for(f, gx, gy)
            if labels and z >= 0:
                dw = d.dwell_at(tid, f)
                secs = dw[1] if dw and dw[0] == z else 0.0
                name = R.zone_text(d.zones[z])
                label(L, tg, f"{name}  ·  {secs:.1f} s" if secs >= 0.3 else name, k)

    def chip(self, L, x, y, s, u, fill=CYAN, fg=INK):
        """Etiqueta neon do passo, decodificando na entrada."""
        f = font("monob", 28)
        w = L.d.textlength(s, font=f) + 3 * (len(s) - 1)
        a = R.clamp01(u / 0.12)
        L.box((x, y, x + w + 30, y + 48), fill=fill, r=5, a=a, glow=0.5)
        L.text((x + 15, y + 25), scramble(s, u / 0.35, int(u * FPS) * 7919 + len(s)), f, fg, a, "lm", track=3)

    def block(self, L, u, chip, head, sub, y=1036, fill=CYAN):
        """Etiqueta, manchete e linha de apoio, embaixo (acima da legenda do Instagram).
        sub: um texto, ou uma lista de (texto, início, fim) que se revezam."""
        self.chip(L, 64, y, chip, u, fill)
        a = R.clamp01((u - 0.06) / 0.2)
        yb = L.rich(64, y + 150, [(head, WHITE)], font("head", 96), 900, 98, a, glow=0.28, rise=26)
        subs = sub if isinstance(sub, list) else [(sub, 0.25, 99)]
        for s, a0, a1 in subs:
            k = R.fade(u, a0, a1, 0.22, 0.12)
            if k > 0:
                L.rich(64, yb + 68, [(s, SOFT)], font("demi", 40), 860, 52, k, rise=12)
        return yb

    def hud(self, L, ov, t, src_t, name, prog, state=None):
        """Linha de status (gravando, câmera, tempo do vídeo de origem e o estado) e o progresso dos passos."""
        y = 238
        if int(t * 2.5) % 2 == 0:
            L.d.ellipse((67, y - 9, 85, y + 9), fill=(255, 70, 70, 255))
        L.text((98, y), "REC", font("monob", 24), WHITE, 0.9, "lm", track=2)
        L.text((178, y), "CAM 01", font("mono", 24), SOFT, 0.85, "lm", track=2)
        if src_t is not None:
            L.text((316, y), f"T+{src_t:05.2f}", font("mono", 24), SOFT, 0.85, "lm", track=1)
        if state:
            s, col = state
            f = font("monob", 26)
            w = L.d.textlength(s, font=f) + 2 * (len(s) - 1)
            L.box((W - 64 - w - 28, y - 24, W - 64, y + 24), fill=col, r=5, glow=0.6)
            L.text((W - 64 - 14, y + 1), s, f, INK, 1.0, "rm", track=2)
        x0, x1, yb, gap = 64, W - 64, 278, 12
        sw = (x1 - x0 - 4 * gap) / 5
        cur = STEPS.get(name, 0 if name == "hook" else 6)
        for i in range(5):
            xa = x0 + i * (sw + gap)
            L.box((xa, yb, xa + sw, yb + 6), fill=(255, 255, 255, 46), r=3)
            k = 1.0 if i + 1 < cur else (prog if i + 1 == cur else 0.0)
            if k > 0:
                L.box((xa, yb, xa + max(6, sw * k), yb + 6), fill=CYAN, r=3, glow=0.8)
            on = i + 1 == cur
            L.text((xa, yb + 28), TX["steps"][i], font("monob" if on else "mono", 20), CYAN if on else DIM,
                   1.0 if on else 0.75, "lm", track=2, glow=0.4 if on else 0)

    def state(self, name, u):
        """O selo de estado no canto da linha de status."""
        if name == "detect":
            return (TX["rewind"], AMBER) if u < 0.3 else ((TX["raw"], WHITE) if u < 1.2 else None)
        if name in ("crop", "filter") or (name == "gaze" and u < GAZE_RUN) or (name == "zones" and u < ZONES_FF):
            return TX["frozen"], CYAN
        if name == "zones":
            dur = self.seg["zones"][1] - self.seg["zones"][0]
            speed = (self.src_end - FILTER_F / self.fps) / (dur - ZONES_FF)
            return TX["ff"].format(s=f"{speed:.0f}"), AMBER
        return None

    def finish(self, img, ov, L, t, glitch=0.0):
        img = img * self.scan * self.vig * self.scrim
        ov = (ov * self.ov_mask).astype(np.uint8)
        img = img + ov + glow(ov, 5, 4) * 1.1
        out = L.onto(img)
        return self.glitch(out, glitch, int(t * FPS))

    def glitch(self, img, amount, seed):
        """Faixas deslocadas e canais separados nos cortes."""
        if amount <= 0.02:
            return img
        rng = np.random.default_rng(seed)
        out = img.copy()
        dx = int(16 * amount)
        out[..., 2] = np.roll(img[..., 2], dx, axis=1)
        out[..., 0] = np.roll(img[..., 0], -dx, axis=1)
        for _ in range(int(2 + 10 * amount)):
            y0, h = int(rng.integers(0, H - 80)), int(rng.integers(8, 90))
            out[y0:y0 + h] = np.roll(out[y0:y0 + h], int(rng.integers(-90, 90) * amount), axis=1)
        if amount > 0.5:  # uma faixa clara rápida
            y0 = int(rng.integers(0, H - 20))
            out[y0:y0 + 6] = np.clip(out[y0:y0 + 6].astype(np.int16) + 90, 0, 255).astype(np.uint8)
        return out

    def cut_glitch(self, t):
        g = 0.0
        for name, (a, b) in self.seg.items():
            if a > 0:
                g = max(g, math.exp(-((t - a) / 0.07) ** 2))
        c0, f0 = self.seg["crop"][0], self.seg["filter"][0]
        hits = [(c0 + 1.2, 0.35), (f0 + VANISH, 0.6), (f0 + COUNTER + 0.55, 0.5)]
        hits += [(f0 + x, 0.45) for x in X_OFF + X_ONP]
        for te, amt in hits:
            g = max(g, amt * math.exp(-((t - te) / 0.05) ** 2))
        return g

    def frame(self, t):
        name = next((n for n, (a, b) in self.seg.items() if a <= t < b), SCENES[-1][0])
        t0, t1 = self.seg[name]
        ov = np.zeros((H, W, 3), np.uint8)
        L = Layer()
        img, src_t = getattr(self, "s_" + name)(t - t0, t1 - t0, ov, L, t)
        self.hud(L, ov, t, src_t, name, R.clamp01((t - t0) / (t1 - t0)), self.state(name, t - t0))
        return self.finish(img, ov, L, t, self.cut_glitch(t))

    # ---------- cenas

    def s_hook(self, u, dur, ov, L, t):
        src = HOOK_SRC + u
        f = int(round(src * self.fps))
        k = ease(u / dur)
        view = View(650 - 40 * k, 540, ZFULL * (1.04 + 0.08 * k))
        img = view.warp(self.ai(f))
        self.people(ov, L, view, f, t)
        L.text((64, 1060), scramble(TX["hook_tag"], u / 0.45, int(u * FPS)), font("monob", 26), CYAN, 1.0, "lm",
               track=3, glow=0.5)
        yb = L.rich(64, 1176, TX["hook"], font("head", 104), 920, 104, 1.0, glow=0.3)
        L.text((66, yb + 74), TX["hook_sub"], font("demi", 38), SOFT, R.clamp01((u - 0.5) / 0.3), "ls")
        return img, src

    def s_detect(self, u, dur, ov, L, t):
        rw = 0.3  # rebobina até o início do trecho
        if u < rw:
            src = HOOK_SRC + 3.0 - (HOOK_SRC + 3.0 - DETECT_SRC) * ease(u / rw)
        else:
            src = DETECT_SRC + (u - rw)
        f = int(round(src * self.fps))
        k = ease(u / dur)
        view = View(600 + 50 * k, 540, ZFULL * (1.0 + 0.05 * k))
        if u < rw:  # a análise do gancho voltando, com linhas de fita
            img = view.warp(self.ai(f))
            self.people(ov, L, view, f, t, labels=False, tags=False)
            rng = np.random.default_rng(f)
            for _ in range(6):
                y = int(rng.integers(300, H - 300))
                img[y:y + 4] += 70
            return img, src
        raw = view.warp(self.S.src(f).astype(np.float32)) * 0.85
        ai = view.warp(self.ai(f, heat=False))
        ys = H * ease((u - 0.65) / 1.0)  # a varredura desce
        a = np.clip((ys - np.arange(H, dtype=np.float32)) / 40.0, 0, 1)[:, None, None]
        img = ai * a + raw * (1 - a)
        self.people(ov, L, view, f, t, beams=False, below=ys - 10)
        ov[int(max(ys, 0)):] = 0  # o esqueleto só aparece acima da linha
        if 0 < ys < H:
            line = np.zeros((H, W, 3), np.uint8)
            cv2.line(line, (0, int(ys)), (W, int(ys)), (255, 245, 170), 4)
            img += line + glow(line, 6, 4) * 2.4
            L.text((W - 64, ys - 30), "SCANNING  ·  POSE  ·  ID", font("monob", 22), CYAN, 1.0, "rm", track=2)
        self.block(L, u - rw, TX["chip"].format(i=1, name=TX["steps"][0]), *TX["detect"])
        return img, src

    def s_crop(self, u, dur, ov, L, t):
        f = CROP_F
        left, top, side = self.win
        k = ease(u / 0.6)  # a câmera se afasta até o quadro inteiro
        view = View(lerp(650, SW / 2, k), 540, math.exp(lerp(math.log(ZFULL * 1.05), math.log(Z0), k)))
        img = view.warp(self.ai(f, heat=False))
        img += 255 * max(0.0, 1 - u / 0.18) * 0.5  # o flash do congelamento
        s = ease((u - 1.2) / 0.5)  # a faixa se espreme no painel e o recorte sobe para o outro
        if view.band is not None and s < 1:
            for y in view.band:
                cv2.line(ov, (0, int(y)), (W, int(y)), bgr(CYAN, 0.6 * (1 - s)), 2, cv2.LINE_AA)
        if s <= 0:
            self.people(ov, L, view, f, t, beams=False, tags=False, a=0.4)
            x1, y1, x2, y2 = self.hero["head"]
            g = ease((u - 0.6) / 0.45)  # a janela cresce da cabeça até o recorte
            box = (lerp(x1, left, g), lerp(y1, top, g), lerp(x2, left + side, g), lerp(y2, top + side, g))
            a0, b0 = view.pt(box[0], box[1]), view.pt(box[2], box[3])
            a_w = R.clamp01((u - 0.55) / 0.15)
            cv2.rectangle(ov, ip(a0), ip(b0), bgr(CYAN, 0.7 * a_w), 2, cv2.LINE_AA)
            corners(ov, (*a0, *b0), CYAN, a_w, 24, 4)
            L.text((a0[0], a0[1] - 22), TX["panels"][3], font("monob", 22), CYAN, R.clamp01((u - 0.9) / 0.2),
                   "lm", track=2, glow=0.5)
        else:
            img *= 1 - 0.5 * s
            band = (0, view.band[0], W, view.band[1])
            win = (*view.pt(left, top), *view.pt(left + side, top + side))
            if s < 1:  # o quadro inteiro se espreme; o recorte cresce
                for key, r0, pic in (("full", band, self.still), ("crop", win, self.inputs["crop"])):
                    paste(img, pic, [lerp(a, b, s) for a, b in zip(r0, self.panel_rect(key))])
            else:
                self.panels(img, ov, L, u)
        L.text((W / 2, 400), TX["model_sees"], font("monob", 30), WHITE, R.clamp01((u - 1.6) / 0.3), "mm",
               track=4, glow=0.4)
        self.block(L, u, TX["chip"].format(i=2, name=TX["steps"][1]), TX["crop"][0],
                   TX["crop"][1].format(k=f"{self.gain:.1f}"))
        return img, f / self.fps

    def panel_rect(self, key):
        x0 = PX[0] if key == "full" else PX[1]
        return x0, PY, x0 + P, PY + P

    def panels(self, img, ov, L, u):
        """As duas entradas do modelo lado a lado; depois, zoom nos pixels da cabeça ("enhance")."""
        e = ease((u - 2.3) / 0.6)
        for i, key in enumerate(("full", "crop")):
            x0, y0, x1, y1 = self.panel_rect(key)
            # retângulo mostrado (em pixels da entrada): da entrada inteira até a região da cabeça
            hx0, hy0, hx1, hy1 = self.head_rect[key]
            cw = math.exp(lerp(math.log(448), math.log(hx1 - hx0), e))
            ch = math.exp(lerp(math.log(448), math.log(hy1 - hy0), e))
            cx, cy = lerp(224, (hx0 + hx1) / 2, e), lerp(224, (hy0 + hy1) / 2, e)
            kx, ky = P / cw, P / ch
            M = np.float32([[kx, 0, -(cx - cw / 2) * kx], [0, ky, -(cy - ch / 2) * ky]])
            pic = cv2.warpAffine(self.inputs[key], M, (P, P), flags=cv2.INTER_NEAREST)
            paste(img, pic, (x0, y0, x1, y1))
            col = RED if key == "full" else CYAN
            cv2.rectangle(ov, (x0 - 2, int(y0) - 2), (x1 + 1, int(y1) + 1), bgr(CYAN, 0.5), 2, cv2.LINE_AA)
            corners(ov, (x0 - 8, y0 - 8, x1 + 7, y1 + 7), CYAN, 1.0, 30, 4)
            L.text((x0, y0 - 30), f"{TX['panels'][i]} → 448 PX", font("monob", 24), CYAN, 1.0, "lm", track=2,
                   glow=0.4)
            if e < 1:  # a cabeça marcada na entrada inteira
                hx, hy = self.head_in[key]
                px, py = x0 + (hx - (cx - cw / 2)) * kx, y0 + (hy - (cy - ch / 2)) * ky
                r = int(self.head_px[key][1] / 2 * ky + 12)
                cv2.circle(ov, ip((px, py)), r, bgr(col, 1 - e), 3, cv2.LINE_AA)
            w, h = self.head_px[key]
            a = R.clamp01((u - 2.6) / 0.3)
            L.box((x0 + 14, y1 - 60, x0 + 14 + 262, y1 - 14), fill=(4, 10, 16, 210), outline=col, r=5, a=a)
            L.text((x0 + 28, y1 - 37), TX["panels"][2].format(w=round(w), h=round(h)), font("monob", 24), col, a,
                   "lm", track=1)

    def s_gaze(self, u, dur, ov, L, t):
        run = u >= GAZE_RUN
        f = CROP_F + (int((u - GAZE_RUN) * self.fps) if run else 0)
        hc0 = head_center(self.hero)
        k = ease((u - GAZE_RUN) / (dur - GAZE_RUN))
        view = View(lerp((hc0[0] + self.peak[0]) / 2, 640, k), lerp(470, 540, k), ZFULL * lerp(1.15, 1.0, k))
        img = view.warp(self.ai(f, heat=run))
        hc, pk = view.pt(*hc0), view.pt(*self.peak)
        a_h = R.fade(u, 0.1, GAZE_RUN + 0.3, 0.35, 0.4) * (0.8 + 0.2 * math.sin(u * 2 * math.pi * 1.6))
        if a_h > 0:  # o mapa de calor que o Gaze-LLE devolveu para ele
            img += view.warp(self.hm_img) * a_h * (1 - 0.6 * ease((u - GAZE_BEAM - 0.2) / 0.6))
            a_l = R.fade(u, 0.2, GAZE_LOCK + 0.05, 0.15, 0.12)
            L.box((pk[0] - 190, pk[1] + 96, pk[0] + 190, pk[1] + 146), fill=(40, 6, 30, 210), outline=MAGENTA,
                  r=6, a=a_l)
            L.text((pk[0], pk[1] + 122), TX["heatmap"], font("monob", 26), (255, 190, 235), a_l, "mm", track=2)
        if not run:
            self.people(ov, L, view, f, t, beams=False, tags=False, a=0.35)
            a_p = R.clamp01((u - GAZE_LOCK) / 0.2)
            if a_p > 0:  # a mira trava no pico
                r = int(lerp(90, 30, ease((u - GAZE_LOCK) / 0.3)))
                cv2.circle(ov, ip(pk), r, bgr(WHITE, a_p), 3, cv2.LINE_AA)
                coords = f"{TX['peak']}  x {self.peak[0]:.0f}  y {self.peak[1]:.0f}"
                L.text((pk[0] - 150, pk[1] + 122), scramble(coords, (u - GAZE_LOCK) / 0.35, int(u * FPS)),
                       font("monob", 24), WHITE, a_p, "lm", track=1)
            b = ease((u - GAZE_BEAM) / 0.3)
            if b > 0:  # o feixe sai da cabeça até o pico
                beam(ov, hc, (lerp(hc[0], pk[0], b), lerp(hc[1], pk[1], b)), 1.0, t, ring=b >= 1)
            if u > GAZE_BEAM + 0.3 and self.peak_zone >= 0:
                label(L, pk, R.zone_text(self.d.zones[self.peak_zone]), R.clamp01((u - GAZE_BEAM - 0.3) / 0.2))
        else:
            self.people(ov, L, view, f, t)
        self.block(L, u, TX["chip"].format(i=3, name=TX["steps"][2]), *TX["gaze"])
        return img, f / self.fps

    def filtered(self):
        """Passo 4: no quadro FILTER_F, o olhar de cada cliente sem as regras e o motivo do descarte."""
        if self.flagged is None:
            f = FILTER_F
            out = []
            for tid, p in self.d.by_frame[f].items():
                if tid in self.d.staff or p["head"] is None:
                    continue
                g = self.raw.gaze_at(tid, f)
                if g is None or g[2] < 0.5:
                    continue
                gx, gy = g[0], g[1]
                ruled = self.d.gaze_at(tid, f)  # com as regras, o pico na borda zera a confiança
                edge = ruled is not None and ruled[2] < 0.5
                why = "off" if edge else ("onp" if self.d.on_person(f, gx, gy) else None)
                z = self.raw.zone_at(gx, gy)
                out.append({"id": tid, "head": head_center(p), "tgt": (gx, gy), "why": why,
                            "zone": R.zone_text(self.d.zones[z]) if z >= 0 else ""})
            # primeiro o que sai pela borda, depois os que caem em outra pessoa
            out.sort(key=lambda o: (o["why"] != "off", o["head"][0]))
            self.flagged = out
        return self.flagged

    def s_filter(self, u, dur, ov, L, t):
        f = FILTER_F
        view = View(240, 427, ZFULL * (1.26 + 0.03 * ease(u / dur)))
        img = view.warp(self.ai(f, heat=False))
        dim = 0.72 * ease((u - COUNTER) / 0.4)
        img *= 1 - dim
        self.people(ov, L, view, f, t, beams=False, tags=u < COUNTER, a=1 - dim)
        flags = {"off": list(X_OFF), "onp": list(X_ONP)}
        for o in self.filtered():
            when = flags[o["why"]].pop(0) if o["why"] and flags[o["why"]] else None
            hit = when is not None and u >= when
            a = R.clamp01(u / 0.25) * (1 - ease((u - VANISH) / 0.3) if hit else 1 - dim)
            if a <= 0:
                continue
            hc, tg = view.pt(*o["head"]), view.pt(*o["tgt"])
            tg = (min(max(tg[0], 18), W - 18), tg[1])
            flick = 0.4 if hit and u < when + 0.12 and int(u * FPS) % 2 else 1.0
            beam(ov, hc, tg, a * flick, t + o["id"] * 0.37, col=RED if hit else AMBER,
                 core=(255, 170, 180) if hit else (255, 235, 175), ring=not hit)
            if hit:
                x_mark(ov, tg, a)
                label(L, tg, TX[o["why"]], a, col=RED, fg=(255, 210, 215))
            elif o["zone"]:
                label(L, tg, o["zone"] + "?", a)
        if u >= COUNTER:
            self.counter(L, ov, u - COUNTER)
        subs = TX["filter"][1]
        self.block(L, u, TX["chip"].format(i=4, name=TX["steps"][3]), TX["filter"][0],
                   [(subs[0], 0.35, X_ONP[0]), (subs[1], X_ONP[0], COUNTER),
                    (subs[2].format(bad=f"{self.bad:.0f}", good=f"{self.good:.0f}"), COUNTER, 99)])
        return img, f / self.fps

    def counter(self, L, ov, u):
        """O mesmo "óculos" sem e com as regras."""
        a = R.clamp01(u / 0.25)
        L.text((W / 2, 470), TX["counter"][0], font("monob", 34), WHITE, a, "mm", track=6, glow=0.4)
        cx0, cx1, y = 290, 790, 700
        L.text((cx0, 560), TX["counter"][1], font("monob", 26), RED, a, "mm", track=3)
        L.text((cx0, y), f"{self.bad:.0f} s", font("num", 190), RED, a, "mm", glow=0.5)
        s = ease((u - 0.55) / 0.25)
        if s > 0:  # risca o número falso
            cv2.line(ov, (int(cx0 - 140), y + 4), (int(cx0 - 140 + 280 * s), y - 26), bgr(WHITE, 0.95), 7,
                     cv2.LINE_AA)
        b = R.clamp01((u - 0.75) / 0.2)
        if b > 0:
            n = lerp(self.bad, self.good, ease((u - 0.75) / 0.7))
            L.text((cx1, 560), TX["counter"][2], font("monob", 26), CYAN, b, "mm", track=3)
            L.text((cx1, y), f"{n:.0f} s", font("num", 190), CYAN, b, "mm", glow=0.6)
            L.text((W / 2, y), "→", font("monob", 72), SOFT, b, "mm")

    def zone_spot(self, i, view):
        """Onde escrever o nome da zona: o centro da parte dela que aparece no quadro (ou None)."""
        q = self.d.zone_q[i]
        x0 = int(max(0, (view.cx - W / 2 / view.z) / 4))
        x1 = int(min(q.shape[1], (view.cx + W / 2 / view.z) / 4))
        sub = q[:, x0:x1]
        if sub.sum() < 300:
            return None
        gy, gx = np.mgrid[0:sub.shape[0], x0:x1]
        return view.pt((gx * sub).sum() / sub.sum() * 4, (gy * sub).sum() / sub.sum() * 4)

    def s_zones(self, u, dur, ov, L, t):
        if u < ZONES_FF:
            src = FILTER_F / self.fps
        else:
            src = FILTER_F / self.fps + (self.src_end - FILTER_F / self.fps) * (u - ZONES_FF) / (dur - ZONES_FF)
        f = int(round(src * self.fps))
        k = ease(u / dur)
        view = View(lerp(300, 560, k), 540, ZFULL)
        img = view.warp(self.ai(f, gain=1.25))
        if u >= ZONES_FF:
            self.people(ov, L, view, f, t, labels=False, tags=False, a=0.8)
        vals = self.d.cum_zone[min(f, self.d.n_frames - 1)]
        fill, lines = np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8)
        for i, z in enumerate(self.d.zones):
            g = ease((u - 0.1 - 0.05 * i) / 0.5)
            if g <= 0:
                continue
            hot = R.clamp01(vals[i] / max(self.vals.max(), 1e-6))
            col = tuple(int(lerp(c0, c1, hot)) for c0, c1 in zip(CYAN, AMBER))
            for poly in z["polys"]:
                pts = [view.pt(*p) for p in poly]
                cv2.fillPoly(fill, [np.array(pts, np.int32)], bgr(col, 0.16 * g))
                poly_partial(lines, pts, g, bgr(col, 0.85), 2)
            c = self.zone_spot(i, view)
            if c is not None and g > 0.6:
                c = (min(max(c[0], 110), W - 110), min(max(c[1], 420), 960))
                a = R.clamp01((g - 0.6) / 0.4)
                L.text(c, R.zone_text(z), font("monob", 26), WHITE, a, "mm", track=2, glow=0.3)
                if u >= ZONES_FF:
                    L.text((c[0], c[1] + 42), f"{vals[i]:.1f} s", font("num", 46), AMBER, a, "mm", glow=0.4)
        fill[:330], lines[:330] = 0, 0
        img += fill
        cv2.add(ov, lines, ov)
        self.block(L, u, TX["chip"].format(i=5, name=TX["steps"][4]), *TX["zones"])
        return img, src

    def s_result(self, u, dur, ov, L, t):
        k = ease(u / dur)
        view = View(lerp(700, 300, k), 540, ZFULL)
        img = view.warp(self.final_bg)
        for r, zi in enumerate(self.order[:3]):
            c = view.pt(*self.marks[zi])
            seen = next((s for s in np.arange(0, dur, 1 / FPS)
                         if 80 < View(lerp(700, 300, ease(s / dur)), 540).pt(*self.marks[zi])[0] < W - 80), dur)
            t_in = max(0.4 + 0.2 * r, seen)
            a = R.clamp01((u - t_in) / 0.25)
            if a <= 0:
                continue
            rad = int(34 * ease((u - t_in) / 0.3))
            cv2.circle(ov, ip(c), rad, bgr(AMBER), -1, cv2.LINE_AA)
            L.text((c[0], c[1] + 3), str(r + 1), font("num", 52), INK, a, "mm")
        top = [self.vals[i] / self.vals.sum() for i in self.order[:2]]
        zone = R.zone_text(self.d.zones[self.order[0]])
        head = (TX["result_close"] if top[0] - top[1] < 0.03 else TX["result"]).format(zone=zone)
        y = 1000
        self.chip(L, 64, y, TX["result_tag"].format(s=f"{self.src_end:.0f}"), u)
        a = R.clamp01((u - 0.06) / 0.2)
        L.rich(64, y + 140, [(head, WHITE)], font("head", 92), 900, 94, a, glow=0.3, rise=26)
        c = ease((u - 0.3) / 1.2)
        stats = TX["result_stats"].format(n=round(self.d.n_customers * c), g=round(self.d.cum_count[-1] * c),
                                          s=f"{self.vals.sum() * c:.0f}")
        L.text((66, y + 196), stats, font("mono", 26), SOFT, R.clamp01((u - 0.3) / 0.2), "ls", track=1)
        for r, zi in enumerate(self.order[:3]):
            a = R.clamp01((u - 0.8 - 0.2 * r) / 0.25)
            if a <= 0:
                continue
            yy = y + 278 + r * 74
            share = self.vals[zi] / self.vals.sum()
            grow = ease((u - 0.8 - 0.2 * r) / 0.8)
            L.text((64, yy), str(r + 1), font("num", 52), AMBER, a, "ls")
            L.text((110, yy - 4), R.zone_text(self.d.zones[zi], "name"), font("demi", 36), WHITE, a, "ls")
            L.text((900, yy), f"{share * 100 * grow:.0f}%", font("num", 52), AMBER, a, "rs", glow=0.4)
            L.box((110, yy + 14, 900, yy + 20), fill=(255, 255, 255, 40), r=3, a=a)
            L.box((110, yy + 14, 110 + max(6, 790 * share / top[0] * grow), yy + 20), fill=AMBER, r=3, a=a, glow=0.6)
        return img, self.src_end

    def p2_frame(self, k):
        """Quadro k do voo da parte 2 (4:5), recortado em 9:16 e no visual de holograma."""
        if k not in self._p2:
            img = cv2.imread(str(self.part2 / "voo" / f"{k:04d}.png")) if self.part2 else None
            if img is not None:
                h, w = img.shape[:2]
                cw = int(round(h * W / H))
                x0 = (w - cw) // 2
                img = cyber(cv2.resize(img[:, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC))
            self._p2[k] = img
            if len(self._p2) > 8:
                self._p2.pop(next(iter(self._p2)))
        return self._p2[k]

    def s_next(self, u, dur, ov, L, t):
        n = 165
        k = 1 + min(n - 1, int(ease(u / dur) * (n - 1)))
        img = self.p2_frame(k)
        img = View(300, 540).warp(self.final_bg) if img is None else img.copy()
        self.block(L, u, TX["next_tag"], TX["next"][0], TX["next"][1], fill=MAGENTA)
        L.text((64, 1470), TX["credit"], font("mono", 20), DIM, R.clamp01((u - 0.6) / 0.3), "ls")
        return img, None

    def timeline(self):
        S = self.seg
        ev = {
            "cuts": [a for n, (a, b) in S.items() if a > 0],
            "scan": S["detect"][0] + 0.65, "freeze": S["crop"][0], "window": S["crop"][0] + 0.6,
            "squash": S["crop"][0] + 1.2, "enhance": S["crop"][0] + 2.3, "heatmap": S["gaze"][0] + 0.1,
            "lock": S["gaze"][0] + GAZE_LOCK, "beam": S["gaze"][0] + GAZE_BEAM, "run": S["gaze"][0] + GAZE_RUN,
            "x": [S["filter"][0] + w for w in X_OFF + X_ONP], "vanish": S["filter"][0] + VANISH,
            "counter": S["filter"][0] + COUNTER, "strike": S["filter"][0] + COUNTER + 0.55,
            "count": S["filter"][0] + COUNTER + 0.75, "ff": S["zones"][0] + ZONES_FF,
            "rows": [S["result"][0] + 0.8 + 0.2 * r for r in range(3)],
        }
        return {"total": self.total, "fps": FPS, "segments": [[n, a, b] for n, (a, b) in S.items()], "events": ev}


class NoRules(R.Data):
    """Os mesmos dados sem a regra de pessoas: o olhar sobre alguém também conta para a zona."""

    def on_person(self, f, x, y, r=4):
        return False


def no_rules(data_dir):
    """O que um pipeline ingênuo contaria: sem a regra da borda nem a de pessoas."""
    edge, R.EDGE = R.EDGE, -1  # nenhum pico vira "fora da imagem"
    try:
        return NoRules(data_dir)
    finally:
        R.EDGE = edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="dados do extract_v2.py (com zones.json)")
    ap.add_argument("out_dir")
    ap.add_argument("--part2", default="", help="pasta com voo/ do Blender (outputs/.../promo_en)")
    ap.add_argument("--stills", default="", help="instantes (s) para gravar como JPG, separados por vírgula")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = Reel(args.data_dir, args.part2)
    print(json.dumps({n: [round(a, 2), round(b, 2)] for n, (a, b) in r.seg.items()}), round(r.total, 2), "s")
    if args.stills:
        st = out / "stills"
        st.mkdir(exist_ok=True)
        for s in args.stills.split(","):
            cv2.imwrite(str(st / f"t{float(s):05.2f}.jpg"), r.frame(float(s)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    (out / "reel_timeline.json").write_text(json.dumps(r.timeline(), indent=1))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                           str(out / "reel_mudo.mp4")], stdin=subprocess.PIPE)
    n = int(round(r.total * FPS))
    for i in range(n):
        ff.stdin.write(r.frame(i / FPS).tobytes())
        if i % 90 == 0:
            print(f"{i / FPS:.1f}s / {r.total:.1f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    cv2.imwrite(str(out / "reel_capa.jpg"), r.frame(1.2), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("ok", n, "quadros")


if __name__ == "__main__":
    main()
