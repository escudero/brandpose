"""Trilha do reel do Instagram (sintetizada, sem música de terceiros), em clima synthwave/cyberpunk.

Lá menor a 120 bpm, com compassos de 2 s a partir de 3 s, onde caem os cortes do reel.py:
  gancho    impacto, baixo pulsando com o filtro abrindo, subida até a entrada
  passo 1   a batida inteira (bumbo, palmas, chimbal, baixo, arpejo e pads)
  passos 2-3  tempo congelado: a bateria para, sobram pads e arpejo abafado, com os sons da interface
  passo 3   a batida volta quando o tempo volta a correr
  passo 4   zumbidos de erro em cada olhar descartado, impacto e contagem no placar
  passo 5   avanço rápido: chimbal dobrado e subida até o resultado
  resultado a parte mais cheia, com uma melodia e uma nota por linha do ranking
  parte 2   a bateria sai; um prato invertido leva de volta ao começo (o reel repete)
Sincroniza pelo reel_timeline.json do reel.py e junta a trilha ao vídeo mudo, com a
loudness em -14 LUFS (a referência das plataformas).

Uso: python reel_audio.py OUT_DIR  ->  OUT_DIR/reel_trilha.wav e OUT_DIR/reel.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import audio as A  # os sintetizadores do vídeo de demonstração

SR = A.SR
BEAT, T0 = 0.5, 3.0  # 120 bpm; o primeiro tempo forte da batida
rng = np.random.default_rng(11)
T, env, lp, hp, place = A.T, A.env, A.lp, A.hp, A.place

A1, C2, E2, F1, G1 = 55.0, 65.41, 82.41, 43.65, 49.0
F2, G2, A2, B2 = 87.31, 98.0, 110.0, 123.47
C3, D3, E3, F3, G3, A3, B3 = 130.81, 146.83, 164.81, 174.61, 196.0, 220.0, 246.94
C4, D4, E4, F4, G4, A4, B4 = 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88
C5, D5, E5, G5, A5 = 523.25, 587.33, 659.25, 783.99, 880.0

# acordes: (baixo, sub, pad, arpejo)
CHORDS = {
    "Am": (A2, A1, [A2, E3, A3, C4, E4], [A3, C4, E4, A4, E4, C4, A3, E3]),
    "F": (F2, F1, [F3, A3, C4, E4], [F3, A3, C4, F4, C4, A3, F3, C4]),
    "C": (C3, C2, [C3, G3, C4, E4, G4], [C4, E4, G4, C5, G4, E4, C4, G3]),
    "G": (G2, G1, [G3, B3, D4, G4], [G3, B3, D4, G4, D4, B3, G3, D4]),
}
# um acorde por compasso, a partir de T0 (o gancho fica em lá menor)
BARS = ["Am", "F", "C", "G", "Am", "F", "C", "G", "Am", "F", "G", "Am", "F", "C", "G", "Am"]


# ---------------------------------------------------------------- sintetizadores

def saw(f, d, nmax=40, phase=0.0):
    """Dente de serra com banda limitada (soma de harmônicos)."""
    t = T(d)
    n = int(max(1, min(nmax, SR / 2 / f - 1)))
    return sum(np.sin(2 * np.pi * f * k * t + phase * k) / k for k in range(1, n + 1)) * (2 / np.pi)


def bass(f, d):
    """Baixo synthwave: serra com o filtro fechando depressa (o "plac" do começo) e um corpo escuro."""
    x = saw(f, d, 30)
    bright, dark = lp(x, 2600, 2), lp(x, 380, 2)
    k = np.exp(-T(d) / 0.05)
    return (dark + (bright - dark) * k) * env(len(x), 0.003, 0.04)


def sub(f, d):
    return np.sin(2 * np.pi * f * T(d)) * env(int(d * SR), 0.01, 0.05)


def pad(freqs, d, a=0.4, r=0.6, cutoff=2200):
    """Pad de serras desafinadas (supersaw), aberto no estéreo."""
    out = np.zeros((int(d * SR), 2))
    for f in freqs:
        for ch, cents in ((0, -9), (1, 9), (0, 3), (1, -3)):
            out[:, ch] += saw(f * 2 ** (cents / 1200), d, 18, rng.uniform(0, 6.28))
    out = lp(out / len(freqs), cutoff, 2)
    return out * env(len(out), a, r)[:, None]


def pluck(f, d=0.2, bright=3200):
    """Nota curta do arpejo: onda quase quadrada (harmônicos ímpares) que morre rápido."""
    t = T(d)
    x = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 3, 5, 7, 9))
    return lp(x * np.exp(-t / 0.07), bright, 2) * env(len(t), 0.002, 0.03)


def lead(f, d):
    """Melodia do resultado: serra com vibrato que entra devagar."""
    t = T(d)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.4, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    x = sum(np.sin(ph * k) / k for k in range(1, 12))
    return lp(x, 3000, 2) * env(len(t), 0.03, 0.25)


def punch():
    """Bumbo do vídeo de demonstração com um clique no ataque, para aparecer no celular."""
    k = A.kick()
    k[: int(0.004 * SR)] += hp(rng.standard_normal(int(0.004 * SR)), 3000) * 0.3
    return k


def clap():
    n = rng.standard_normal(int(0.3 * SR))
    x = hp(lp(n, 2600, 2), 900, 2)
    t = T(0.3)
    e = sum(np.exp(-np.clip(t - o, 0, None) / 0.005) * (t >= o) for o in (0, 0.011, 0.022))
    e += np.exp(-np.clip(t - 0.03, 0, None) / 0.08) * (t >= 0.03) * 0.8
    return x * e * env(len(t), 0.001, 0.05)


def snare(gain=1.0):
    t = T(0.22)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.04)
    noise = hp(lp(rng.standard_normal(len(t)), 7000), 1200) * np.exp(-t / 0.06)
    return (tone * 0.5 + noise) * env(len(t), 0.001, 0.03) * gain


def open_hat():
    t = T(0.25)
    return hp(rng.standard_normal(len(t)), 7500) * np.exp(-t / 0.07) * env(len(t), 0.002, 0.05)


def crash(d=2.2):
    t = T(d)
    x = hp(rng.standard_normal((len(t), 2)), 4500, 2)
    metal = sum(np.sin(2 * np.pi * f * t) for f in (3250, 4870, 6110, 7390)) * 0.08
    return (x + metal[:, None]) * (np.exp(-t / 0.55) * env(len(t), 0.002, 0.2))[:, None]


def tape_stop(d=0.45):
    """O tempo para: um tom que despenca de altura, com chiado."""
    t = T(d)
    f = 240 * (30 / 240) ** (t / d)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(ph * k) / k for k in range(1, 8)) + lp(rng.standard_normal(len(t)), 2000) * 0.2
    return x * env(len(t), 0.005, 0.1)


def laser(d=0.3, f0=3200, f1=180):
    t = T(d)
    f = f0 * (f1 / f0) ** (t / d)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph + 2.0 * np.sin(ph * 0.5)) * np.exp(-t / 0.12) * env(len(t), 0.002, 0.05)


def beep(f, d=0.07):
    t = T(d)
    return np.sign(np.sin(2 * np.pi * f * t)) * 0.4 * env(len(t), 0.002, 0.02)


def buzz(d=0.22):
    """Erro: duas ondas quadradas em trítono, distorcidas e tremendo."""
    t = T(d)
    x = np.sign(np.sin(2 * np.pi * 116.5 * t)) + np.sign(np.sin(2 * np.pi * 164.8 * t))
    x = np.tanh(2.5 * x) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 32 * t)))
    return lp(x, 3500, 2) * env(len(t), 0.003, 0.04)


def zap(d=0.35, down=True):
    t = T(d)
    f = (2400 * (120 / 2400) ** (t / d)) if down else (150 * (2600 / 150) ** (t / d))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + hp(rng.standard_normal(len(t)), 2000) * 0.3
    return x * np.exp(-t / (d * 0.5)) * env(len(t), 0.003, 0.05)


def shimmer(d=1.2, n=26):
    out = np.zeros(int(d * SR))
    for _ in range(n):
        f = rng.choice([A5, E5, C5, A4 * 2, E5 * 2])
        x = np.sin(2 * np.pi * f * T(0.12)) * np.exp(-T(0.12) / 0.03)
        i = int(rng.uniform(0, d - 0.13) * SR)
        out[i:i + len(x)] += x * rng.uniform(0.3, 1)
    return out * np.linspace(1, 0.4, len(out))


def enhance(d=0.6):
    """Zoom digital: um glissando em degraus subindo, com bipes de dados."""
    t = T(d)
    steps = np.floor(t / 0.04)
    f = 300 * (2600 / 300) ** (steps * 0.04 / d)
    x = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.35 + np.sin(2 * np.pi * np.cumsum(f * 2) / SR) * 0.3
    return lp(x, 6000, 2) * env(len(t), 0.005, 0.08)


def ticks(n=10, gap=0.07, f0=1800, f1=700):
    out = np.zeros(int((n * gap + 0.1) * SR))
    for i in range(n):
        f = f0 * (f1 / f0) ** (i / max(1, n - 1))
        x = beep(f, 0.035)
        j = int(i * gap * SR)
        out[j:j + len(x)] += x
    return out


def stutter(x, reps=5, slice_s=0.045):
    """Repete o começo do som em fatias curtas (gaguejo de glitch)."""
    s = x[: int(slice_s * SR)] * env(int(slice_s * SR), 0.002, 0.01)
    return np.concatenate([s] * reps + [x[: int(0.2 * SR)]])


def ping_pong(x, delay=0.375, fb=0.38, mix=0.5):
    """Eco alternando os lados (colcheia pontuada)."""
    n = int(delay * SR)
    out = np.zeros((len(x) + 8 * n, 2))
    out[: len(x)] += x if x.ndim == 2 else np.stack([x, x], 1)
    mono = x.mean(1) if x.ndim == 2 else x
    g = mix
    for k in range(1, 8):
        out[k * n:k * n + len(mono), k % 2] += mono * g
        g *= fb
    return out[: len(x)]


# ---------------------------------------------------------------- arranjo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, help="pasta do reel.py (reel_timeline.json e reel_mudo.mp4)")
    args = ap.parse_args()
    tl = json.loads((args.out_dir / "reel_timeline.json").read_text())
    seg = {n: (a, b) for n, a, b in tl["segments"]}
    ev, total = tl["events"], tl["total"]
    n = int(total * SR) + 2 * SR
    drums, music, fx, wet = (np.zeros((n, 2)) for _ in range(4))
    duck = np.ones(n)

    def bar_at(t):
        return CHORDS[BARS[min(len(BARS) - 1, max(0, int((t - T0) // 2)))]]

    def kick(t, g=0.9):
        place(drums, punch(), t, g)
        i0, dl = int(t * SR), int(0.3 * SR)
        duck[i0:i0 + dl] = np.minimum(duck[i0:i0 + dl], 1 - 0.55 * np.exp(-np.arange(min(dl, n - i0)) / (0.09 * SR)))

    def groove(t0, t1, clap_on=True, hats16=False, half=False):
        b = t0
        while b < t1 - 1e-6:
            k = round((b - T0) / BEAT)
            if not half or k % 4 == 0:  # pela metade: bumbo no 1 e palmas no 3
                kick(b)
            if clap_on and (k % 4 == 2 if half else k % 2 == 1):
                place(drums, clap(), b, 0.38)
            place(drums, A.hat(), b + BEAT / 2, 0.15)
            if hats16:
                for q in (0.125, 0.375):
                    place(drums, A.hat(), b + q, 0.08)
            if k % 4 == 3:
                place(drums, open_hat(), b + BEAT / 2, 0.06)
            b += BEAT

    def bassline(t0, t1, gain=0.5, sub_gain=0.35):
        t = t0
        while t < t1 - 1e-6:
            root, low, _, _ = bar_at(t)
            k = round((t - T0) / (BEAT / 2))
            f = root * (2 if k % 4 == 2 else 1)  # galope: a terceira colcheia sobe uma oitava
            place(music, bass(f, BEAT / 2 - 0.01), t, gain)
            if k % 2 == 0:
                place(music, sub(low, BEAT - 0.02), t, sub_gain)
            t += BEAT / 2

    def pads(t0, t1, gain=0.2, cutoff=2200):
        t = t0
        while t < t1 - 1e-6:
            d = min(2.0 - ((t - T0) % 2.0), t1 - t)
            place(music, pad(bar_at(t)[2], d + 0.35, a=0.25, r=0.4, cutoff=cutoff), t, gain)
            t += d

    def arp(t0, t1, gain=0.12, bright=3200):
        t = t0
        notes = []
        while t < t1 - 1e-6:
            k = round((t - T0) / (BEAT / 4))
            notes.append((t, bar_at(t)[3][k % 8]))
            t += BEAT / 4
        x = np.zeros((int((t1 - t0 + 1) * SR), 2))
        for tn, f in notes:
            place(x, pluck(f, 0.2, bright), tn - t0, 1.0)
        place(music, ping_pong(x), t0, gain)

    def roll(t0, t1, g=0.18):
        t, step = t0, 0.125
        while t < t1 - 1e-6:
            place(drums, snare(), t, g * (0.4 + 0.6 * (t - t0) / (t1 - t0)))
            t += step
            if t - t0 > (t1 - t0) / 2:
                step = 0.0625

    names = ("hook", "detect", "crop", "gaze", "filter", "zones", "result", "next")
    hook, det, crop, gaze, filt, zones, res, nxt = (seg[k] for k in names)

    # gancho: impacto, baixo pulsando com o filtro abrindo, subida
    place(fx, A.boom(2.2), 0.0, 0.5)
    place(fx, crash(1.6), 0.0, 0.18)
    place(wet, A.chime([A4, E5, A5], 2.0), 0.02, 0.12)
    t = 0.0
    while t < hook[1] - 1e-6:
        x = bass(A2 if round(t / 0.25) % 4 != 2 else A3, 0.24)
        k = t / hook[1]
        place(music, lp(x, 300 + 2600 * k ** 2, 2), t, 0.42)
        if round(t / 0.25) % 2 == 0:
            place(music, sub(A1, 0.48), t, 0.25)
        t += 0.25
    place(music, pad(CHORDS["Am"][2], hook[1] + 0.3, a=0.6, r=0.3, cutoff=1500), 0.0, 0.16)
    for q in np.arange(0.5, hook[1], 0.125):
        place(drums, A.hat(), q, 0.04 + 0.04 * q / hook[1])
    place(fx, A.riser(1.2, 300, 7000), hook[1] - 1.2, 0.1)
    roll(hook[1] - 1.0, hook[1])
    place(fx, stutter(beep(1500, 0.2)), hook[1] - 0.3, 0.08)

    # passo 1: rebobina e entra a batida
    place(fx, A.rewind(0.32), det[0], 0.25)
    place(fx, crash(), det[0], 0.2)
    groove(det[0], det[1])
    bassline(det[0], det[1])
    pads(det[0], det[1])
    arp(det[0], det[1])
    sc = ev["scan"]
    noise = lp(rng.standard_normal((int(1.0 * SR), 2)), 5000) * np.sin(np.linspace(0, np.pi, int(1.0 * SR)))[:, None]
    place(fx, noise, sc, 0.08)
    place(fx, zap(0.6, down=False), sc, 0.06)

    # passos 2 e 3 congelados: a bateria para; pads, sub e arpejo abafado
    place(fx, tape_stop(), crop[0], 0.3)
    place(fx, A.boom(1.6), crop[0], 0.3)
    pads(crop[0], ev["run"], gain=0.22, cutoff=1400)
    arp(crop[0] + 0.5, ev["run"], gain=0.09, bright=1300)
    t = crop[0]
    while t < ev["run"] - 1e-6:  # pulso discreto para não perder o tempo
        place(music, sub(bar_at(t)[1], BEAT * 2 - 0.05), t, 0.3)
        place(drums, A.hat(), t + BEAT, 0.05)
        t += BEAT * 2
    place(fx, beep(1320, 0.06), ev["window"], 0.12)
    place(fx, beep(1760, 0.06), ev["window"] + 0.08, 0.1)
    place(fx, zap(0.4), ev["squash"], 0.14)
    place(fx, enhance(0.6), ev["enhance"], 0.12)
    place(wet, A.blip(), ev["enhance"] + 0.3, 0.12)
    place(wet, shimmer(1.0), ev["heatmap"], 0.1)
    place(fx, beep(2093, 0.05), ev["lock"], 0.14)
    place(fx, beep(2093, 0.05), ev["lock"] + 0.09, 0.14)
    place(fx, laser(0.32), ev["beam"], 0.2)
    place(wet, A.pluck(E5, 1.2), ev["beam"] + 0.3, 0.12)
    place(fx, A.riser(1.0, 400, 6000), ev["run"] - 1.0, 0.09)
    roll(ev["run"] - 0.5, ev["run"], 0.14)

    # o tempo volta: batida até o fim do passo 3
    place(fx, crash(), ev["run"], 0.16)
    groove(ev["run"], gaze[1])
    bassline(ev["run"], gaze[1])
    pads(ev["run"], gaze[1])
    arp(ev["run"], gaze[1])

    # passo 4: batida pela metade, erros, impacto e contagem do placar
    groove(filt[0], filt[1], clap_on=True, half=True)
    bassline(filt[0], filt[1], gain=0.45)
    pads(filt[0], filt[1], gain=0.2, cutoff=1800)
    arp(filt[0], ev["counter"], gain=0.1, bright=2400)
    for tx in ev["x"]:
        place(fx, buzz(), tx, 0.2)
        place(fx, stutter(buzz(0.12), 3, 0.03), tx + 0.02, 0.06)
    place(fx, zap(0.5, down=False)[::-1].copy(), ev["vanish"] - 0.25, 0.1)
    place(fx, A.boom(2.0), ev["counter"], 0.42)
    place(fx, crash(1.4), ev["counter"], 0.12)
    place(fx, zap(0.25), ev["strike"], 0.18)
    place(fx, ticks(10, 0.07), ev["count"], 0.1)
    arp(ev["counter"] + 1.0, filt[1], gain=0.11)

    # passo 5: zonas acendendo, depois avanço rápido e subida
    for i in range(9):
        place(wet, A.pluck([A4, C5, E5, A5, E5 * 2, C5, A4 * 2, E5, A5][i], 0.5), zones[0] + 0.1 + 0.05 * i, 0.06)
    ff = ev["ff"]
    pads(zones[0], zones[1], gain=0.2)
    groove(zones[0], ff, clap_on=False, half=True)
    groove(ff, zones[1], hats16=True)
    bassline(zones[0], zones[1])
    arp(zones[0], zones[1], gain=0.12)
    gl = T(zones[1] - ff)
    glide = np.sin(2 * np.pi * np.cumsum(np.geomspace(220, 660, len(gl))) / SR) * env(len(gl), 0.3, 0.05)
    place(fx, lp(glide, 2500), ff, 0.05)
    place(fx, A.riser(2.0, 300, 8000), zones[1] - 2.0, 0.12)
    roll(zones[1] - 1.0, zones[1], 0.2)

    # resultado: a parte mais cheia, com melodia e uma nota por linha do ranking
    place(fx, A.boom(2.4), res[0], 0.5)
    place(fx, crash(2.4), res[0], 0.22)
    groove(res[0], res[1], hats16=True)
    bassline(res[0], res[1], gain=0.52)
    pads(res[0], res[1], gain=0.22, cutoff=2600)
    arp(res[0], res[1], gain=0.12)
    for t0_, f, d in [(0, E5, 1.0), (1.0, C5, 1.0), (2.0, C5, 0.5), (2.5, A4, 1.5), (4.0, G4, 0.5), (4.5, E5, 1.45)]:
        place(music, np.stack([lead(f, d)] * 2, 1), res[0] + t0_, 0.1)
        place(wet, np.stack([lead(f, d)] * 2, 1), res[0] + t0_, 0.05)
    for t_, f in zip(ev["rows"], [A5, E5, C5]):
        place(wet, A.pluck(f, 1.4), t_, 0.2)

    # parte 2: a bateria sai, o arpejo continua e um prato invertido volta ao começo
    place(fx, zap(0.5, down=False), nxt[0] - 0.05, 0.1)
    place(fx, A.boom(1.8), nxt[0], 0.3)
    pads(nxt[0], nxt[1], gain=0.24, cutoff=1800)
    arp(nxt[0], nxt[1], gain=0.11)
    t = nxt[0]
    while t < nxt[1] - 1e-6:
        place(music, sub(bar_at(t)[1], BEAT * 2 - 0.05), t, 0.32)
        t += BEAT * 2
    place(fx, crash(1.4)[::-1].copy(), nxt[1] - 1.4, 0.16)
    place(fx, A.riser(1.4, 300, 6000), nxt[1] - 1.4, 0.06)

    mix = drums + music * duck[:, None] + fx + A.reverb(wet) * 1.0 + wet * 0.5
    mix += A.reverb(music * 0.25) * 0.35  # um pouco de ambiente na harmonia
    mix = mix[: int(total * SR)]
    # equalização para celular: menos subgrave (que o alto-falante não toca) e mais brilho
    mix = hp(mix, 28, 2)
    mix = mix - 0.25 * lp(mix, 60, 2) + 0.4 * hp(mix, 4000, 2)
    mix = np.tanh(1.2 * mix / np.abs(mix).max() * 1.6) / np.tanh(1.2 * 1.6)
    mix *= 0.9 / np.abs(mix).max()
    wav = args.out_dir / "reel_trilha.wav"
    wavfile.write(wav, SR, (mix * 32767).astype(np.int16))

    # loudness: -14 LUFS integrados, pico real em -1,5 dBTP (medição em duas passadas)
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(wav), "-af",
                           "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    m = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
    norm = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
            "linear=true")
    out = args.out_dir / "reel.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.out_dir / "reel_mudo.mp4"), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", norm, "-ar", str(SR), "-c:a", "aac",
                    "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    print(json.dumps({"video": str(out), "seconds": round(len(mix) / SR, 2), "measured_lufs": m["input_i"],
                      "measured_tp": m["input_tp"]}))


if __name__ == "__main__":
    main()
