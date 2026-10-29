"""Trilha do vídeo promocional da visão do cliente (sintetizada, sem música de terceiros).

Usa os sintetizadores de scripts/promo/audio.py com outro arranjo: começa misteriosa (si
menor) na câmera real, para junto com o tempo, abre em ré maior quando a loja vira 3D,
pulsa durante a visão dele e resolve no fim. Sincroniza pelo promo_<idioma>_timeline.json
do promo.py e junta a trilha ao vídeo mudo.

Uso: python promo_audio.py RUN_DIR [--lang en]  ->  RUN_DIR/promo_<idioma>.mp4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "promo"))
import audio as A  # noqa: E402  (os sintetizadores do vídeo de demonstração)

SR = A.SR
B1, D2, FS2, G2, A2, B2 = 61.74, 73.42, 92.50, 98.0, 110.0, 123.47
D3, E3, FS3, G3, A3, B3 = 146.83, 164.81, 185.0, 196.0, 220.0, 246.94
CS4, D4, E4, FS4, G4, A4, B4 = 277.18, 293.66, 329.63, 369.99, 392.0, 440.0, 493.88
D5, E5, FS5, A5 = 587.33, 659.25, 739.99, 880.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--lang", choices=["pt", "en"], default="pt")
    args = ap.parse_args()
    name = f"promo_{args.lang}"
    tl = json.loads((args.run_dir / f"{name}_timeline.json").read_text())
    seg = {n: (a, b) for n, a, b in tl["segments"]}
    ev, total = tl["events"], tl["total"]
    n = int(total * SR) + SR
    dry, wet, duck = np.zeros((n, 2)), np.zeros((n, 2)), np.ones(n)
    place, T, env = A.place, A.T, A.env

    # gancho: o ambiente da loja e um arpejo de si menor, em suspense
    g1 = seg["congela"][0]
    amb = A.lp(A.rng.standard_normal((int((g1 + 1.4) * SR), 2)), 900)
    place(dry, amb * env(len(amb), 0.3, 0.9)[:, None], 0, 0.05)
    place(dry, A.pad([B2, FS3, B3, D4], g1 + 0.5, a=1.0, r=0.5, cutoff=1300), 0, 0.24)
    for t, f in [(0.05, B4), (0.6, FS4), (1.15, D5), (1.75, B4), (ev["enters"], A4), (ev["enters"] + 0.3, FS4)]:
        place(wet, A.pluck(f, 1.4), t, 0.11)

    # congela: o tempo para (tom caindo) e fica um brilho no ar
    c0 = seg["congela"][0]
    place(dry, A.rewind(0.45), c0, 0.10)
    place(wet, A.chime([1975.5, 2959.96], 2.2), c0 + 0.05, 0.10)
    place(dry, A.riser(ev["wipe"] - c0, 200, 6500), c0, 0.11)

    # revela: impacto e o acorde abre em ré maior
    w = ev["wipe"]
    place(dry, A.boom(2.6), w, 0.55)
    place(wet, A.chime([D5, FS5, A5, 1174.66], 3.2), w + 0.02, 0.24)
    t = w + 0.3
    while t < seg["revela"][1]:  # brilhos enquanto a cortina passa
        place(wet, A.pluck(A.rng.choice([D5, FS5, A5, 1174.66]), 1.0), t, 0.05)
        t += A.rng.uniform(0.12, 0.22)

    # camas: ré maior, si menor, sol, lá, ré (da cortina até o fim da visão)
    chords = [(w, [D3, A3, D4, FS4, A4]), (seg["voo"][0] + 2.4, [B2, FS3, B3, D4, FS4]),
              (seg["visao"][0], [G2, D3, G3, B3, D4]), (seg["visao"][0] + 4.0, [A2, E3, A3, CS4, E4]),
              (seg["visao"][0] + 8.0, [D3, A3, D4, FS4, A4]), (seg["mapa"][0], [G2, D3, G3, B3, D4]),
              (seg["fim"][0], [D2, A2, D3, FS3, A3, E4])]
    for (t0, notes), (t1, _) in zip(chords, chords[1:] + [(total + 0.5, None)]):
        place(dry, A.pad(notes, t1 - t0 + 0.7, a=0.5, r=0.9, cutoff=2400), t0 - 0.15, 0.27)

    # voo: vento que acompanha a câmera (mais rápida no meio) e a chegada nos olhos
    v0, v1 = seg["voo"]
    tt = T(v1 - v0)
    speed = np.sin(np.pi * tt / tt[-1]) ** 2
    place(dry, A.lp(A.rng.standard_normal((len(tt), 2)), 1100) * speed[:, None], v0, 0.16)
    place(dry, A.riser(1.6, 300, 5000), ev["eyes"] - 1.6, 0.09)
    place(dry, A.boom(1.8), ev["eyes"], 0.32)
    place(wet, A.chime([A5, 1318.5], 2.0), ev["eyes"] + 0.02, 0.14)

    # pulso: começa no voo, dobra o chimbal no trecho acelerado
    t, end = v0 + 2.4, seg["mapa"][0] - 0.1
    while t < end:
        fast = t >= ev["fast"]
        place(dry, A.kick(), t, 0.42 if t < seg["visao"][0] else 0.5)
        i0, dl = int(t * SR), int(0.35 * SR)
        duck[i0:i0 + dl] = np.minimum(duck[i0:i0 + dl], 1 - 0.28 * np.exp(-np.arange(min(dl, n - i0)) / (0.08 * SR)))
        for h in ((0.15, 0.3, 0.45) if fast else (0.3,)):
            place(dry, A.hat(), t + h, 0.05)
        t += 0.6
    melody = [D5, FS5, A5, FS5, E5, D5, B4, D5, E5, FS5, A5, D5]
    for i, t in enumerate(np.arange(seg["visao"][0] + 0.6, ev["fast"], 1.2)):
        place(wet, A.pluck(melody[i % len(melody)], 1.2), t, 0.07)
    f0_, f1_ = ev["fast"], seg["mapa"][0]
    gl = T(f1_ - f0_)
    glide = np.sin(2 * np.pi * np.cumsum(np.geomspace(220, 440, len(gl))) / SR) * env(len(gl), 0.3, 0.2)
    place(dry, A.lp(glide, 2000), f0_, 0.05)
    place(dry, A.riser(1.2, 400, 7000), f1_ - 1.2, 0.08)

    # mapa: corte com impacto e uma nota por zona do ranking
    m0 = seg["mapa"][0]
    place(dry, A.boom(2.0), m0, 0.4)
    place(wet, A.blip(), m0 + 0.05, 0.10)
    for t, f in zip(ev["rows"], [D5, E5, FS5, A5]):
        place(wet, A.pluck(f, 1.6), t, 0.20)

    # manchetes novas: um toque discreto
    for t in ev["heads"][1:]:
        place(wet, A.blip(), t + 0.05, 0.06)

    # fim: resolve em ré maior com a marca
    place(dry, A.boom(2.6), ev["final"], 0.34)
    place(wet, A.chime([D5, FS5, A5, 1174.66], 3.5), ev["final"] + 0.02, 0.28)

    mix = dry * duck[:, None] + A.reverb(wet) * 1.0 + wet * 0.5
    mix = mix[: int(total * SR)]
    mix *= env(len(mix), 0.02, 1.0)[:, None]
    mix = np.tanh(1.3 * mix) / np.tanh(1.3)
    mix *= 0.89 / np.abs(mix).max()  # pico em -1 dBFS
    wav = args.run_dir / f"{name}_trilha.wav"
    wavfile.write(wav, SR, (mix * 32767).astype(np.int16))
    out = args.run_dir / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.run_dir / f"{name}_mudo.mp4"), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
    rms = np.sqrt((mix ** 2).mean())
    print(json.dumps({"video": str(out), "seconds": round(len(mix) / SR, 2),
                      "peak_dbfs": round(20 * np.log10(np.abs(mix).max()), 2), "rms_dbfs": round(20 * np.log10(rms), 2)}))


if __name__ == "__main__":
    main()
