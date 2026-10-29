"""Do mapa do Gaze-LLE (na imagem) para pontos nas faces dos móveis (em 3D).

O Gaze-LLE devolve, para cada pessoa, um mapa de calor na imagem: onde ela deve estar olhando.
Ele aprende com imagens, então o alvo é o que se vê naquele pixel. Para cada pixel do mapa:

1. o ponto 3D visto ali: a profundidade do LojaSim (Depth Anything com a escala dos pés das
   pessoas). Pixel de piso (máscara do LojaSim) não é gôndola;
2. a face do móvel: se o raio da câmera cruza uma face perto da profundidade, vale o cruzamento
   (geometria exata do móvel); senão, a face mais perto do ponto (até 50 cm);
3. a câmera às vezes vê o lado do móvel que a pessoa não vê (ela está do outro lado): aí o alvo
   passa para a face do mesmo móvel virada para ela, na mesma altura;
4. nota do alvo: distância de quem olha uma prateleira (até ~3,5 m conta inteiro, depois cai), nada
   no meio do caminho entre os olhos e o ponto, e o giro da cabeça medido pelos pontos do rosto
   (palpite, com peso pela confiança).

A massa que sobra (chão, gente, fora da imagem, nada plausível) não conta como atenção em gôndola.
O tempo de cada amostra (a cada quantos quadros o Gaze-LLE roda ÷ fps) é multiplicado pela chance
de o alvo estar na imagem (inout) e pela fração do mapa que caiu em cada ponto de cada face. Assim
a incerteza do olhar se espalha pelo mapa em vez de virar um ponto só.
"""

import math

import cv2
import numpy as np

BORDA_PX = 32         # pico colado na borda = alvo fora da imagem (o recorte é limitado ao quadro)
MASSA = 0.95          # usa os pixels que somam 95% da massa do mapa
MAX_PIX = 400
DIST_CHEIA, DIST_MAX = 3.5, 7.0
SIGMA_GIRO = math.radians(60)
PESO_GIRO = 0.6      # o giro pelos pontos do rosto é ruidoso: no máximo 60% da nota
PERTO_FACE = 0.5      # ponto da profundidade a até 50 cm de uma face conta para ela


def _celulas(hm, ds):
    v = hm.ravel()
    ordem = np.argsort(v)[::-1]
    acum = np.cumsum(v[ordem])
    n = int(np.searchsorted(acum, MASSA * acum[-1])) + 1
    sel = ordem[:min(n, MAX_PIX)]
    yy, xx = np.unravel_index(sel, hm.shape)
    return np.c_[(xx + 0.5) * ds, (yy + 0.5) * ds], v[sel]


def alvos_dos_pixels(cena, uv, O):
    """Para cada pixel: (face, s1, s2, P); face = -1 sem móvel perto, -2 chão. Sem as notas."""
    W, H = cena.cam["W"], cena.cam["H"]
    n = len(uv)
    D = cena.raios(uv)
    q = np.round(cena.sem_distorcao(uv)).astype(int)
    dentro = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    qx, qy = np.clip(q[:, 0], 0, W - 1), np.clip(q[:, 1], 0, H - 1)
    Pd = np.asarray(cena.pontos[qy, qx], float)
    td = ((Pd - cena.C) * D).sum(1)
    chao = cena.piso[qy, qx] | (Pd[:, 2] < 0.05)
    t, s1, s2 = cena.bate(np.repeat(cena.C[None], n, 0), D)
    face = np.where(chao, -2, -1)
    S1, S2, P = np.zeros(n), np.zeros(n), np.zeros((n, 3))
    for i in np.where(dentro & ~chao & (td > 0.3))[0]:
        tol = max(0.6, 0.25 * td[i])
        perto = np.where(np.abs(t[i] - td[i]) < tol)[0]
        if len(perto):
            k = perto[np.argmin(np.abs(t[i, perto] - td[i]))]
            face[i], S1[i], S2[i], P[i] = k, s1[i, k], s2[i, k], cena.C + t[i, k] * D[i]
            continue
        melhor = None
        a_, b_, d_ = cena.na_faces(Pd[i])
        k = int(np.argmin(d_))
        if d_[k] < PERTO_FACE:
            melhor = (d_[k], k, a_[k], b_[k])
        if melhor:
            face[i], S1[i], S2[i], P[i] = melhor[1], melhor[2], melhor[3], Pd[i]
    # a pessoa do outro lado do móvel: a face do mesmo móvel virada para ela
    for i in np.where(face >= 0)[0]:
        f = cena.faces[face[i]]
        if f.n @ (O - P[i]) > 0.05:
            continue
        irmas = [k for k, g in enumerate(cena.faces) if g.item == f.item and g.lado != "topo" and g.n @ (O - g.c) > 0.05]
        if not irmas:
            face[i] = -1
            continue
        k = max(irmas, key=lambda k: cena.faces[k].n @ (O - cena.faces[k].c))
        g = cena.faces[k]
        a, b, _ = cena.na_face(np.array([*P[i][:2], np.clip(P[i][2], 0, g.w2)]), k)
        face[i], S1[i], S2[i], P[i] = k, a, b, g.canto(a, b)
    return face, S1, S2, P


def projetar_olhares(cena, pessoas, gaze, heatmaps, masks, meta, frames):
    fps, ds = meta["fps"], meta["hm_ds"]
    dt = meta["gaze_every"] / fps
    hs, ws = heatmaps.shape[1:]
    mshape = tuple(meta["mask_shape"])
    por_id = {}
    for p in pessoas:
        for i in p["ids_rastreador"]:
            por_id.setdefault(i, []).append(p)
    borda = np.zeros((hs, ws), bool)
    b = int(math.ceil(BORDA_PX / ds))
    borda[:b], borda[-b:], borda[:, :b], borda[:, -b:] = True, True, True, True

    amostras = []
    for g in gaze:
        q, tid = g["frame"], g["id"]
        p = next((p for p in por_id.get(tid, []) if p["q0"] <= q < p["q0"] + len(p["x"])), None)
        if p is None:
            continue
        k = q - p["q0"]
        O = np.array([p["x"][k], p["y"][k], p["z_olhos"]])
        hm = heatmaps[g["hm"]].astype(np.float32)
        det = next((d for d in frames[q] if d["id"] == tid), None)
        if det and det.get("head"):  # a própria cabeça (ampliada), como no extract_v2
            x1, y1, x2, y2 = det["head"]
            cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.75, (y2 - y1) * 0.75
            hm[int(max(0, cy - hh) / ds):int((cy + hh) / ds) + 1, int(max(0, cx - hw) / ds):int((cx + hw) / ds) + 1] = 0
        total = float(hm.sum())
        if total <= 0:
            continue
        gente = np.unpackbits(masks[q])[: mshape[0] * mshape[1]].reshape(mshape).astype(np.float32)
        gente = cv2.resize(gente, (ws, hs), interpolation=cv2.INTER_AREA) > 0.5
        livre = hm.copy()
        livre[borda | gente] = 0  # olhar em gente ou fora da imagem não é atenção em gôndola
        alvos, perdas = [], {"fora_ou_gente": round(1 - float(livre.sum()) / total, 4), "chao": 0.0, "sem_movel": 0.0}
        if livre.sum() > 0:
            uv, massa = _celulas(livre, ds)
            face, S1, S2, P = alvos_dos_pixels(cena, uv, O)
            perdas["chao"] = round(float(massa[face == -2].sum()) / total, 4)
            perdas["sem_movel"] = round(float(massa[face == -1].sum()) / total, 4)
            ok = np.where(face >= 0)[0]
            if len(ok):
                dist = np.linalg.norm(P[ok] - O, axis=1)
                nota = np.where(dist <= DIST_CHEIA, 1.0, np.exp(-(dist - DIST_CHEIA) / 1.5)) * (dist < DIST_MAX) * (dist > 0.2)
                giro, conf = p["cab_giro"][k], p["cab_conf"][k]
                if giro is not None and conf > 0:
                    dif = np.angle(np.exp(1j * (np.arctan2(P[ok, 1] - O[1], P[ok, 0] - O[0]) - giro)))
                    c = PESO_GIRO * conf
                    nota *= (1 - c) + c * np.exp(-dif ** 2 / (2 * SIGMA_GIRO ** 2))
                vivos = np.where(nota > 0.02)[0]
                if len(vivos):
                    itens = [cena.faces[face[ok[v]]].item for v in vivos]
                    esc = cena.bloqueado(np.repeat(O[None], len(vivos), 0), P[ok[vivos]], ignora=itens, folga=0.15)
                    nota[vivos[esc]] = 0
                for j, i in enumerate(ok):
                    if nota[j] > 0.02:
                        alvos.append([int(face[i]), round(float(S1[i]), 3), round(float(S2[i]), 3),
                                      round(float(massa[i] * nota[j] / total), 5), *np.round(P[i], 3).tolist()])
        amostras.append({"q": q, "pessoa": p["id"], "O": np.round(O, 3).tolist(), "dt": dt, "inout": g["inout"],
                         "fracao": round(sum(a[3] for a in alvos), 4), "pico": g["xy"], "perdas": perdas, "alvos": alvos})
    return amostras
