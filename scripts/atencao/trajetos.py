"""Trajeto de cada pessoa no piso da loja e a direção da cabeça, quadro a quadro.

1. Pé no piso: tornozelos (z = 8 cm); sem eles, nariz a 1,57 m; sem nariz, ombros a 1,40 m.
2. Altura dos olhos de cada pessoa: onde o raio do nariz cruza a vertical sobre os pés (mediana).
3. Pedaços do mesmo cliente com números diferentes (o rastreador perde e acha de novo) são
   juntados: um termina, outro começa logo depois, perto o bastante para ter andado até lá.
4. Buracos de até 1 s são preenchidos; o trajeto é suavizado; pé dentro de móvel vai para a borda.
5. Equipe: quem passa mais da metade do tempo atrás do balcão/caixa não conta como cliente.
6. Cabeça: giro em relação à câmera pelo nariz em relação às orelhas (nariz ~10 cm à frente);
   só orelhas à vista = de costas. Serve de "palpite" para escolher entre alvos possíveis do olhar.

Saída: trajetos.json
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

NARIZ_Z, OMBRO_Z, TORNOZELO_Z = 1.57, 1.40, 0.08
BURACO_S = 1.0
JUNTA_S = 2.0          # junta pedaços separados por até 2 s
MIN_S = 1.0            # trajetos mais curtos são ruído
ATENDIMENTO = {"balcao", "checkout", "balcao_atendimento"}


def _pe(cena, p, H):
    kp, kc = np.array(p["kp"]), np.array(p["kc"])
    if kc[15] > 0.5 and kc[16] > 0.5 and p["box"][3] < H - 6:
        return cena.ao_piso(((kp[15] + kp[16]) / 2)[None], TORNOZELO_Z)[0][:2], 0
    if kc[0] > 0.6:
        return cena.ao_piso(kp[0][None], NARIZ_Z)[0][:2], 1
    if kc[5] > 0.5 and kc[6] > 0.5:
        return cena.ao_piso(((kp[5] + kp[6]) / 2)[None], OMBRO_Z)[0][:2], 2
    return None, -1


def _altura_nariz(cena, nariz_uv, pe_xy):
    """Ponto do raio do nariz mais perto da vertical sobre o pé: a altura do nariz."""
    r = cena.raios(nariz_uv[None])[0]
    rh = r[:2] / np.linalg.norm(r[:2])
    t = ((pe_xy - cena.C[:2]) @ rh) / np.linalg.norm(r[:2])
    return cena.C[2] + t * r[2]


def _giro_cabeca(cena, kp, kc, pe_xy, z):
    """Giro da cabeça no piso (rad, a partir de +x) e confiança, ou (nan, 0)."""
    nariz = kc[0] > 0.5
    orelhas = [j for j in (3, 4) if kc[j] > 0.5]
    if not orelhas:
        return math.nan, 0.0
    p3 = np.array([*pe_xy, z])
    para_cam = cena.C[:2] - pe_xy
    para_cam = para_cam / np.linalg.norm(para_cam)
    lat = np.array([para_cam[1], -para_cam[0], 0.0])
    e = 0.05
    q0, ql, qv = cena.projetar(np.array([p3, p3 + e * lat, p3 + [0, 0, e]]))
    Ji = np.linalg.inv(np.stack([(ql - q0) / e, (qv - q0) / e], 1))
    yaw_cam = math.atan2(para_cam[1], para_cam[0])
    if nariz:
        l = (Ji @ (kp[0] - kp[orelhas].mean(0)))[0]
        if len(orelhas) == 2:
            a = math.asin(float(np.clip(l / 0.10, -1, 1)))
            conf = 0.8 if (kc[1] > 0.5 and kc[2] > 0.5) else 0.5
        else:
            a, conf = math.copysign(math.radians(80), l), 0.6
    elif len(orelhas) == 2:
        a, conf = math.pi, 0.5
    else:
        return math.nan, 0.0
    return yaw_cam + a, conf


def _suave(v, sig, ang=False):
    v = np.array(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return v
    t = np.arange(len(v))
    u = np.unwrap(v[ok]) if ang else v[ok]
    return gaussian_filter1d(np.interp(t, t[ok], u), sig, mode="nearest")


def calcular(cena, frames, fps, W, H):
    # ---------- 1. medidas por detecção ----------
    med = {}
    alturas = {}
    for q, fr in enumerate(frames):
        for p in fr:
            xy, fonte = _pe(cena, p, H)
            if xy is None or not np.isfinite(xy).all() or np.linalg.norm(xy) > 30:
                continue
            kp, kc = np.array(p["kp"]), np.array(p["kc"])
            med.setdefault(p["id"], {})[q] = {"xy": xy, "fonte": fonte, "kp": kp, "kc": kc}
            if fonte == 0 and kc[0] > 0.7:
                alturas.setdefault(p["id"], []).append(_altura_nariz(cena, kp[0], xy))

    # ---------- 3. junta pedaços do mesmo cliente ----------
    span = {i: (min(m), max(m)) for i, m in med.items() if len(m) >= 3}
    pai = {i: i for i in span}
    raiz = lambda i: i if pai[i] == i else raiz(pai[i])
    usados_ini = set()
    for i in sorted(span, key=lambda i: span[i][1]):
        fim = span[i][1]
        xy_fim = med[i][fim]["xy"]
        melhor = None
        for j in span:
            if j == i or j in usados_ini or raiz(j) == raiz(i):
                continue
            ini = span[j][0]
            dt = (ini - fim) / fps
            if not (0 < dt <= JUNTA_S):
                continue
            # o pedaço novo não pode coexistir com nenhum quadro do grupo de i
            if any(span[k][1] >= ini for k in span if raiz(k) == raiz(i)):
                continue
            dist = np.linalg.norm(med[j][ini]["xy"] - xy_fim)
            if dist < 0.6 + 1.5 * dt and (melhor is None or dist < melhor[0]):
                melhor = (dist, j)
        if melhor:
            pai[melhor[1]] = raiz(i)
            usados_ini.add(melhor[1])
    grupos = {}
    for i in span:
        grupos.setdefault(raiz(i), []).append(i)

    # ---------- 2, 4, 5, 6 ----------
    pessoas = []
    for g, ids in sorted(grupos.items()):
        m = {}
        for i in ids:
            m.update(med[i])
        qs = sorted(m)
        trechos, atual = [], [qs[0]]
        for q in qs[1:]:
            if q - atual[-1] > BURACO_S * fps:
                trechos.append(atual); atual = [q]
            else:
                atual.append(q)
        trechos.append(atual)
        alt = [a for i in ids for a in alturas.get(i, [])]
        z_nariz = float(np.clip(np.median(alt), 1.35, 1.85)) if len(alt) >= 5 else NARIZ_Z
        for k, tr in enumerate(trechos):
            q0, q1 = tr[0], tr[-1]
            n = q1 - q0 + 1
            if n < MIN_S * fps:
                continue
            xy = np.full((n, 2), np.nan)
            for q in tr:
                xy[q - q0] = m[q]["xy"]
            # pulo de posição (troca entre tornozelo e nariz, oclusão): fora da mediana local
            med_loc = np.array([np.nanmedian(xy[max(0, i - 8):i + 9], 0) for i in range(n)])
            xy[np.linalg.norm(xy - med_loc, axis=1) > 0.35] = np.nan
            x, y = _suave(xy[:, 0], 5), _suave(xy[:, 1], 5)
            xy = cena.empurrar_para_fora(np.c_[x, y])
            x, y = gaussian_filter1d(xy[:, 0], 2), gaussian_filter1d(xy[:, 1], 2)
            vel = gaussian_filter1d(np.hypot(np.gradient(x), np.gradient(y)) * fps, 4)
            giro, conf = np.full(n, np.nan), np.zeros(n)
            for q in tr:
                d = m[q]
                giro[q - q0], conf[q - q0] = _giro_cabeca(cena, d["kp"], d["kc"], np.array([x[q - q0], y[q - q0]]),
                                                          z_nariz)
            giro_s = _suave(giro, 3, ang=True) if np.isfinite(giro).any() else giro
            staff = float(cena.lado_de_tras(np.c_[x, y], ATENDIMENTO).mean()) > 0.5
            pessoas.append({
                "id": f"{g}" + (f".{k}" if len(trechos) > 1 else ""), "ids_rastreador": sorted(ids),
                "equipe": staff, "q0": q0, "z_olhos": round(z_nariz + 0.03, 3),
                "x": np.round(x, 3).tolist(), "y": np.round(y, 3).tolist(), "vel": np.round(vel, 2).tolist(),
                "cab_giro": [None if not np.isfinite(v) else round(float(v), 3) for v in giro_s],
                "cab_conf": np.round(gaussian_filter1d(conf, 3), 2).tolist(),
                "medido": [q - q0 for q in tr],
            })
    rid = {i: p["id"] for p in pessoas for i in p["ids_rastreador"]}
    return pessoas, rid
