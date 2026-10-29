"""A loja em 3D vinda do LojaSim: câmera calibrada sem cliques e móveis montados (plano_final.json).

O LojaSim (projeto irmão, ../LojaSim) faz, a partir do mesmo vídeo, a câmera (altura, ângulos, lente)
e a planta da loja: cada móvel é uma caixa (x, y, L, d, H, giro), com o tipo, a frente e quantos
lados têm produto. Aqui cada caixa vira:
- um obstáculo (para saber o que a pessoa enxerga e o que esconde o pé);
- faces de exposição: os lados com produto (frente; as costas também se tiver 2 lados) e,
  nos móveis baixos sem frente (mesa, freezer baú, pallet), os 4 lados e o tampo.

Cada face tem um sistema próprio: s1 = metros da esquerda para a direita, como vê quem está de
frente para ela; s2 = altura do chão (no tampo, metros da borda da frente para o fundo).
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOJASIM = Path(os.environ.get("LOJASIM", Path(__file__).resolve().parents[3] / "LojaSim"))
sys.path.insert(0, str(LOJASIM / "pipeline"))
import camera as _cam  # noqa: E402  (modelo de câmera do LojaSim)

SEM_EXPOSICAO = {"porta", "janela", "parede", "coluna", "lixeira", "desconhecido", "carrinhos"}
TODOS_OS_LADOS = {"mesa", "freezer_horizontal", "pallet", "cesto", "torre_giratoria", "arara", "banca_frutas"}
COM_TAMPO = {"mesa", "freezer_horizontal", "pallet", "cesto", "banca_frutas", "display_chao"}
NOME_LADO = {"+x": "lado +x", "-x": "lado -x", "+y": "lado +y", "-y": "lado -y", "topo": "tampo"}


@dataclass
class Face:
    id: str
    item: str
    tipo: str
    lado: str          # +x, -x, +y, -y ou topo
    frente: bool       # é a frente do móvel
    c: np.ndarray      # centro
    n: np.ndarray      # normal para fora
    a1: np.ndarray     # eixo de s1 (esquerda → direita de quem olha)
    a2: np.ndarray     # eixo de s2 (para cima; no tampo, da frente para o fundo)
    w1: float
    w2: float
    z0: float = 0.0    # s2 do canto de baixo (0 nas faces verticais)
    extra: dict = field(default_factory=dict)

    def canto(self, s1, s2):
        return self.c + (s1 - self.w1 / 2) * self.a1 + (s2 - self.z0 - self.w2 / 2) * self.a2


class Cena:
    def __init__(self, cena_dir, dados_dir):
        self.cena_dir, self.dados_dir = Path(cena_dir), Path(dados_dir)
        self.cam = json.loads((self.cena_dir / "camera_final.json").read_text())
        self.C = np.array([0.0, 0.0, self.cam["h"]])
        plano = json.loads((self.dados_dir / "plano_final.json").read_text())
        self.itens = plano["itens"]
        self.catalogo = json.loads((LOJASIM / "editor" / "catalogo.json").read_text())["tipos"]
        self.caixas = self._caixas()
        self.faces = self._faces()
        # pontos 3D de cada pixel da imagem sem distorção (profundidade com a escala dos pés) e o piso
        W, H = self.cam["W"], self.cam["H"]
        self.pontos = np.load(self.cena_dir / "pontos.npy", mmap_mode="r")
        f = self.cena_dir / "piso.npy"
        self.piso = np.unpackbits(np.load(f))[:H * W].reshape(H, W).astype(bool) if f.exists() else np.zeros((H, W), bool)

    def sem_distorcao(self, uv):
        """Pixel do vídeo (com a distorção da lente) → pixel da imagem sem distorção (pontos.npy, piso.npy)."""
        return _cam.project({**self.cam, "k1": 0.0}, self.C + self.raios(uv))

    def na_face(self, P, k):
        """Ponto 3D → (s1, s2, distância ao retângulo da face k)."""
        f = self.faces[k]
        rel = P - f.c
        u, v, w = rel @ f.a1, rel @ f.a2, rel @ f.n
        uc, vc = np.clip(u, -f.w1 / 2, f.w1 / 2), np.clip(v, -f.w2 / 2, f.w2 / 2)
        return uc + f.w1 / 2, vc + f.w2 / 2, np.sqrt((u - uc) ** 2 + (v - vc) ** 2 + w ** 2)

    def na_faces(self, P):
        """Ponto 3D → (s1, s2, distância) para todas as faces de uma vez."""
        if not hasattr(self, "_F"):
            fs = self.faces
            self._F = [np.array([getattr(f, a) for f in fs]) for a in ("c", "a1", "a2", "n", "w1", "w2")]
        Cc, A1, A2, N, W1, W2 = self._F
        rel = P - Cc
        u, v, w = (rel * A1).sum(1), (rel * A2).sum(1), (rel * N).sum(1)
        uc, vc = np.clip(u, -W1 / 2, W1 / 2), np.clip(v, -W2 / 2, W2 / 2)
        return uc + W1 / 2, vc + W2 / 2, np.sqrt((u - uc) ** 2 + (v - vc) ** 2 + w ** 2)

    # ---------- câmera ----------
    def projetar(self, P):
        return _cam.project(self.cam, P)

    def raios(self, uv):
        r = _cam.rays(self.cam, uv)
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def ao_piso(self, uv, z):
        return _cam.unproject(self.cam, uv, z)

    # ---------- móveis ----------
    def nome(self, item_id):
        it = next(i for i in self.itens if i["id"] == item_id)
        t = self.catalogo.get(it["tipo"], {})
        n = sum(1 for i in self.itens if i["tipo"] == it["tipo"] and i["id"] <= item_id)
        return f"{t.get('nome', it['tipo'])} {n}"

    def _caixas(self):
        cx = []
        for it in self.itens:
            a = math.radians(it["rot"])
            cx.append({"id": it["id"], "tipo": it["tipo"], "c": np.array([it["x"], it["y"]]),
                       "R": np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]),
                       "L": it["L"], "d": it["d"], "H": it["H"]})
        return cx

    def _faces(self):
        faces = []
        for it, b in zip(self.itens, self.caixas):
            if it["tipo"] in SEM_EXPOSICAO:
                continue
            t = self.catalogo.get(it["tipo"], {})
            frente = it.get("frente") or it.get("frente_auto")
            if it["tipo"] in TODOS_OS_LADOS or not (t.get("frente") and frente):
                lados = ["+x", "-x", "+y", "-y"]
            else:
                oposto = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}[frente]
                lados = [frente] + ([oposto] if it.get("lados") == 2 else [])
            for lado in lados:
                eixo = 0 if lado[1] == "x" else 1
                sinal = 1.0 if lado[0] == "+" else -1.0
                n2 = b["R"][:, eixo] * sinal
                meio = (b["L"] if eixo == 0 else b["d"]) / 2
                larg = b["d"] if eixo == 0 else b["L"]
                n = np.array([n2[0], n2[1], 0.0])
                a1 = np.cross([0, 0, 1.0], n)
                c = np.array([*(b["c"] + n2 * meio), b["H"] / 2])
                faces.append(Face(f"{it['id']}:{lado}", it["id"], it["tipo"], lado, lado == frente,
                                  c, n, a1, np.array([0, 0, 1.0]), larg, b["H"]))
            if it["tipo"] in COM_TAMPO:
                # tampo: s1 ao longo do comprimento, s2 da borda -y local para a +y local
                a1 = np.array([*b["R"][:, 0], 0.0])
                a2 = np.array([*b["R"][:, 1], 0.0])
                faces.append(Face(f"{it['id']}:topo", it["id"], it["tipo"], "topo", False,
                                  np.array([*b["c"], b["H"]]), np.array([0, 0, 1.0]), a1, a2, b["L"], b["d"],
                                  z0=0.0))
        return faces

    # ---------- geometria ----------
    def bate(self, O, D, faces=None, tmax=40.0):
        """Raios O + t·D contra as faces. Devolve (t, s1, s2) com t = inf onde não bate. Formas (R, F)."""
        faces = faces if faces is not None else self.faces
        O, D = np.atleast_2d(O), np.atleast_2d(D)
        N = np.array([f.n for f in faces]); Cc = np.array([f.c for f in faces])
        A1 = np.array([f.a1 for f in faces]); A2 = np.array([f.a2 for f in faces])
        W1 = np.array([f.w1 for f in faces]); W2 = np.array([f.w2 for f in faces])
        den = D @ N.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((Cc * N).sum(1)[None] - O @ N.T) / den
        P = O[:, None] + t[..., None] * D[:, None]
        rel = P - Cc[None]
        u = (rel * A1[None]).sum(2)
        v = (rel * A2[None]).sum(2)
        ok = (np.abs(den) > 1e-6) & (t > 0.05) & (t < tmax) & (np.abs(u) <= W1 / 2) & (np.abs(v) <= W2 / 2)
        t = np.where(ok, t, np.inf)
        return t, u + W1 / 2, v + W2 / 2

    def bloqueado(self, A, B, ignora=None, folga=0.03):
        """O segmento A→B (N×3) atravessa alguma caixa? (ignora: id do móvel de cada segmento)."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        res = np.zeros(len(A), bool)
        for b in self.caixas:
            if b["tipo"] in ("porta", "janela"):
                continue
            Ri = b["R"].T
            a = np.c_[(A[:, :2] - b["c"]) @ Ri.T, A[:, 2]]
            d = np.c_[(B[:, :2] - A[:, :2]) @ Ri.T, B[:, 2] - A[:, 2]]
            lo = np.array([-b["L"] / 2 + folga, -b["d"] / 2 + folga, folga])
            hi = np.array([b["L"] / 2 - folga, b["d"] / 2 - folga, b["H"] - folga])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - a) / d, (hi - a) / d
            tmin = np.nanmax(np.minimum(t1, t2), 1)
            tmax = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tmax >= tmin) & (tmax > 0.02) & (tmin < 0.98)
            if ignora is not None:
                hit &= np.asarray(ignora) != b["id"]
            res |= hit
        return res

    def dentro(self, xy, margem=0.0, tipos_fora=("porta", "janela")):
        """Índice do móvel que contém cada ponto do piso (−1 se nenhum)."""
        xy = np.atleast_2d(xy)
        idx = np.full(len(xy), -1)
        for k, b in enumerate(self.caixas):
            if b["tipo"] in tipos_fora:
                continue
            p = (xy - b["c"]) @ b["R"]
            m = (np.abs(p[:, 0]) < b["L"] / 2 + margem) & (np.abs(p[:, 1]) < b["d"] / 2 + margem)
            idx[m & (idx < 0)] = k
        return idx

    def empurrar_para_fora(self, xy, margem=0.15):
        """Pés dentro de um móvel vão para a borda mais próxima (+ margem)."""
        xy = np.array(xy, float)
        for _ in range(3):
            k = self.dentro(xy, margem * 0.5)
            if (k < 0).all():
                break
            for i in np.where(k >= 0)[0]:
                b = self.caixas[k[i]]
                p = (xy[i] - b["c"]) @ b["R"]
                folgas = [b["L"] / 2 - p[0], b["L"] / 2 + p[0], b["d"] / 2 - p[1], b["d"] / 2 + p[1]]
                j = int(np.argmin(folgas))
                if j == 0: p[0] = b["L"] / 2 + margem
                elif j == 1: p[0] = -b["L"] / 2 - margem
                elif j == 2: p[1] = b["d"] / 2 + margem
                else: p[1] = -b["d"] / 2 - margem
                xy[i] = b["c"] + b["R"] @ p
        return xy

    def lado_de_tras(self, xy, tipos, prof=1.6):
        """O ponto está atrás (lado oposto à frente) de algum móvel desses tipos, a até `prof` m?"""
        xy = np.atleast_2d(xy)
        res = np.zeros(len(xy), bool)
        for it, b in zip(self.itens, self.caixas):
            fr = it.get("frente") or it.get("frente_auto")
            if it["tipo"] not in tipos or not fr:
                continue
            p = (xy - b["c"]) @ b["R"]
            eixo, s = (0 if fr[1] == "x" else 1), (1 if fr[0] == "+" else -1)
            meio, larg = (b["L"], b["d"]) if eixo == 0 else (b["d"], b["L"])
            atras = -s * p[:, eixo] - meio / 2
            res |= (atras > -0.1) & (atras < prof) & (np.abs(p[:, 1 - eixo]) < larg / 2 + 0.5)
        return res
