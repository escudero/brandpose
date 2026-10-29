"""Estatística da atenção: por face, por móvel, por altura da prateleira e mapa de calor de cada face.

Unidade: segundos de atenção = tempo da amostra × chance de o alvo estar na imagem × fração do
mapa do olhar que caiu na face. Só clientes (a equipe atrás do balcão fica de fora).

Intervalos de 95%: bootstrap por pessoa (sorteia os clientes com reposição, 2000 vezes). A
pergunta que ele responde: "com outros clientes parecidos, quanto isso variaria?". Com poucos
clientes, o intervalo é largo, e é para ser: avisa quando o número não sustenta uma conclusão.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

CEL = 0.05  # mapa de calor em células de 5 cm
FAIXAS = [("Chão (0–50 cm)", 0.0, 0.5), ("Mãos (50–110 cm)", 0.5, 1.1),
          ("Olhos (110–160 cm)", 1.1, 1.6), ("Alto (160 cm ou mais)", 1.6, 9.0)]
TAMPO = "Tampo (de cima)"
MIN_OLHOU_S = 0.3  # pessoa "olhou" a face se somou ao menos 0,3 s nela
B = 2000


def faixa(face, s2):
    if face.lado == "topo":
        return len(FAIXAS)
    return next(i for i, (_, a, b) in enumerate(FAIXAS) if a <= s2 < b) if s2 < 9 else len(FAIXAS) - 1


def ic(boot, eixo=0):
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=eixo)
    return lo, hi


def calcular(cena, pessoas, amostras, duracao_s, semente=7):
    clientes = [p["id"] for p in pessoas if not p["equipe"]]
    ip = {pid: i for i, pid in enumerate(clientes)}
    F, NB = len(cena.faces), len(FAIXAS) + 1
    A = np.zeros((len(clientes), F))          # segundos por pessoa × face
    Bf = np.zeros((len(clientes), NB))        # segundos por pessoa × faixa de altura
    olhar_total = np.zeros(len(clientes))     # segundos com o olhar estimado (qualquer alvo)
    destino = {"gondolas": 0.0, "fora_ou_gente": 0.0, "chao": 0.0, "sem_movel": 0.0}
    mapas = [np.zeros((max(1, int(np.ceil(f.w2 / CEL))), max(1, int(np.ceil(f.w1 / CEL))))) for f in cena.faces]
    for a in amostras:
        if a["pessoa"] not in ip:
            continue
        i = ip[a["pessoa"]]
        peso = a["dt"] * a["inout"]
        olhar_total[i] += a["dt"]
        destino["gondolas"] += a["dt"] * a["fracao"]
        for k_, v_ in a.get("perdas", {}).items():
            destino[k_] += a["dt"] * v_
        for f, s1, s2, w, *_ in a["alvos"]:
            seg = peso * w
            face = cena.faces[f]
            A[i, f] += seg
            Bf[i, faixa(face, s2)] += seg
            m = mapas[f]
            r = min(m.shape[0] - 1, max(0, int(s2 / CEL)))
            c = min(m.shape[1] - 1, max(0, int(s1 / CEL)))
            m[r, c] += seg

    rng = np.random.default_rng(semente)
    sorteio = rng.integers(0, len(clientes), (B, len(clientes))) if clientes else np.zeros((B, 0), int)
    bootA = A[sorteio].sum(1)                                   # B × F
    bootB = Bf[sorteio].sum(1)                                  # B × NB
    tot = A.sum()
    share_boot = bootA / np.maximum(bootA.sum(1, keepdims=True), 1e-9)
    faixa_boot = bootB / np.maximum(bootB.sum(1, keepdims=True), 1e-9)
    s_lo, s_hi = ic(bootA)
    p_lo, p_hi = ic(share_boot)
    f_lo, f_hi = ic(faixa_boot)

    faces = []
    for k, f in enumerate(cena.faces):
        seg = A[:, k].sum()
        olharam = int((A[:, k] >= MIN_OLHOU_S).sum())
        faces.append({
            "face": f.id, "movel": f.item, "nome": f"{cena.nome(f.item)} · {'frente' if f.frente else f.lado}",
            "tipo": f.tipo, "lado": f.lado, "frente": f.frente, "largura_m": round(f.w1, 2), "altura_m": round(f.w2, 2),
            "segundos": round(float(seg), 2), "ic_segundos": [round(float(s_lo[k]), 2), round(float(s_hi[k]), 2)],
            "parcela": round(float(seg / tot), 4) if tot else 0.0,
            "ic_parcela": [round(float(p_lo[k]), 4), round(float(p_hi[k]), 4)],
            "pessoas": olharam,
            "por_pessoa_s": round(float(seg / olharam), 2) if olharam else 0.0,
            "por_metro_s": round(float(seg / max(f.w1, 0.1)), 2),
        })
    faces.sort(key=lambda r: -r["segundos"])

    moveis = {}
    for k, f in enumerate(cena.faces):
        moveis.setdefault(f.item, []).append(k)
    mov = []
    for item, ks in moveis.items():
        seg_b = bootA[:, ks].sum(1)
        mov.append({"movel": item, "nome": cena.nome(item), "tipo": cena.faces[ks[0]].tipo,
                    "segundos": round(float(A[:, ks].sum()), 2),
                    "ic_segundos": [round(float(v), 2) for v in np.percentile(seg_b, [2.5, 97.5])],
                    "parcela": round(float(A[:, ks].sum() / tot), 4) if tot else 0.0,
                    "ic_parcela": [round(float(v), 4) for v in np.percentile(seg_b / np.maximum(bootA.sum(1), 1e-9), [2.5, 97.5])],
                    "pessoas": int((A[:, ks].sum(1) >= MIN_OLHOU_S).sum())})
    mov.sort(key=lambda r: -r["segundos"])

    nomes_faixa = [n for n, *_ in FAIXAS] + [TAMPO]
    faixas = [{"faixa": nomes_faixa[j], "segundos": round(float(Bf[:, j].sum()), 2),
               "parcela": round(float(Bf[:, j].sum() / max(Bf.sum(), 1e-9)), 4),
               "ic_parcela": [round(float(f_lo[j]), 4), round(float(f_hi[j]), 4)]}
              for j in range(NB) if Bf[:, j].sum() > 0 or j < len(FAIXAS)]

    # a melhor face é mesmo a melhor? fração dos sorteios em que ela fica em 1º
    top = int(np.argmax(A.sum(0))) if tot else 0
    lider = float((bootA.argmax(1) == top).mean()) if tot else 0.0

    resumo = {
        "duracao_s": round(duracao_s, 1), "clientes": len(clientes), "equipe": sum(p["equipe"] for p in pessoas),
        "olhar_estimado_s": round(float(olhar_total.sum()), 1), "atencao_gondolas_s": round(float(tot), 1),
        "fracao_em_gondolas": round(float(tot / max(olhar_total.sum(), 1e-9)), 3),
        "faces": len(cena.faces), "amostras": len(amostras),
        "face_lider": cena.faces[top].id if tot else None, "chance_lider_continuar_1a": round(lider, 3),
        # para onde foi o mapa do olhar (fração do tempo); "sem_movel" alto = falta móvel no modelo da loja
        "destino_do_olhar": {k: round(v / max(olhar_total.sum(), 1e-9), 3) for k, v in destino.items()}
                            | {"descartado_pelas_regras": round(1 - sum(destino.values()) / max(olhar_total.sum(), 1e-9), 3)},
    }
    por_pessoa = [{"pessoa": pid, "olhar_s": round(float(olhar_total[i]), 2), "gondolas_s": round(float(A[i].sum()), 2),
                   "faces_olhadas": int((A[i] >= MIN_OLHOU_S).sum())} for pid, i in ip.items()]
    mapas = [gaussian_filter(m, 1.5) for m in mapas]
    return {"resumo": resumo, "faces": faces, "moveis": mov, "faixas": faixas, "por_pessoa": por_pessoa}, mapas
