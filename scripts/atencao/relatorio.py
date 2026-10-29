"""Relatório da atenção nas gôndolas: um HTML só (imagens embutidas) a partir do atencao.json.

- resumo e para onde foi o olhar (gôndolas, chão, fora da imagem/gente, onde falta móvel no modelo);
- ranking por móvel e por face, com intervalo de 95% (bootstrap por cliente);
- atenção por altura da prateleira;
- mapa de calor de cada face: sobre a foto da própria face (retificada da imagem da câmera) quando
  a câmera vê aquele lado; senão, sobre um desenho da face;
- planta de cima com as faces coloridas pela atenção.

Uso: python relatorio.py ANALISE_DIR --cena C --loja L [--top 8]
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cena3d  # noqa: E402

PX_M = 200  # pixels por metro na foto retificada da face


def b64(fig_ou_img):
    if isinstance(fig_ou_img, np.ndarray):
        ok, buf = cv2.imencode(".jpg", fig_ou_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    b = io.BytesIO()
    fig_ou_img.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig_ou_img)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def foto_da_face(cena, f, foto):
    """A face vista pela câmera, retificada (s1 → x, altura → y). None se a câmera vê o outro lado."""
    if f.n @ (cena.C - f.c) <= 0:
        return None
    w, h = max(2, int(f.w1 * PX_M)), max(2, int(f.w2 * PX_M))
    s1, s2 = np.meshgrid((np.arange(w) + 0.5) / PX_M, f.w2 - (np.arange(h) + 0.5) / PX_M)
    P = f.c + (s1 - f.w1 / 2)[..., None] * f.a1 + (s2 - f.w2 / 2)[..., None] * f.a2
    uv = cena.projetar(P.reshape(-1, 3)).reshape(h, w, 2).astype(np.float32)
    return cv2.remap(foto, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def desenho_da_face(f):
    w, h = max(2, int(f.w1 * PX_M)), max(2, int(f.w2 * PX_M))
    img = np.full((h, w, 3), 225, np.uint8)
    if f.lado != "topo":
        for z in np.arange(0.36, f.w2, 0.36):  # prateleira a cada ~36 cm (como no LojaSim)
            y = int(h - z * PX_M)
            cv2.line(img, (0, y), (w, y), (170, 170, 170), 2)
    return img


def mapa_sobre(base, mapa, vmax, alturas=True):
    h, w = base.shape[:2]
    m = cv2.resize(np.flipud(mapa), (w, h), interpolation=cv2.INTER_CUBIC)
    m = np.clip(m / max(vmax, 1e-9), 0, 1)
    cor = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = (np.clip(m * 1.6, 0, 0.85))[..., None]
    out = (base * (1 - a) + cor * a).astype(np.uint8)
    if alturas:
        for z in (0.5, 1.1, 1.6):
            y = int(h - z * PX_M)
            if 0 < y < h:
                cv2.line(out, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(out, f"{int(z * 100)} cm", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def barras(linhas, titulo, chave="segundos", ic="ic_segundos", unidade="s"):
    linhas = [l for l in linhas if l[chave] > 0][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(linhas) + 1))
    v = np.array([l[chave] for l in linhas])
    lo, hi = np.array([l[ic][0] for l in linhas]), np.array([l[ic][1] for l in linhas])
    if unidade == "%":
        v, lo, hi = v * 100, lo * 100, hi * 100
    ax.barh(range(len(linhas)), v, color="#f28e2b")
    ax.errorbar(v, range(len(linhas)), xerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], fmt="none", ecolor="#333", capsize=3)
    ax.set_yticks(range(len(linhas)), [l.get("nome") or l.get("faixa") for l in linhas])
    ax.set_xlabel(f"atenção ({unidade}) · barra preta = intervalo de 95%")
    ax.set_title(titulo, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def planta(cena, res, trajetos):
    seg = {f["face"]: f["segundos"] for f in res["faces"]}
    vmax = max(seg.values()) or 1
    fig, ax = plt.subplots(figsize=(7, 7))
    for p in trajetos:
        if not p["equipe"]:
            ax.plot(p["x"], p["y"], color="#9ecae1", lw=0.8, alpha=0.7)
    for b in cena.caixas:
        cs = np.array([b["c"] + b["R"] @ np.array([sx * b["L"] / 2, sy * b["d"] / 2]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        ax.fill(cs[:, 0], cs[:, 1], color="#bbb", ec="#888")
        ax.text(*b["c"], cena.nome(b["id"]), fontsize=6, ha="center", va="center")
    cmap = plt.get_cmap("inferno")
    for f in cena.faces:
        if f.lado == "topo":
            continue
        a, b = f.canto(0, 0), f.canto(f.w1, 0)
        v = seg.get(f.id, 0) / vmax
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(0.15 + 0.8 * v), lw=2 + 7 * v, solid_capstyle="butt")
    ax.plot(0, 0, "k^", ms=10)
    ax.text(0, 0, "  câmera", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title("Planta: faces mais grossas e claras = mais atenção; azul = trajetos", loc="left", fontsize=10)
    ax.axis("off")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analise", type=Path)
    ap.add_argument("--cena", type=Path, required=True)
    ap.add_argument("--loja", type=Path, required=True)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    cena = cena3d.Cena(args.cena, args.loja)
    res = json.loads((args.analise / "atencao.json").read_text())
    trajetos = json.loads((args.analise / "trajetos.json").read_text())["pessoas"]
    mapas = np.load(args.analise / "mapas.npz")
    foto = cv2.imread(str(args.cena / "clean_plate.png"))
    r = res["resumo"]
    face_por_id = {f.id: f for f in cena.faces}

    dest = r["destino_do_olhar"]
    nomes_dest = {"gondolas": "Gôndolas e móveis do modelo", "fora_ou_gente": "Fora da imagem ou em outra pessoa",
                  "chao": "Chão", "sem_movel": "Superfície sem móvel no modelo", "descartado_pelas_regras": "Descartado pelas regras (longe, atrás de móvel, contra a cabeça)"}
    avisos = []
    if dest["sem_movel"] > 0.2:
        avisos.append(f"{dest['sem_movel']:.0%} do olhar caiu em superfícies que não estão no modelo da loja. "
                      "Confira no editor do LojaSim se faltam móveis (ou se estão fora do lugar) e rode de novo.")
    if r["clientes"] < 30:
        avisos.append(f"Só {r['clientes']} clientes: os intervalos são largos. Use as diferenças grandes, não a ordem fina do ranking.")
    if r["chance_lider_continuar_1a"] < 0.8:
        avisos.append(f"A face em 1º lugar só fica em 1º em {r['chance_lider_continuar_1a']:.0%} dos sorteios: o 1º lugar não está garantido.")

    blocos = []
    for f in res["faces"][: args.top]:
        if f["segundos"] <= 0:
            continue
        geo = face_por_id[f["face"]]
        base = foto_da_face(cena, geo, foto)
        origem = "foto da câmera" if base is not None else "desenho (a câmera não vê este lado)"
        base = base if base is not None else desenho_da_face(geo)
        img = mapa_sobre(base, mapas[f["face"]], mapas[f["face"]].max(), geo.lado != "topo")
        blocos.append(f"""<div class="face"><img src="{b64(img)}"><div><b>{html.escape(f['nome'])}</b><br>
          {f['segundos']:.1f} s ({f['ic_segundos'][0]:.1f}–{f['ic_segundos'][1]:.1f}) · {f['parcela']:.0%} da atenção ·
          {f['pessoas']} pessoa(s) · {f['largura_m']:.2f} × {f['altura_m']:.2f} m<br><small>fundo: {origem}</small></div></div>""")

    linhas_dest = "".join(f"<tr><td>{nomes_dest[k]}</td><td>{v:.0%}</td></tr>" for k, v in dest.items())
    tab = "".join(f"<tr><td>{html.escape(f['nome'])}</td><td>{f['segundos']:.1f}</td><td>{f['ic_segundos'][0]:.1f}–{f['ic_segundos'][1]:.1f}</td>"
                  f"<td>{f['parcela']:.1%}</td><td>{f['ic_parcela'][0]:.0%}–{f['ic_parcela'][1]:.0%}</td><td>{f['pessoas']}</td>"
                  f"<td>{f['por_pessoa_s']:.1f}</td><td>{f['por_metro_s']:.1f}</td></tr>" for f in res["faces"] if f["segundos"] > 0)
    doc = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Atenção nas gôndolas</title>
<style>body{{font:15px/1.45 system-ui;margin:28px auto;max-width:1100px;color:#222}} h1{{margin-bottom:4px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap}} .kpi{{background:#f6f6f6;border-radius:10px;padding:12px 16px}} .kpi b{{font-size:22px;display:block}}
.aviso{{background:#fff4d6;border-left:4px solid #f2b705;padding:8px 12px;margin:6px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #eee;padding:4px 6px;text-align:left}}
.face{{display:flex;gap:14px;align-items:center;margin:12px 0}} .face img{{max-height:260px;max-width:560px;border-radius:6px}}
.duas{{display:flex;gap:16px;flex-wrap:wrap}} .duas img{{max-width:540px}}</style>
<h1>Onde os clientes olham nas gôndolas</h1>
<p>Vídeo de {r['duracao_s']:.0f} s · {r['clientes']} clientes ({r['equipe']} da equipe fora da conta) · {len(res['faces'])} faces de exposição · {r['amostras']} olhares estimados</p>
<div class="kpis"><div class="kpi"><b>{r['atencao_gondolas_s']:.1f} s</b>de atenção em gôndolas</div>
<div class="kpi"><b>{r['fracao_em_gondolas']:.0%}</b>do tempo olhando, nas gôndolas</div>
<div class="kpi"><b>{html.escape(res['moveis'][0]['nome']) if res['moveis'] else '—'}</b>móvel mais olhado</div>
<div class="kpi"><b>{res['faixas'][int(np.argmax([f['parcela'] for f in res['faixas']]))]['faixa']}</b>altura mais olhada</div></div>
{''.join(f'<div class="aviso">{html.escape(a)}</div>' for a in avisos)}
<h2>Para onde foi o olhar</h2><table>{linhas_dest}</table>
<h2>Ranking</h2><div class="duas"><img src="{b64(barras(res['moveis'], 'Por móvel'))}"><img src="{b64(barras(res['faixas'], 'Por altura', 'parcela', 'ic_parcela', '%'))}"></div>
<img src="{b64(barras(res['faces'], 'Por face (lado do móvel)'))}">
<h2>Mapa de calor nas faces mais olhadas</h2>{''.join(blocos)}
<h2>Planta</h2><img src="{b64(planta(cena, res, trajetos))}" style="max-width:640px">
<h2>Todas as faces</h2><table><tr><th>Face</th><th>s</th><th>IC 95%</th><th>parcela</th><th>IC 95%</th><th>pessoas</th><th>s/pessoa</th><th>s/metro</th></tr>{tab}</table>
<h2>Como ler</h2><ul>
<li>Atenção = tempo em que o olhar estimado (Gaze-LLE) caiu na face, pesado pela chance de o alvo estar na imagem e pela parte do mapa do olhar que caiu nela.</li>
<li>O alvo de cada pixel do olhar é o ponto 3D visto ali (profundidade com a escala dos pés das pessoas), levado à face do móvel do modelo do LojaSim. Se a pessoa está do outro lado do móvel, conta a face virada para ela.</li>
<li>Intervalo de 95%: sorteio dos clientes com reposição (2000 vezes). Mede a variação entre clientes, não o erro do modelo de olhar.</li>
<li>Sem gabarito manual: são estimativas de modelos. O vídeo de conferência (conferencia.mp4) mostra cada olhar sobre a imagem.</li></ul>
</html>"""
    saida = args.analise / "relatorio.html"
    saida.write_text(doc)
    print("relatório", saida)


if __name__ == "__main__":
    main()
