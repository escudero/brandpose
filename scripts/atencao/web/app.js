// Experiência 3D da atenção: a loja montada pelo LojaSim com o mapa de calor do olhar nascendo ao vivo.
// Dados em window.DADOS e a loja em window.GLB (gerados pelo experiencia.py).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const D = window.DADOS;
const $ = (id) => document.getElementById(id);
const b2t = ([x, y, z]) => new THREE.Vector3(x, z, -y);  // mundo da loja (z para cima) → three (y para cima)
const fmt = (v, c = 1) => v.toLocaleString("pt-BR", { minimumFractionDigits: c, maximumFractionDigits: c });
const DUR = D.quadros / D.fps;

// ---------- cores ----------
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function inferno(t) {
  t = Math.min(1, Math.max(0, t)) * (INFERNO.length - 1);
  const i = Math.min(INFERNO.length - 2, Math.floor(t)), f = t - i, a = INFERNO[i], b = INFERNO[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---------- cena ----------
const canvas = $("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const labels = new CSS2DRenderer({ element: $("labels") });
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
scene.fog = new THREE.FogExp2(0x07090d, 0.028);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;
scene.add(new THREE.HemisphereLight(0x9fb3ff, 0x0b0d12, 0.35));
const luz = new THREE.DirectionalLight(0xffe2c0, 1.1);
luz.castShadow = true;
luz.shadow.mapSize.set(2048, 2048);
scene.add(luz, luz.target);

const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 300);
const ctl = new OrbitControls(camera, canvas);
ctl.enableDamping = true;
ctl.maxPolarAngle = 1.48;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.75, 0.55, 0.62);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- limites da loja ----------
const xs = D.faces.map((f) => f.c[0]).concat(D.pessoas.flatMap((p) => p.xy.filter((_, i) => i % 2 === 0)));
const ys = D.faces.map((f) => f.c[1]).concat(D.pessoas.flatMap((p) => p.xy.filter((_, i) => i % 2 === 1)));
const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(q * (s.length - 1))]; };
const X0 = pct(xs, 0.05) - 1.5, X1 = pct(xs, 0.95) + 1.5, Y0 = pct(ys, 0.05) - 1.5, Y1 = pct(ys, 0.95) + 1.5;
const centro = b2t([(X0 + X1) / 2, (Y0 + Y1) / 2, 0.6]);
const raio = Math.max(X1 - X0, Y1 - Y0);
luz.position.copy(centro).add(new THREE.Vector3(raio * 0.4, 12, raio * 0.3));
luz.target.position.copy(centro);
Object.assign(luz.shadow.camera, { left: -raio, right: raio, top: raio, bottom: -raio, near: 1, far: 50 });

// ---------- loja (glb) ----------
const bin = Uint8Array.from(atob(window.GLB), (c) => c.charCodeAt(0));
const gltf = await new GLTFLoader().parseAsync(bin.buffer, "");
gltf.scene.traverse((o) => {
  if (!o.isMesh) return;
  o.castShadow = o.receiveShadow = true;
  const ms = Array.isArray(o.material) ? o.material : [o.material];
  for (const m of ms) {
    if (m.color) m.color.multiplyScalar(0.62);  // a loja fica sóbria para o calor brilhar
    if (m.emissive) m.emissiveIntensity = Math.min(m.emissiveIntensity ?? 1, 0.25);
  }
});
scene.add(gltf.scene);

// ---------- calor nas faces ----------
const faces = D.faces.map((f, i) => {
  const cols = Math.max(1, Math.ceil(f.w1 / D.cel)), rows = Math.max(1, Math.ceil(f.w2 / D.cel));
  const cv = document.createElement("canvas");
  cv.width = cols; cv.height = rows;
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -4, toneMapped: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(f.w1, f.w2), mat);
  const a1 = b2t(f.a1), a2 = b2t(f.a2), n = b2t(f.n);
  mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(a1, a2, n));
  mesh.position.copy(b2t(f.c)).addScaledVector(n, 0.02);
  mesh.renderOrder = 2;
  mesh.userData.face = i;
  scene.add(mesh);
  return { ...f, cols, rows, grade: new Float32Array(cols * rows), cv, ctx: cv.getContext("2d"), tex, mesh, vivo: 0, sujo: true };
});

// ---------- calor no piso: onde os clientes ficam ----------
const PC = 0.1, PW = Math.ceil((X1 - X0) / PC), PH = Math.ceil((Y1 - Y0) / PC);
const piso = { grade: new Float32Array(PW * PH), cv: Object.assign(document.createElement("canvas"), { width: PW, height: PH }) };
piso.ctx = piso.cv.getContext("2d");
piso.tex = new THREE.CanvasTexture(piso.cv);
piso.tex.colorSpace = THREE.SRGBColorSpace;
{
  const m = new THREE.Mesh(new THREE.PlaneGeometry(X1 - X0, Y1 - Y0),
    new THREE.MeshBasicMaterial({ map: piso.tex, transparent: true, depthWrite: false, toneMapped: false }));
  m.rotation.x = -Math.PI / 2;  // plano XY → piso; +Y da textura = −z do three = +y da loja
  m.position.copy(b2t([(X0 + X1) / 2, (Y0 + Y1) / 2, 0.012]));
  m.renderOrder = 1;
  scene.add(m);
}

function splat(g, cols, rows, c, r, w, sig) {
  const R = Math.ceil(sig * 2.5);
  for (let y = Math.max(0, r - R); y <= Math.min(rows - 1, r + R); y++)
    for (let x = Math.max(0, c - R); x <= Math.min(cols - 1, c + R); x++)
      g[y * cols + x] += w * Math.exp(-((x - c) ** 2 + (y - r) ** 2) / (2 * sig * sig));
}

// normalização: o máximo do resultado final (calculado uma vez), para o calor crescer até ele
let VMAX = 1, PMAX = 1;
function aplicarAmostra(a) {
  for (const [fi, s1, s2, w] of a.a) {
    const f = faces[fi];
    splat(f.grade, f.cols, f.rows, Math.floor(s1 / D.cel), Math.floor(s2 / D.cel), a.k * w, 1.6);
    f.vivo += a.k * w; f.sujo = true;
  }
}
function pintarFace(f) {
  const img = f.ctx.createImageData(f.cols, f.rows);
  for (let r = 0; r < f.rows; r++) for (let c = 0; c < f.cols; c++) {
    const v = Math.min(1, f.grade[r * f.cols + c] / VMAX), o = ((f.rows - 1 - r) * f.cols + c) * 4;
    const [R, G, B] = inferno(0.12 + 0.88 * Math.sqrt(v));
    img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = v < 0.01 ? 0 : 60 + 195 * Math.min(1, Math.sqrt(v) * 1.3);
  }
  f.ctx.putImageData(img, 0, 0); f.tex.needsUpdate = true; f.sujo = false;
}
function pintarPiso() {
  const img = piso.ctx.createImageData(PW, PH);
  for (let r = 0; r < PH; r++) for (let c = 0; c < PW; c++) {
    const v = Math.min(1, piso.grade[r * PW + c] / PMAX), o = ((PH - 1 - r) * PW + c) * 4;
    img.data[o] = 30 + 20 * v; img.data[o + 1] = 120 + 110 * v; img.data[o + 2] = 255; img.data[o + 3] = v < 0.02 ? 0 : 150 * Math.sqrt(v);
  }
  piso.ctx.putImageData(img, 0, 0); piso.tex.needsUpdate = true;
}
function pisoAte(q0, q1) {
  for (const p of D.pessoas) {
    const n = p.xy.length / 2;
    for (let k = 0; k < n; k++) {
      const q = p.q0 + k * D.passo;
      if (q < q0 || q >= q1) continue;
      splat(piso.grade, PW, PH, Math.floor((p.xy[2 * k] - X0) / PC), Math.floor((p.xy[2 * k + 1] - Y0) / PC), D.passo / D.fps, 2.2);
    }
  }
}
// passada única para achar os máximos finais
for (const a of D.amostras) aplicarAmostra(a);
VMAX = Math.max(...faces.flatMap((f) => Array.from(f.grade))) * 0.55 || 1;
pisoAte(0, 1e9);
PMAX = Math.max(...piso.grade) * 0.5 || 1;
function zerar() {
  for (const f of faces) { f.grade.fill(0); f.vivo = 0; f.sujo = true; }
  piso.grade.fill(0);
}
zerar();

// ---------- colunas de luz sobre os móveis (sobem com a atenção) ----------
const porMovel = {};
faces.forEach((f, i) => (porMovel[f.item] ??= []).push(i));
const colunas = D.moveis.filter((m) => porMovel[m.id]).map((m) => {
  const geo = new THREE.CylinderGeometry(0.09, 0.09, 1, 20, 1, true);
  geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffb547, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending,
    depthWrite: false, toneMapped: false });
  const c = new THREE.Mesh(geo, mat);
  c.position.copy(b2t([m.x, m.y, m.H + 0.05]));
  c.scale.y = 0.001;
  const div = document.createElement("div");
  div.className = "rot";
  const rot = new CSS2DObject(div);
  scene.add(c, rot);
  return { m, c, div, rot, seg: 0 };
});

// ---------- pessoas ----------
const pessoas = D.pessoas.map((p) => {
  const cor = new THREE.Color(p.cor);
  const g = new THREE.Group();
  const corpo = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, p.z - 0.55, 6, 14),
    new THREE.MeshStandardMaterial({ color: cor, emissive: cor, emissiveIntensity: 0.55, roughness: 0.4, transparent: true, opacity: 0.92 }));
  corpo.position.y = (p.z - 0.55) / 2 + 0.17;
  const cab = new THREE.Mesh(new THREE.SphereGeometry(0.12, 20, 14),
    new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: cor, emissiveIntensity: 1.4 }));
  cab.position.y = p.z;
  const anel = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.36, 40),
    new THREE.MeshBasicMaterial({ color: cor, transparent: true, opacity: 0.8, side: THREE.DoubleSide, toneMapped: false }));
  anel.rotation.x = -Math.PI / 2; anel.position.y = 0.02;
  corpo.castShadow = true;
  g.add(corpo, cab, anel);
  g.visible = false;
  scene.add(g);
  // feixe do olhar: cone aditivo dos olhos ao alvo
  const fg = new THREE.ConeGeometry(0.16, 1, 24, 1, true);
  fg.translate(0, -0.5, 0); fg.rotateX(-Math.PI / 2);  // ponta na origem, abre para −z... depois lookAt
  const feixe = new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ color: cor, transparent: true, opacity: 0.35,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }));
  feixe.visible = false;
  scene.add(feixe);
  const ponto = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }));
  ponto.visible = false;
  scene.add(ponto);
  const rastroGeo = new THREE.BufferGeometry();
  rastroGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
  const rastro = new THREE.Line(rastroGeo, new THREE.LineBasicMaterial({ color: cor, transparent: true, opacity: 0.6 }));
  scene.add(rastro);
  return { ...p, cor, g, feixe, ponto, rastro, n: p.xy.length / 2 };
});
const amostrasPorPessoa = {};
D.amostras.forEach((a) => (amostrasPorPessoa[a.p] ??= []).push(a));

function posicao(p, q) {
  const k = (q - p.q0) / D.passo;
  if (k < 0 || k > p.n - 1) return null;
  const i = Math.floor(k), f = k - i, j = Math.min(p.n - 1, i + 1);
  return [p.xy[2 * i] + (p.xy[2 * j] - p.xy[2 * i]) * f, p.xy[2 * i + 1] + (p.xy[2 * j + 1] - p.xy[2 * i + 1]) * f];
}

// ---------- painéis ----------
$("sub").textContent = `${D.resumo.clientes} clientes · ${fmt(DUR, 0)} s de vídeo · ${D.faces.length} faces de exposição · olhar por Gaze-LLE, loja em 3D pelo LojaSim`;
const linhas = colunas.map((c) => {
  const el = document.createElement("div");
  el.className = "lin";
  el.innerHTML = `<div class="bar"></div><span><b></b><em></em></span>`;
  $("linhas").appendChild(el);
  return { c, el, bar: el.querySelector(".bar"), nome: el.querySelector("b"), val: el.querySelector("em") };
});
const FAIXAS = [["Alto (160 cm+)", 1.6, 9], ["Olhos (110–160)", 1.1, 1.6], ["Mãos (50–110)", 0.5, 1.1], ["Chão (0–50)", 0, 0.5]];
const faixaEls = FAIXAS.map(([nome]) => {
  const el = document.createElement("div");
  el.className = "faixa";
  el.innerHTML = `<span>${nome}</span><div><i style="width:0"></i></div><b>0%</b>`;
  $("faixas").appendChild(el);
  return { i: el.querySelector("i"), b: el.querySelector("b") };
});
let faixaSeg = [0, 0, 0, 0];

function atualizarPaineis(nAgora, nOlh) {
  let total = 0;
  for (const c of colunas) { c.seg = porMovel[c.m.id].reduce((s, i) => s + faces[i].vivo, 0); total += c.seg; }
  const max = Math.max(1e-6, ...colunas.map((c) => c.seg));
  const finalMax = Math.max(...colunas.map((c) => porMovel[c.m.id].reduce((s, i) => s + D.faces[i].seg, 0)));
  for (const c of colunas) {
    const h = 0.02 + 2.8 * (c.seg / Math.max(finalMax, 1e-6));
    c.c.scale.y = h;
    c.c.material.opacity = c.seg > 0.05 ? 0.25 + 0.5 * (c.seg / max) : 0;
    c.rot.position.copy(c.c.position).add(new THREE.Vector3(0, h + 0.18, 0));
    c.rot.visible = c.seg > 0.15;
    c.div.innerHTML = `${D.moveis.find((m) => m.id === c.m.id) ? nomeCurto(c.m.nome) : ""} · <b>${fmt(c.seg)} s</b>`;
  }
  const ord = [...linhas].sort((a, b) => b.c.seg - a.c.seg);
  ord.forEach((l, i) => {
    l.el.style.display = i < 7 ? "" : "none";
    l.el.style.order = i;
    l.bar.style.width = `${(100 * l.c.seg) / max}%`;
    l.nome.textContent = `${i + 1}. ${l.c.m.nome}`;
    l.val.textContent = `${fmt(l.c.seg)} s · ${total ? Math.round((100 * l.c.seg) / total) : 0}%`;
  });
  $("linhas").style.display = "flex"; $("linhas").style.flexDirection = "column";
  const ft = faixaSeg.reduce((a, b) => a + b, 0) || 1, fm = Math.max(...faixaSeg) || 1;
  faixaEls.forEach((e, j) => { e.i.style.width = `${(100 * faixaSeg[j]) / fm}%`; e.b.textContent = `${Math.round((100 * faixaSeg[j]) / ft)}%`; });
  $("k-seg").textContent = `${fmt(total)} s`;
  $("k-agora").textContent = nAgora;
  $("k-olh").textContent = nOlh.toLocaleString("pt-BR");
}
const nomeCurto = (n) => n.replace("Geladeira de porta de vidro", "Geladeira").replace("Display de papelão", "Display");

// ---------- tempo ----------
const video = $("video");
let t = 0, tocando = false, vel = 1, qAplicado = 0, iAmostra = 0, nOlh = 0;
const ordem = [...D.amostras].sort((a, b) => a.q - b.q);
function avancarAte(q) {
  if (q < qAplicado) { zerar(); faixaSeg = [0, 0, 0, 0]; qAplicado = 0; iAmostra = 0; nOlh = 0; }
  while (iAmostra < ordem.length && ordem[iAmostra].q <= q) {
    const a = ordem[iAmostra++];
    aplicarAmostra(a); nOlh++;
    for (const [fi, , s2, w] of a.a) {
      if (D.faces[fi].lado === "topo") continue;
      faixaSeg[FAIXAS.findIndex(([, lo, hi]) => s2 >= lo && s2 < hi)] += a.k * w;
    }
  }
  pisoAte(qAplicado, q);
  qAplicado = q;
}

function quadro(q) {
  avancarAte(q);
  let agora = 0;
  for (const p of pessoas) {
    const xy = posicao(p, q);
    p.g.visible = !!xy;
    const pos = p.rastro.geometry.attributes.position;
    let m = 0;
    for (let k = Math.max(0, Math.floor((q - p.q0) / D.passo) - 60); k <= Math.min(p.n - 1, (q - p.q0) / D.passo) && m < 64; k++, m++)
      pos.setXYZ(m, p.xy[2 * k], 0.03, -p.xy[2 * k + 1]);
    pos.needsUpdate = true;
    p.rastro.geometry.setDrawRange(0, m);
    p.rastro.visible = !!xy;
    if (!xy) { p.feixe.visible = p.ponto.visible = false; continue; }
    agora++;
    p.g.position.copy(b2t([xy[0], xy[1], 0]));
    // olhar: a amostra mais recente (até 0,3 s atrás)
    const as = amostrasPorPessoa[p.id] || [];
    let a = null;
    for (let i = as.length - 1; i >= 0; i--) if (as[i].q <= q) { a = as[i]; break; }
    if (a && q - a.q <= 9 && a.a.length) {
      let W = 0; const P = new THREE.Vector3();
      for (const t of a.a) { P.addScaledVector(b2t([t[4], t[5], t[6]]), t[3]); W += t[3]; }
      P.divideScalar(W);
      const O = b2t([xy[0], xy[1], p.z]);
      const d = O.distanceTo(P);
      p.feixe.position.copy(O);
      p.feixe.lookAt(P);
      p.feixe.scale.set(Math.max(0.6, d * 0.5), Math.max(0.6, d * 0.5), d);
      p.feixe.material.opacity = 0.18 + 0.4 * Math.min(1, W * 3);
      p.feixe.visible = true;
      p.ponto.position.copy(P);
      p.ponto.scale.setScalar(0.8 + 0.4 * Math.sin(performance.now() / 120));
      p.ponto.visible = true;
      p.g.children[0].rotation.y = 0;
      p.g.rotation.y = Math.atan2(-(P.z - O.z), P.x - O.x) - Math.PI / 2;
    } else { p.feixe.visible = p.ponto.visible = false; }
  }
  for (const f of faces) if (f.sujo) pintarFace(f);
  pintarPiso();
  atualizarPaineis(agora, nOlh);
}

// ---------- câmeras ----------
// ---------- filme: a câmera anda do começo ao fim do vídeo ----------
// Shift+1 / Shift+2 / Shift+3 gravam começo / meio / fim.
// pose = giro em volta do alvo (graus), elevação (graus), distância (× raio da loja) e alvo (fração do caminho
// entre o lugar das primeiras pessoas e o centro). Shift+1 / Shift+2 gravam a vista atual como começo / fim.
const inicioPessoas = (() => {
  const ps = D.pessoas.filter((p) => p.q0 < 3 * D.fps);
  const v = new THREE.Vector3();
  ps.forEach((p) => v.add(b2t([p.xy[0], p.xy[1], 0.6])));
  return ps.length ? v.divideScalar(ps.length) : centro.clone();
})();
const centroAtencao = (() => {  // média das faces pesada pela atenção final
  const v = new THREE.Vector3(); let w = 0;
  D.faces.forEach((f) => { v.addScaledVector(b2t(f.c), f.seg); w += f.seg; });
  return w ? v.divideScalar(w).setY(0.6) : centro.clone();
})();
const POSES = JSON.parse(localStorage.getItem("brandpose-filme-v2") || "null") || {
  ini: { az: 115, el: 45, dist: 0.8, alvo: [centroAtencao.x, centroAtencao.y, centroAtencao.z] },
  meio: { az: 75, el: 33, dist: 0.5, alvo: [centroAtencao.x, centroAtencao.y, centroAtencao.z] },
  fim: { az: 105, el: 42, dist: 0.55, alvo: [centroAtencao.x, centroAtencao.y, centroAtencao.z] },
};
window.POSES = POSES;  // para ajustar pelo console
const suave = (u) => u * u * (3 - 2 * u);
// curva suave (Catmull-Rom) passando por começo, meio e fim, sem parar no meio
function spline(a, b, c, e) {
  const [p0, p1, s] = e < 0.5 ? [a, b, e * 2] : [b, c, e * 2 - 1];
  const m0 = e < 0.5 ? (b - a) * 0.5 : (c - a) * 0.25, m1 = e < 0.5 ? (c - a) * 0.25 : (c - b) * 0.5;
  const h = (t) => [2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2];
  const [h0, h1, h2, h3] = h(s);
  return h0 * p0 + h1 * m0 + h2 * p1 + h3 * m1;
}
function poseFilme(u) {
  const { ini: a, meio: m, fim: b } = POSES, e = suave(Math.min(1, Math.max(0, u)));
  const az = THREE.MathUtils.degToRad(spline(a.az, m.az, b.az, e)), el = THREE.MathUtils.degToRad(spline(a.el, m.el, b.el, e));
  const dist = raio * spline(a.dist, m.dist, b.dist, e);
  const alvo = new THREE.Vector3(...[0, 1, 2].map((i) => spline(a.alvo[i], m.alvo[i], b.alvo[i], e)));
  const pos = alvo.clone().add(new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)).multiplyScalar(dist));
  return { pos, alvo };
}
function poseAtual() {
  const d = camera.position.clone().sub(ctl.target);
  const r = d.length();
  return { az: THREE.MathUtils.radToDeg(Math.atan2(d.x, d.z)), el: THREE.MathUtils.radToDeg(Math.asin(d.y / r)), dist: r / raio,
    alvo: ctl.target.toArray() };
}
addEventListener("keydown", (e) => {
  const k = { Digit1: "ini", Digit2: "meio", Digit3: "fim" }[e.code];
  if (!e.shiftKey || !k) return;
  POSES[k] = poseAtual();
  localStorage.setItem("brandpose-filme-v2", JSON.stringify(POSES));
  console.log("pose do filme", JSON.stringify(POSES));
});

const vistas = {
  filme: () => { camera.fov = 42; ctl.enabled = false; ctl.autoRotate = false; },
  voo: () => { camera.fov = 42; ctl.enabled = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.5; },
  cctv: () => {
    const C = D.camera;
    camera.fov = (2 * Math.atan(C.H / 2 / C.f) * 180) / Math.PI;
    camera.position.copy(b2t([0, 0, C.h]));
    camera.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(b2t(C.direita), b2t(C.cima), b2t(C.tras)));
    ctl.enabled = false; ctl.autoRotate = false;
  },
  cima: () => {
    camera.fov = 38; ctl.enabled = true; ctl.autoRotate = false;
    alvoCam = { pos: centro.clone().add(new THREE.Vector3(0.01, raio * 1.25, 0.01)), tgt: centro.clone() };
  },
};
let vista = "filme", alvoCam = null;
function mudarVista(v) {
  vista = v;
  for (const k of ["filme", "voo", "cctv", "cima"]) $(`v-${k}`).classList.toggle("on", k === v);
  if (v === "voo" && !ctlLivre) alvoCam = { pos: centro.clone().add(new THREE.Vector3(raio * 0.42, raio * 0.36, raio * 0.42)), tgt: centro.clone() };
  vistas[v]();
  camera.updateProjectionMatrix();
}
let ctlLivre = false;
for (const k of ["filme", "voo", "cctv", "cima"]) $(`v-${k}`).onclick = () => { ctlLivre = false; mudarVista(k); };
camera.position.copy(centro).add(new THREE.Vector3(raio * 1.4, raio * 1.1, raio * 1.4));
ctl.target.copy(centro);
mudarVista("filme");

// ---------- controles ----------
function tocar(sim) {
  tocando = sim;
  $("play").textContent = sim ? "❚❚" : "▶";
  if (sim) { if (t >= DUR - 0.05) t = 0; video.currentTime = t; video.playbackRate = vel; video.play().catch(() => {}); }
  else video.pause();
}
$("play").onclick = () => tocar(!tocando);
$("tempo").oninput = (e) => { t = (e.target.value / 1000) * DUR; video.currentTime = t; };
$("vel").onclick = () => { vel = vel === 1 ? 4 : vel === 4 ? 8 : 1; $("vel").textContent = `${vel}×`; video.playbackRate = vel; };
$("final").onclick = () => { tocar(false); t = DUR; mudarVista("filme"); };
$("b-pip").onclick = () => { const s = $("pip").style; s.display = s.display === "none" ? "" : "none"; };
$("comecar").onclick = () => {
  $("intro").style.opacity = 0;
  setTimeout(() => ($("intro").style.display = "none"), 1200);
  t = 0; tocar(true);
};
addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); tocar(!tocando); } });

// ---------- dica ao passar o mouse ----------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
canvas.addEventListener("pointermove", (e) => {
  mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(faces.map((f) => f.mesh))[0];
  const d = $("dica");
  if (!hit) { d.style.display = "none"; return; }
  const f = faces[hit.object.userData.face];
  d.innerHTML = `<b>${f.nome}</b><div class="n">${fmt(f.vivo)} s</div>
    <small>no vídeo inteiro: ${fmt(f.seg)} s (95%: ${fmt(f.ic[0])}–${fmt(f.ic[1])}) · ${Math.round(f.parcela * 100)}% da atenção ·
    ${f.pessoas} cliente(s) · ${fmt(f.w1, 2)} × ${fmt(f.w2, 2)} m</small>`;
  d.style.display = "block";
  d.style.left = `${Math.min(e.clientX + 16, innerWidth - 300)}px`;
  d.style.top = `${e.clientY + 16}px`;
});
canvas.addEventListener("pointerdown", () => {
  if (vista === "filme") { ctl.target.copy(poseFilme(t / DUR).alvo); ctlLivre = true; mudarVista("voo"); }  // mexeu: vira voo livre
  ctl.autoRotate = false; alvoCam = null;
});

// ---------- laço ----------
function redimensionar() {
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  labels.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", redimensionar);
redimensionar();
let antes = performance.now();
renderer.setAnimationLoop(() => {
  const agora = performance.now(), dt = Math.min(0.1, (agora - antes) / 1000);
  antes = agora;
  if (tocando) {
    t += dt * vel;
    if (Math.abs(video.currentTime - t) > 0.4 && !video.seeking) video.currentTime = t;
    if (t >= DUR) { t = DUR; tocar(false); }
  }
  $("tempo").value = Math.round((1000 * t) / DUR);
  const mm = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  $("relogio").textContent = `${mm(t)} / ${mm(DUR)}`;
  quadro(Math.floor(t * D.fps));
  if (vista === "filme") {
    const p = poseFilme(t / DUR);
    camera.position.lerp(p.pos, 0.12);
    ctl.target.lerp(p.alvo, 0.12);
    camera.lookAt(ctl.target);
  }
  if (alvoCam && vista !== "cctv" && vista !== "filme") {
    camera.position.lerp(alvoCam.pos, 0.04);
    ctl.target.lerp(alvoCam.tgt, 0.06);
    if (camera.position.distanceTo(alvoCam.pos) < 0.05) alvoCam = null;
  }
  if (vista === "voo" || vista === "cima") ctl.update();
  composer.render();
  labels.render(scene, camera);
});

// ---------- gravação (index.html#gravar): o gravar.mjs chama irPara(t) quadro a quadro ----------
if (location.hash === "#gravar") {
  renderer.setAnimationLoop(null);
  $("intro").remove();
  $("play").textContent = "❚❚";
  window.irPara = async (s) => {
    t = Math.min(DUR, s);
    $("tempo").value = Math.round((1000 * t) / DUR);
    const mm = (x) => `${Math.floor(x / 60)}:${String(Math.floor(x % 60)).padStart(2, "0")}`;
    $("relogio").textContent = `${mm(t)} / ${mm(DUR)}`;
    quadro(Math.floor(t * D.fps));
    const p = poseFilme(t / DUR);
    camera.position.copy(p.pos); ctl.target.copy(p.alvo); camera.lookAt(p.alvo);
    await new Promise((r) => { video.onseeked = r; video.currentTime = Math.min(t, video.duration - 0.05); setTimeout(r, 500); });
    composer.render();
    labels.render(scene, camera);
  };
  window.pronto = true;
}
