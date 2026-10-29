// Grava a experiência (vista Filme) em MP4, quadro a quadro: node gravar.mjs PASTA_EXPERIENCIA [fps] [largura altura]
// Precisa do puppeteer-core (ex.: NODE_PATH=../Modeling3D/work/tools/shots/node_modules) e do ffmpeg.
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import path from "node:path";
const [pasta, fps = "30", W = "1920", H = "1080"] = process.argv.slice(2);
const b = await puppeteer.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: "new",
  args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] });
const p = await b.newPage();
await p.setViewport({ width: +W, height: +H });
p.on("pageerror", (e) => console.log("ERRO:", e.message));
await p.goto("file://" + path.resolve(pasta, "index.html") + "#gravar", { waitUntil: "load" });
await p.waitForFunction("window.pronto === true", { timeout: 120000 });
const dur = await p.evaluate(() => window.DADOS.quadros / window.DADOS.fps);
const saida = path.resolve(pasta, "..", "atencao_3d.mp4");
const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", fps, "-i", "-", "-c:v", "libx264",
  "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", saida], { stdio: ["pipe", "inherit", "inherit"] });
const n = Math.ceil(dur * fps), extra = 3 * fps;  // 3 s parado no resultado final
for (let i = 0; i < n + extra; i++) {
  await p.evaluate((s) => window.irPara(s), Math.min(i, n) / fps);
  ff.stdin.write(await p.screenshot({ type: "jpeg", quality: 92 }));
  if (i % 150 === 0) console.log(`${i}/${n + extra}`);
}
ff.stdin.end();
await new Promise((r) => ff.on("close", r));
await b.close();
console.log("vídeo", saida);
