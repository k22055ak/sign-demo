import { FilesetResolver, HolisticLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { lmToArray, normalizeFrame, rawToFeatures, limitFrames, FEATURE_DIM } from "./features.js";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// 学習データの抽出（Python版Tasks）と同じモデルファイルを、アプリと同じ場所に置いて使う。
// （Googleの「latest」を直接読むと、将来中身が変わって学習時とずれるおそれがあるため）
const HOLISTIC_MODEL = "./holistic_landmarker.task";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
const ONNX_MODEL = "./dgtsta_wlasl100.onnx";
const GLOSSES = "./glosses.json";

const LIVE_WINDOW_MS = 2000;   // リアルタイム判定に使う直近の長さ
const LIVE_INTERVAL_MS = 500;  // 判定の間隔
const LIVE_SMOOTH = 3;         // 直近何回分の確率を平均するか

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const setStatus = (s) => { $("status").textContent = s; };

let vision, holistic, session, glosses, delegateUsed = "";
let lastTs = 0;
let camRunning = false, recording = false, evalRunning = false;
let recFrames = [], recStart = 0;
let fps = 0, lastFrameAt = 0;
let liveBuf = [], liveProbs = [], lastLiveAt = 0, liveBusy = false;

// ---------- 読み込み ----------
async function createHolistic() {
  for (const delegate of ["GPU", "CPU"]) {
    try {
      const h = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HOLISTIC_MODEL, delegate },
        runningMode: "VIDEO",
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minHandLandmarksConfidence: 0.5,
      });
      delegateUsed = delegate;
      return h;
    } catch (e) {
      console.warn(`Holistic (${delegate}) の作成に失敗`, e);
    }
  }
  throw new Error("MediaPipe Holistic を読み込めませんでした");
}

async function init() {
  try {
    setStatus("MediaPipe を読み込み中…");
    vision = await FilesetResolver.forVisionTasks(MP_WASM);
    holistic = await createHolistic();

    setStatus("認識モデルを読み込み中…");
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(ONNX_MODEL, { executionProviders: ["wasm"] });
    glosses = await (await fetch(GLOSSES)).json();
    await runModel(Array.from({ length: 30 }, () => new Float64Array(225)));  // ウォームアップ

    setStatus(`準備完了（MediaPipe: ${delegateUsed} / 単語数: ${glosses.length}）`);
    $("btnCam").disabled = false;
    $("btnEval").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("読み込みに失敗しました: " + e.message);
  }
}

// ---------- 1フレームの処理 ----------
function nextTs(t) {
  lastTs = Math.max(Math.round(t), lastTs + 1);   // VIDEOモードは時刻が単調増加である必要がある
  return lastTs;
}

function hasHand(raw) {
  for (let i = 99; i < 225; i++) if (raw[i] !== 0) return true;
  return false;
}

function processFrame(source) {
  const res = holistic.detectForVideo(source, nextTs(performance.now()));
  const pose = lmToArray(res.poseLandmarks?.[0], 33);
  let lh = lmToArray(res.leftHandLandmarks?.[0], 21);
  let rh = lmToArray(res.rightHandLandmarks?.[0], 21);
  if ($("chkSwap").checked) [lh, rh] = [rh, lh];
  return { raw: normalizeFrame(pose, lh, rh), res };
}

// ---------- 認識モデル ----------
async function runModel(frames) {
  const fr = limitFrames(frames, 300);
  const x = new ort.Tensor("float32", rawToFeatures(fr), [1, fr.length, FEATURE_DIM]);
  const t0 = performance.now();
  const out = await session.run({ skeleton: x });
  return { probs: out.probs.data, ms: performance.now() - t0 };
}

function topK(probs, k) {
  return Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, k);
}

function showResult(probs, head, k = 5) {
  $("resultHead").textContent = head;
  $("result").innerHTML = topK(probs, k).map(([i, p]) =>
    `<li><strong>${glosses[i]}</strong> ${(p * 100).toFixed(1)}%` +
    `<div class="bar"><span style="width:${(p * 100).toFixed(1)}%"></span></div></li>`).join("");
}

// ---------- 描画 ----------
function draw(res) {
  if (overlay.width !== video.videoWidth) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const dots = (lms, color, r) => {
    if (!lms) return;
    ctx.fillStyle = color;
    for (const p of lms) {
      ctx.beginPath();
      ctx.arc(p.x * overlay.width, p.y * overlay.height, r, 0, 2 * Math.PI);
      ctx.fill();
    }
  };
  dots(res.poseLandmarks?.[0]?.slice(0, 25), "rgba(255,255,255,.8)", 3);
  dots(res.leftHandLandmarks?.[0], "#3ddc84", 3);
  dots(res.rightHandLandmarks?.[0], "#ffb020", 3);
}

// ---------- カメラ ----------
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    });
    video.srcObject = stream;
    await video.play();
    camRunning = true;
    $("btnCam").disabled = true;
    $("btnRec").disabled = false;
    const portrait = video.videoHeight > video.videoWidth;
    setStatus(`カメラ ${video.videoWidth}×${video.videoHeight}（MediaPipe: ${delegateUsed}）` +
      (portrait ? " ※縦向きです。横向きにすると学習データに近くなります" : ""));
    scheduleNext();
  } catch (e) {
    setStatus("カメラを開始できませんでした: " + e.message);
  }
}

function stopCamera() {
  camRunning = false;
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  $("btnCam").disabled = false;
  $("btnRec").disabled = true;
}

let lastVideoTime = -1;
function scheduleNext() {
  if (!camRunning) return;
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(() => {
    if (video.currentTime !== lastVideoTime) { lastVideoTime = video.currentTime; onFrame(); }
    else scheduleNext();
  });
}

function onFrame() {
  if (!camRunning) return;
  const now = performance.now();
  const { raw, res } = processFrame(video);
  draw(res);

  if (lastFrameAt) fps = 0.9 * fps + 0.1 * (1000 / (now - lastFrameAt));
  lastFrameAt = now;
  const L = res.leftHandLandmarks?.length ? "○" : "×";
  const R = res.rightHandLandmarks?.length ? "○" : "×";
  $("badge").textContent = `${fps.toFixed(1)} fps ／ 左手${L} 右手${R}`;

  if (recording) {
    recFrames.push(raw);
    $("btnRec").textContent = `停止して判定（${((now - recStart) / 1000).toFixed(1)}秒）`;
  }
  if ($("chkLive").checked) liveStep(raw, now);
  scheduleNext();
}

// ---------- 録画して判定 ----------
async function toggleRecord() {
  if (!recording) {
    recFrames = [];
    recStart = performance.now();
    recording = true;
    $("btnRec").classList.add("rec");
    $("recdot").style.display = "block";
    return;
  }
  recording = false;
  $("btnRec").classList.remove("rec");
  $("btnRec").textContent = "録画して判定";
  $("recdot").style.display = "none";
  const frames = recFrames;
  const sec = (performance.now() - recStart) / 1000;
  if (frames.length < 8) {
    showResult(new Float32Array(glosses.length), "短すぎます。もう少し長く録画してください", 0);
    return;
  }
  const handPct = (100 * frames.filter(hasHand).length) / frames.length;
  const { probs, ms } = await runModel(frames);
  showResult(probs, `${frames.length}フレーム（${sec.toFixed(1)}秒、${(frames.length / sec).toFixed(1)}fps）・` +
    `手の検出 ${handPct.toFixed(0)}%・判定 ${ms.toFixed(1)}ms`);
}

// ---------- リアルタイム判定（試験的） ----------
async function liveStep(raw, now) {
  liveBuf.push({ t: now, raw });
  while (liveBuf.length && now - liveBuf[0].t > LIVE_WINDOW_MS) liveBuf.shift();
  if (liveBusy || now - lastLiveAt < LIVE_INTERVAL_MS || liveBuf.length < 10) return;
  lastLiveAt = now;
  const frames = liveBuf.map((b) => b.raw);
  if (frames.filter(hasHand).length < frames.length * 0.3) {
    liveProbs = [];
    showResult(new Float32Array(glosses.length), "リアルタイム：手が映っていません", 0);
    return;
  }
  liveBusy = true;
  try {
    const { probs, ms } = await runModel(frames);
    liveProbs.push(probs);
    if (liveProbs.length > LIVE_SMOOTH) liveProbs.shift();
    const avg = new Float32Array(probs.length);
    for (const p of liveProbs) for (let i = 0; i < p.length; i++) avg[i] += p[i] / liveProbs.length;
    showResult(avg, `リアルタイム：直近${(LIVE_WINDOW_MS / 1000).toFixed(1)}秒（${frames.length}フレーム）・判定 ${ms.toFixed(1)}ms`, 3);
  } finally {
    liveBusy = false;
  }
}

// ---------- 精度の確認（テスト動画） ----------
function extractFromFile(file, rate) {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    Object.assign(v.style, { position: "fixed", left: "0", top: "0", width: "2px", opacity: "0.01" });
    v.muted = true;
    v.playsInline = true;
    v.src = URL.createObjectURL(file);
    document.body.appendChild(v);
    const frames = [];
    let finished = false, lastProgress = performance.now();
    const cleanup = () => { finished = true; clearInterval(watch); URL.revokeObjectURL(v.src); v.remove(); };
    const fail = (msg) => { if (!finished) { cleanup(); reject(new Error(msg)); } };
    const done = () => { if (!finished) { cleanup(); resolve(frames); } };
    // 10秒間フレームが進まなければ打ち切る（途中で止まる動画への対策）
    const watch = setInterval(() => {
      if (performance.now() - lastProgress > 10000) {
        frames.length ? done() : fail("再生が始まりませんでした");
      }
    }, 1000);
    const cb = () => {
      if (finished) return;
      frames.push(processFrame(v).raw);
      lastProgress = performance.now();
      if (!v.ended) v.requestVideoFrameCallback(cb);
    };
    v.onended = done;
    v.onerror = () => fail("ブラウザが再生できない動画形式です");
    v.onloadedmetadata = () => {
      v.playbackRate = rate;
      v.requestVideoFrameCallback(cb);
      v.play().catch((e) => fail("再生できません: " + e.message));
    };
  });
}

// 動画を1フレームずつ止めて取り出す（取りこぼしがなく、学習時と同じフレーム数になる）
async function extractBySeeking(file, nFrames) {
  const v = document.createElement("video");
  Object.assign(v.style, { position: "fixed", left: "0", top: "0", width: "2px", opacity: "0.01" });
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.src = URL.createObjectURL(file);
  document.body.appendChild(v);
  const cleanup = () => { URL.revokeObjectURL(v.src); v.remove(); };
  try {
    await new Promise((resolve, reject) => {
      v.onloadeddata = resolve;
      v.onerror = () => reject(new Error("ブラウザが再生できない動画形式です"));
      setTimeout(() => reject(new Error("読み込みが終わりませんでした")), 15000);
    });
    const fps = nFrames / v.duration;
    const frames = [];
    for (let i = 0; i < nFrames; i++) {
      const t = Math.min((i + 0.5) / fps, v.duration - 1e-3);
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 2000);
        v.onseeked = () => { clearTimeout(timer); resolve(); };
        v.currentTime = t;
      });
      frames.push(processFrame(v).raw);
    }
    return frames;
  } finally {
    cleanup();
  }
}

let dumped = {};
function saveDump() {
  const blob = new Blob([JSON.stringify(dumped)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "browser_landmarks.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

async function runEval() {
  const log = $("evalLog");
  const lf = $("fileLabels").files[0];
  const files = Array.from($("fileVideos").files);
  if (!lf || !files.length) { log.textContent = "ラベルと動画を選んでください"; return; }
  if (!HTMLVideoElement.prototype.requestVideoFrameCallback) {
    log.textContent = "このブラウザは requestVideoFrameCallback に対応していません（PCのChromeを使ってください）";
    return;
  }
  const labels = JSON.parse(await lf.text());
  const mode = $("selRate").value;
  if (camRunning) stopCamera();
  evalRunning = true;
  dumped = { _swap: $("chkSwap").checked, _mode: mode };
  $("btnEval").disabled = true;
  $("btnDump").disabled = true;

  let n = 0, top1 = 0, top5 = 0, ratioSum = 0, frameSum = 0, timeSum = 0;
  const lines = [], skipped = [];
  const render = (doneAll) => {
    const head = n === 0 ? `${skipped.length}本 飛ばしました` :
      `${n}本  Top-1 ${(top1 / n).toFixed(3)}  Top-5 ${(top5 / n).toFixed(3)}  ` +
      `フレームの取得率 ${(ratioSum / n).toFixed(3)}  処理速度 ${(frameSum / timeSum).toFixed(1)}fps` +
      `  （再生できず飛ばした動画 ${skipped.length}本）`;
    log.textContent = (doneAll ? "【完了】" : "") + head + "\n\n" + lines.slice(-200).join("\n");
  };
  try {
    for (const f of files) {
      const vid = f.name.replace(/\.[^.]+$/, "");
      if (!(vid in labels)) { lines.push(`${vid}: ラベルにないので飛ばしました`); render(false); continue; }
      const [gloss, nExpected] = labels[vid];
      holistic.close();                       // 学習時と同じく、動画ごとに追跡をリセット
      holistic = await createHolistic();
      const t0 = performance.now();
      let frames;
      try {
        frames = mode === "seek"
          ? await extractBySeeking(f, nExpected)
          : await extractFromFile(f, parseFloat(mode));
      } catch (e) {
        skipped.push(vid);
        lines.push(`${vid}  正解:${gloss}  飛ばしました（${e.message}）`);
        render(false);
        continue;
      }
      if (frames.length < 2) {
        skipped.push(vid);
        lines.push(`${vid}  正解:${gloss}  飛ばしました（フレームが取れませんでした）`);
        render(false);
        continue;
      }
      timeSum += (performance.now() - t0) / 1000;
      frameSum += frames.length;
      if (Object.keys(dumped).length < 32) {
        dumped[vid] = frames.map((fr) => Array.from(fr, (x) => Math.round(x * 1e5) / 1e5));
        $("btnDump").disabled = false;
      }
      const { probs } = await runModel(frames);
      const ranks = topK(probs, 5).map(([i]) => glosses[i]);
      n++;
      top1 += ranks[0] === gloss;
      top5 += ranks.includes(gloss);
      ratioSum += frames.length / nExpected;
      lines.push(`${vid}  正解:${gloss}  予測:${ranks[0]}  ${ranks[0] === gloss ? "○" : "×"}  ` +
        `フレーム ${frames.length}/${nExpected}`);
      render(false);
    }
    if (skipped.length) lines.push(`\n再生できず飛ばした動画: ${skipped.join(", ")}`);
    render(true);
  } catch (e) {
    console.error(e);
    log.textContent = "評価中にエラーが起きました: " + e.message + "\n\n" + log.textContent;
  } finally {
    evalRunning = false;
    $("btnEval").disabled = false;
  }
}

// ---------- ボタン ----------
$("btnCam").onclick = startCamera;
$("btnRec").onclick = toggleRecord;
$("btnEval").onclick = runEval;
$("btnDump").onclick = saveDump;
$("chkLive").onchange = () => { liveBuf = []; liveProbs = []; };

init();
