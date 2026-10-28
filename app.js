import { FilesetResolver, HolisticLandmarker, PoseLandmarker, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { lmToArray, normalizeFrame, rawToFeatures, limitFrames, FEATURE_DIM } from "./features.js";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// 学習データの抽出（Python版Tasks）と同じモデルファイルを、アプリと同じ場所に置いて使う。
// （Googleの「latest」を直接読むと、将来中身が変わって学習時とずれるおそれがあるため）
const HOLISTIC_MODEL = "./holistic_landmarker.task";
// 「体＋手のみ」モード（速度の試算用）で使うモデル。判定の精度は保証しない
const POSE_MODELS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
};
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
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
let extractMode = "holistic";            // "holistic" | "lite" | "full"
const poseLms = {}, poseDelegate = {};
let handLm = null, handDelegate = "", switching = false;
let lastTs = 0;
let camRunning = false, recording = false, evalRunning = false;
let recFrames = [], recStart = 0;
let fps = 0, lastFrameAt = 0;
let liveBuf = [], liveProbs = [], lastLiveAt = 0, liveBusy = false;

// ---------- 処理時間の計測 ----------
// 直近500回分を保存し、平均と中央値を出す（単位はミリ秒）
const timing = {
  data: {},
  add(key, ms) {
    const a = (this.data[key] ||= []);
    a.push(ms);
    if (a.length > 500) a.shift();
  },
  reset() { this.data = {}; },
  stat(key) {
    const a = this.data[key];
    if (!a || !a.length) return null;
    const s = [...a].sort((x, y) => x - y);
    return { n: a.length, mean: a.reduce((p, c) => p + c, 0) / a.length, med: s[Math.floor(s.length / 2)] };
  },
};

// ---------- 読み込み ----------
async function createTask(Cls, name, options) {
  for (const delegate of ["GPU", "CPU"]) {
    try {
      const task = await Cls.createFromOptions(vision, {
        ...options, baseOptions: { ...options.baseOptions, delegate },
      });
      return { task, delegate };
    } catch (e) {
      console.warn(`${name} (${delegate}) の作成に失敗`, e);
    }
  }
  throw new Error(`${name} を読み込めませんでした`);
}

async function createHolistic() {
  const { task, delegate } = await createTask(HolisticLandmarker, "MediaPipe Holistic", {
    baseOptions: { modelAssetPath: HOLISTIC_MODEL },
    runningMode: "VIDEO",
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
  });
  delegateUsed = delegate;
  return task;
}

async function ensurePoseHand(kind) {
  if (!poseLms[kind]) {
    const { task, delegate } = await createTask(PoseLandmarker, `Pose(${kind})`, {
      baseOptions: { modelAssetPath: POSE_MODELS[kind] },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    poseLms[kind] = task;
    poseDelegate[kind] = delegate;
  }
  if (!handLm) {
    const { task, delegate } = await createTask(HandLandmarker, "Hand", {
      baseOptions: { modelAssetPath: HAND_MODEL },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    handLm = task;
    handDelegate = delegate;
  }
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
    await runModel(Array.from({ length: 30 }, () => new Float64Array(225)), false);  // ウォームアップ

    setStatus(`準備完了（MediaPipe: ${delegateUsed} / 単語数: ${glosses.length}）`);
    $("btnCam").disabled = false;
    $("btnEval").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("読み込みに失敗しました: " + e.message);
  }
}

function modeLabel(mode = extractMode) {
  if (mode === "holistic") return `Holistic（学習と同じ、${delegateUsed}）`;
  return `体(${mode})＋手のみ（速度の試算用、体:${poseDelegate[mode] || "-"} 手:${handDelegate || "-"}）`;
}

async function changeMode() {
  const mode = $("selExtract").value;
  switching = true;
  $("selExtract").disabled = true;
  try {
    if (mode !== "holistic") {
      setStatus("体と手のモデルを読み込み中…");
      await ensurePoseHand(mode);
    }
    extractMode = mode;
    timing.reset();
    fps = 0; lastFrameAt = 0;
    liveBuf = []; liveProbs = [];
    setStatus(`骨格の抽出: ${modeLabel()}` +
      (mode === "holistic" ? "" : " ※学習時と骨格の取り出し方が違うので、判定結果は参考外です"));
  } catch (e) {
    console.error(e);
    setStatus("切り替えに失敗しました: " + e.message);
    $("selExtract").value = extractMode;
  } finally {
    switching = false;
    $("selExtract").disabled = false;
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

// 「体＋手のみ」モード：検出した手を、体の手首（15=左、16=右）に近いほうへ割り当てる
function assignHands(poseL, hands, handedness) {
  let left = null, right = null;
  if (!hands || !hands.length) return [left, right];
  const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
  if (poseL) {
    const lw = poseL[15], rw = poseL[16];
    if (hands.length === 1) {
      d2(hands[0][0], lw) <= d2(hands[0][0], rw) ? (left = hands[0]) : (right = hands[0]);
    } else {
      const a = d2(hands[0][0], lw) + d2(hands[1][0], rw);
      const b = d2(hands[1][0], lw) + d2(hands[0][0], rw);
      [left, right] = a <= b ? [hands[0], hands[1]] : [hands[1], hands[0]];
    }
  } else {
    // 体が取れないときは利き手の推定を使う（MediaPipeは鏡像を前提にするので、"Left"は本人の右手）
    hands.forEach((h, i) => {
      const lab = handedness?.[i]?.[0]?.categoryName;
      if (lab === "Left") right = right || h; else left = left || h;
    });
  }
  return [left, right];
}

function processFrame(source, mode = extractMode) {
  const ts = nextTs(performance.now());
  const t0 = performance.now();
  let poseL, lhL, rhL;
  if (mode === "holistic") {
    const r = holistic.detectForVideo(source, ts);
    timing.add("mp", performance.now() - t0);
    poseL = r.poseLandmarks?.[0]; lhL = r.leftHandLandmarks?.[0]; rhL = r.rightHandLandmarks?.[0];
  } else {
    const rp = poseLms[mode].detectForVideo(source, ts);
    const t1 = performance.now();
    const rh = handLm.detectForVideo(source, ts);
    const t2 = performance.now();
    timing.add("pose", t1 - t0); timing.add("hand", t2 - t1); timing.add("mp", t2 - t0);
    poseL = rp.landmarks?.[0];
    [lhL, rhL] = assignHands(poseL, rh.landmarks, rh.handedness || rh.handednesses);
  }
  const t3 = performance.now();
  const pose = lmToArray(poseL, 33);
  let lh = lmToArray(lhL, 21);
  let rh = lmToArray(rhL, 21);
  if ($("chkSwap").checked) [lh, rh] = [rh, lh];
  const raw = normalizeFrame(pose, lh, rh);
  timing.add("norm", performance.now() - t3);
  return { raw, res: { pose: poseL, lh: lhL, rh: rhL } };
}

// ---------- 認識モデル ----------
async function runModel(frames, record = false) {
  const fr = limitFrames(frames, 300);
  const t0 = performance.now();
  const feat = rawToFeatures(fr);
  const t1 = performance.now();
  const out = await session.run({ skeleton: new ort.Tensor("float32", feat, [1, fr.length, FEATURE_DIM]) });
  const t2 = performance.now();
  if (record) {
    timing.add("feat", t1 - t0);
    timing.add("onnx", t2 - t1);
    timing.add("recFrames", fr.length);
  }
  return { probs: out.probs.data, ms: t2 - t1 };
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

// ---------- 処理時間の表示 ----------
function timingText() {
  const f = (s, d = 1) => (s ? `平均 ${s.mean.toFixed(d)} ms / 中央値 ${s.med.toFixed(d)} ms（${s.n}回）` : "まだありません");
  const frame = timing.stat("frame"), mp = timing.stat("mp");
  const lines = [];
  lines.push(`端末: ${navigator.userAgent}`);
  lines.push(`骨格の抽出: ${modeLabel()}`);
  if (video.videoWidth) lines.push(`カメラ: ${video.videoWidth}×${video.videoHeight}`);
  lines.push("");
  lines.push("■ 1フレームごとの処理（カメラ動作中）");
  lines.push(`  フレームの間隔: ${f(frame)}` + (frame ? `  → ${(1000 / frame.mean).toFixed(1)} fps` : ""));
  lines.push(`  MediaPipe（骨格の抽出）: ${f(mp)}`);
  if (extractMode !== "holistic") {
    lines.push(`    うち 体: ${f(timing.stat("pose"))}`);
    lines.push(`    うち 手: ${f(timing.stat("hand"))}`);
  }
  lines.push(`  正規化: ${f(timing.stat("norm"), 3)}`);
  lines.push(`  画面への描画: ${f(timing.stat("draw"), 2)}`);
  const rf = timing.stat("recFrames");
  lines.push("");
  lines.push("■ 判定（「録画して判定」1回ごと）" + (rf ? `  平均 ${rf.mean.toFixed(0)}フレーム` : ""));
  lines.push(`  特徴量の計算（393次元）: ${f(timing.stat("feat"), 2)}`);
  lines.push(`  DGT-STA（ONNX Runtime Web）: ${f(timing.stat("onnx"), 2)}`);
  if (frame && mp) {
    lines.push("");
    lines.push(`→ 1フレームの間隔のうち、MediaPipeが ${(100 * mp.mean / frame.mean).toFixed(0)}% を占めています`);
  }
  return lines.join("\n");
}

function copyTiming() {
  const text = timingText();
  const ok = () => { $("btnCopyTiming").textContent = "コピーしました"; setTimeout(() => { $("btnCopyTiming").textContent = "結果をコピー"; }, 1500); };
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).then(ok, () => { $("timingLog").textContent = text; });
  } else {
    $("timingLog").textContent = text;
  }
}

setInterval(() => { if (camRunning || timing.stat("onnx")) $("timingLog").textContent = timingText(); }, 1000);

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
  dots(res.pose?.slice(0, 25), "rgba(255,255,255,.8)", 3);
  dots(res.lh, "#3ddc84", 3);
  dots(res.rh, "#ffb020", 3);
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
    timing.reset();
    fps = 0; lastFrameAt = 0;
    $("btnCam").disabled = true;
    $("btnRec").disabled = false;
    const portrait = video.videoHeight > video.videoWidth;
    setStatus(`カメラ ${video.videoWidth}×${video.videoHeight}（${modeLabel()}）` +
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
  if (switching) { lastFrameAt = 0; scheduleNext(); return; }
  const now = performance.now();
  const { raw, res } = processFrame(video);
  const td = performance.now();
  draw(res);
  timing.add("draw", performance.now() - td);

  if (lastFrameAt) {
    fps = 0.9 * fps + 0.1 * (1000 / (now - lastFrameAt));
    timing.add("frame", now - lastFrameAt);
  }
  lastFrameAt = now;
  const L = res.lh ? "○" : "×";
  const R = res.rh ? "○" : "×";
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
  const { probs, ms } = await runModel(frames, true);
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
// 評価では、選んでいるモードに関係なく、学習と同じ Holistic を使う
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
      frames.push(processFrame(v, "holistic").raw);
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
      frames.push(processFrame(v, "holistic").raw);
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
    log.textContent = (doneAll ? "【完了】" : "") + head + "\n\n" + lines.join("\n");
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
$("selExtract").onchange = changeMode;
$("btnResetTiming").onclick = () => { timing.reset(); $("timingLog").textContent = timingText(); };
$("btnCopyTiming").onclick = copyTiming;

init();
