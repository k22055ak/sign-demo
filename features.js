// 骨格(MediaPipe Holistic) → 393次元の特徴量。
// 学習時のPython処理（extract_landmarks.py の正規化、extract_bone_features.py、
// make_angle_v2.py）と同じ計算をJavaScriptで再現したもの。
//
// 1フレームの並び（393次元）:
//   [0,225)   正規化済み座標  Pose33点 + 左手21点 + 右手21点（各x,y,z）
//   [225,288) 左手のボーン21本  [288,351) 右手のボーン21本  [351,363) 腕のボーン4本
//   [363,378) 左手の指の関節角度15個  [378,393) 右手の指の関節角度15個

export const FEATURE_DIM = 393;
const R_SH = 12, R_EL = 14;

// ブラウザ版MediaPipeでは学習時(Python版)とそろわない列を0にする。
// 体の33点のz・脚の8点(25〜32)のx,y,z・腕のボーン4本のz、計53列。
// 学習時に FEATURE_MASK=posez を指定したモデル（ftnz_*）と組み合わせて使う。
const MASK_COLS = (() => {
  const s = new Set();
  for (let i = 0; i < 33; i++) s.add(3 * i + 2);
  for (let i = 25; i < 33; i++) for (let k = 0; k < 3; k++) s.add(3 * i + k);
  for (const c of [353, 356, 359, 362]) s.add(c);
  return [...s].sort((x, y) => x - y);
})();
export const MASK_COUNT = MASK_COLS.length;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const ARM_CONNECTIONS = [[12, 14], [14, 16], [11, 13], [13, 15]];
const FINGERS = [
  [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
];

/** MediaPipeのランドマーク配列 → Float64Array(n*3)。未検出なら全0 */
export function lmToArray(lms, n) {
  const a = new Float64Array(n * 3);
  if (!lms || lms.length < n) return a;
  for (let i = 0; i < n; i++) {
    a[3 * i] = lms[i].x;
    a[3 * i + 1] = lms[i].y;
    a[3 * i + 2] = lms[i].z;
  }
  return a;
}

function isZero(a, o) {
  return a[o] === 0 && a[o + 1] === 0 && a[o + 2] === 0;
}

/** 右肩を原点、右肩-右肘の距離=1に正規化して225次元にする（未検出の点は0のまま） */
export function normalizeFrame(pose, lh, rh) {
  const ox = pose[3 * R_SH], oy = pose[3 * R_SH + 1];
  const scale = Math.hypot(pose[3 * R_SH] - pose[3 * R_EL], pose[3 * R_SH + 1] - pose[3 * R_EL + 1]);
  const out = new Float64Array(225);
  out.set(pose, 0);
  out.set(lh, 99);
  out.set(rh, 162);
  if (scale < 1e-6) return out;
  for (let o = 0; o < 225; o += 3) {
    if (isZero(out, o)) continue;
    out[o] = (out[o] - ox) / scale;
    out[o + 1] = (out[o + 1] - oy) / scale;
    out[o + 2] = out[o + 2] / scale;
  }
  return out;
}

function writeBones(f, base, conns, dst, d) {
  for (const [s, e] of conns) {
    const os = base + 3 * s, oe = base + 3 * e;
    if (isZero(f, os) || isZero(f, oe)) {
      dst[d] = 0; dst[d + 1] = 0; dst[d + 2] = 0;
    } else {
      dst[d] = f[oe] - f[os];
      dst[d + 1] = f[oe + 1] - f[os + 1];
      dst[d + 2] = f[oe + 2] - f[os + 2];
    }
    d += 3;
  }
  return d;
}

function writeAngles(f, base, dst, d) {
  let detected = false;
  for (let i = 0; i < 63; i++) if (f[base + i] !== 0) { detected = true; break; }
  for (const ch of FINGERS) {
    for (let j = 1; j <= 3; j++) {
      if (!detected) { dst[d++] = 0; continue; }
      const a = base + 3 * ch[j - 1], b = base + 3 * ch[j], c = base + 3 * ch[j + 1];
      const v1x = f[b] - f[a], v1y = f[b + 1] - f[a + 1], v1z = f[b + 2] - f[a + 2];
      const v2x = f[c] - f[b], v2y = f[c + 1] - f[b + 1], v2z = f[c + 2] - f[b + 2];
      const n = Math.hypot(v1x, v1y, v1z) * Math.hypot(v2x, v2y, v2z);
      let cos = n > 1e-8 ? (v1x * v2x + v1y * v2y + v1z * v2z) / n : 1.0;
      cos = Math.min(1, Math.max(-1, cos));
      dst[d++] = Math.acos(cos) / Math.PI;
    }
  }
  return d;
}

/** 正規化済み225次元のフレーム列 → Float32Array(T*393) */
export function rawToFeatures(frames) {
  const T = frames.length;
  const out = new Float32Array(T * FEATURE_DIM);
  const buf = new Float64Array(FEATURE_DIM);
  for (let t = 0; t < T; t++) {
    const f = frames[t];
    buf.set(f, 0);
    let d = 225;
    d = writeBones(f, 99, HAND_CONNECTIONS, buf, d);   // 左手
    d = writeBones(f, 162, HAND_CONNECTIONS, buf, d);  // 右手
    d = writeBones(f, 0, ARM_CONNECTIONS, buf, d);     // 腕
    d = writeAngles(f, 99, buf, d);                    // 左手の角度
    d = writeAngles(f, 162, buf, d);                   // 右手の角度
    for (const c of MASK_COLS) buf[c] = 0;             // ずれる列を0に
    out.set(buf, t * FEATURE_DIM);
  }
  return out;
}

/** フレーム数が上限を超えたら等間隔に間引く（モデルの位置埋め込みは300フレームまで） */
export function limitFrames(frames, maxLen = 300) {
  if (frames.length <= maxLen) return frames;
  const out = [];
  for (let i = 0; i < maxLen; i++) out.push(frames[Math.floor((i * frames.length) / maxLen)]);
  return out;
}
