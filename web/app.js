/* 在线视频生成字幕 — 前端（无构建，原生 JS）。详细设计 §7。
 *
 * 页面状态机（单一 state 变量驱动渲染）：
 *   idle → extracting → uploading → queued → processing → done | error
 *
 * 核心：ffmpeg.wasm 单线程 core + WORKERFS 挂载，浏览器内无损解封装音轨，
 * 视频不出本机；仅音频经 XHR 上传，2s 轮询状态，?task= 参数恢复。
 */
"use strict";

const { FFmpeg } = FFmpegWASM;

// 纯音频扩展名：跳过提取直接上传（后端内容嗅探兜底，判断错误无害）。
const AUDIO_EXT = new Set(["wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus", "mka", "wma", "amr"]);

const el = (id) => document.getElementById(id);
const S = {
  idle: el("s-idle"), progress: el("s-progress"),
  done: el("s-done"), error: el("s-error"),
};

let cfg = { rtf: 0.65, max_size_bytes: 209715200, max_duration_sec: 10800 };
let ffmpeg = null;             // 懒加载的 FFmpeg 实例
let audioBlob = null;          // 提取/选择得到的音频，供重试复用
let audioName = "audio";       // 上传字段 filename（原视频名）
let curTask = null;            // 当前 task_id
let pollTimer = null;
let lastLogDuration = null;    // 从 ffmpeg log 解析的时长（秒）

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------
(async function init() {
  try {
    const r = await fetch("/api/config");
    if (r.ok) cfg = await r.json();
  } catch (_) {}
  el("limit-hint").textContent =
    `大小上限 ${(cfg.max_size_bytes / (1 << 20)) | 0}MB · 时长上限 ${(cfg.max_duration_sec / 3600).toFixed(0)}h · 支持任意可解码音视频`;

  wireFileInput();
  wireButtons();

  // ?task= 恢复：直接进入轮询
  const t = new URLSearchParams(location.search).get("task");
  if (t) { openTask(t); }
  else { loadHistory(); }
})();

// ---------------------------------------------------------------------------
// 文件选择与拖放
// ---------------------------------------------------------------------------
function wireFileInput() {
  const input = el("file"), drop = el("drop");
  input.addEventListener("change", () => input.files[0] && handleFile(input.files[0]));
  ["dragenter", "dragover"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("hover"); }));
  ["dragleave", "drop"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("hover"); }));
  drop.addEventListener("drop", (ev) => {
    const f = ev.dataTransfer.files[0];
    if (f) handleFile(f);
  });
}

function wireButtons() {
  el("btn-cancel").onclick = cancelUpload;
  el("btn-retry-up").onclick = () => uploadAudio(audioBlob, audioName);
  el("btn-restart").onclick = resetToIdle;
  el("btn-new").onclick = resetToIdle;
  el("btn-delete").onclick = deleteResult;
  el("btn-refresh-hist").onclick = loadHistory;
}

async function handleFile(file) {
  audioName = file.name;
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  el("p-name").textContent = file.name;
  el("btn-retry-up").classList.add("hidden");

  if (AUDIO_EXT.has(ext)) {
    // 纯音频：跳过提取
    if (!checkSize(file.size)) return;
    audioBlob = file;
    show("progress");
    setState("准备上传…", 0);
    uploadAudio(file, file.name);
    return;
  }

  // 视频：ffmpeg.wasm 无损提取音轨
  show("progress");
  setState("正在浏览器内提取音轨…", 0);
  el("p-taskid-row").classList.add("hidden");
  try {
    const blob = await extractAudio(file);
    el("p-taskid-row").classList.remove("hidden");
    if (!checkSize(blob.size)) return;
    audioBlob = blob;
    uploadAudio(blob, file.name);
  } catch (err) {
    showError(
      "音轨提取失败：" + (err && err.message ? err.message : err),
      "可在本机执行：\n  ffmpeg -i 视频文件 -vn -c:a copy 音频.mka\n然后直接上传生成的音频文件。"
    );
  }
}

function checkSize(bytes) {
  if (bytes > cfg.max_size_bytes) {
    showError(
      `音频体积 ${(bytes / (1 << 20)).toFixed(1)}MB 超过上限 ${(cfg.max_size_bytes / (1 << 20)) | 0}MB。`,
      "若音轨为无损 PCM/FLAC，可先转码：\n  ffmpeg -i 视频 -vn -c:a aac -b:a 128k 音频.m4a"
    );
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// 音轨提取（ffmpeg.wasm 单线程 + WORKERFS）
// ---------------------------------------------------------------------------
async function loadFFmpeg() {
  if (ffmpeg) return ffmpeg;
  ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ message }) => {
    const m = /Duration:\s*(\d+):(\d+):(\d+\.\d+)/.exec(message);
    if (m) lastLogDuration = (+m[1]) * 3600 + (+m[2]) * 60 + parseFloat(m[3]);
    const p = /time=\s*(\d+):(\d+):(\d+\.\d+)/.exec(message);
    if (p && lastLogDuration) {
      const t = (+p[1]) * 3600 + (+p[2]) * 60 + parseFloat(p[3]);
      setBar(Math.min(99, (t / lastLogDuration) * 100));
    }
  });
  // 必须传“绝对 http URL”：UMD 内部用 new URL(s, <硬编码 file:// 基址>) 解析，
  // 传根相对路径会被解析成 file:///vendor/... 从而被浏览器拦截。
  const abs = (p) => new URL(p, location.href).href;
  // 自托管说明：ffmpeg.js UMD 的 classWorker 已打补丁为「经典 worker」（见
  // vendor 说明），其内部用 importScripts() 加载 core，因此 core 必须是 UMD 构建。
  // classWorkerURL/coreURL/wasmURL 均须传绝对 http URL（UMD 内部按硬编码 file://
  // 基址解析，传根相对路径会变成 file:// 被浏览器拦截）。
  await ffmpeg.load({
    classWorkerURL: abs("/vendor/ffmpeg/814.ffmpeg.js"),
    coreURL: abs("/vendor/ffmpeg/ffmpeg-core.js"),
    wasmURL: abs("/vendor/ffmpeg/ffmpeg-core.wasm"),
  });
  return ffmpeg;
}

async function extractAudio(file) {
  const ff = await loadFFmpeg();
  lastLogDuration = null;
  await ff.createDir("/in");
  await ff.mount("WORKERFS", { files: [file] }, "/in");  // 不整体复制进 wasm 内存
  try {
    const rc = await ff.exec([
      "-i", `/in/${file.name}`,
      "-map", "0:a:0",        // 第一条音轨
      "-vn", "-c:a", "copy",  // 只解封装，比特级无损
      "-f", "matroska", "/out.mka",
    ]);
    if (rc !== 0) throw new Error("ffmpeg 退出码 " + rc);
    const data = await ff.readFile("/out.mka");
    return new Blob([data.buffer], { type: "video/x-matroska" });
  } finally {
    try { await ff.unmount("/in"); } catch (_) {}
    try { await ff.deleteFile("/out.mka"); } catch (_) {}
  }
}

// ---------------------------------------------------------------------------
// 上传（XHR：有上传进度；失败自动重试 1 次）
// ---------------------------------------------------------------------------
let curXhr = null;
function uploadAudio(blob, filename, isRetry) {
  setState("上传音频…", 0);
  el("btn-cancel").classList.remove("hidden");
  el("btn-retry-up").classList.add("hidden");

  const fd = new FormData();
  const uploadName = /\.[a-z0-9]+$/i.test(filename) ? filename : filename + ".mka";
  fd.append("file", blob, uploadName);
  fd.append("filename", filename);

  const xhr = new XMLHttpRequest();
  curXhr = xhr;
  xhr.open("POST", "/api/tasks");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) setBar((e.loaded / e.total) * 100); };
  xhr.onload = () => {
    curXhr = null;
    el("btn-cancel").classList.add("hidden");
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status === 201) {
      curTask = body.task_id;
      history.replaceState(null, "", "?task=" + encodeURIComponent(curTask));
      el("p-taskid-row").classList.remove("hidden");
      el("p-taskid").textContent = curTask;
      startPolling();
    } else {
      const msg = (body.error && body.error.message) || `上传失败（HTTP ${xhr.status}）`;
      // 队列满/大小/时长等属确定性拒绝，不重试
      showError(msg);
    }
  };
  xhr.onerror = () => {
    curXhr = null;
    if (!isRetry) { setState("上传失败，正在重试…", 0); uploadAudio(blob, filename, true); }
    else {
      el("btn-cancel").classList.add("hidden");
      el("btn-retry-up").classList.remove("hidden");
      setState("上传失败，可点“重试上传”（音频已在内存，无需重新提取）", 0);
    }
  };
  xhr.send(fd);
}

function cancelUpload() {
  if (curXhr) { curXhr.abort(); curXhr = null; }
  resetToIdle();
}

// ---------------------------------------------------------------------------
// 轮询状态
// ---------------------------------------------------------------------------
function startPolling() {
  clearTimeout(pollTimer);
  poll();
}
async function poll() {
  try {
    const r = await fetch("/api/tasks/" + encodeURIComponent(curTask));
    if (r.status === 404) { showError("任务不存在或已过期。"); return; }
    const st = await r.json();
    renderStatus(st);
    if (["done", "failed", "cancelled", "deleted"].includes(st.state)) return;
  } catch (_) { /* 网络抖动，下次再试 */ }
  pollTimer = setTimeout(poll, 2000);
}

const STAGE_TEXT = {
  queued: "排队中",
  preprocessing: "预处理（转 16k 音频）",
  transcribing: "转写中",
  postprocessing: "生成字幕",
};

function renderStatus(st) {
  if (st.state === "done") { renderDone(st); return; }
  if (st.state === "failed") {
    const e = st.error || {};
    showError(e.message || "任务失败。");
    return;
  }
  if (st.state === "cancelled") { showError("任务已取消。"); return; }
  if (st.state === "deleted") { showError("结果已被删除。"); return; }

  show("progress");
  el("p-name").textContent = st.original_filename || "（任务）";
  el("p-taskid-row").classList.remove("hidden");
  el("p-taskid").textContent = st.task_id;
  el("btn-cancel").classList.add("hidden");
  el("btn-retry-up").classList.add("hidden");

  if (st.state === "queued") {
    const pos = st.queue_position;
    setState(pos ? `排队中（第 ${pos} 位）` : "排队中", indeterminate());
    el("p-detail").textContent = "单 GPU 串行处理，轮到后自动开始。";
  } else if (st.state === "transcribing" && typeof st.progress_percent === "number") {
    // 分块级真实进度（GPU ASR/CPU 对齐交错完成，见 worker/asr）；首块完成前
    // progress_percent 仍为 null，走下面的不确定动画兜底。
    const pct = st.progress_percent;
    setState(`转写中 ${pct}%`, pct);
    const est = Math.round((st.audio_duration_sec || 0) * cfg.rtf);
    el("p-detail").textContent = est
      ? `音频时长 ${fmtSec(st.audio_duration_sec)}，预计转写约 ${fmtSec(est)}。`
      : "";
  } else {
    setState(STAGE_TEXT[st.state] || st.state, indeterminate());
    const est = Math.round((st.audio_duration_sec || 0) * cfg.rtf);
    el("p-detail").textContent = est
      ? `音频时长 ${fmtSec(st.audio_duration_sec)}，预计转写约 ${fmtSec(est)}。`
      : "";
  }
}

// 无分块进度时用动画条示意（progress_percent 为 P1）。
let barPhase = 0;
function indeterminate() { barPhase = (barPhase + 7) % 90; return 10 + barPhase; }

// ---------------------------------------------------------------------------
// 完成：预览 + 下载 + 删除
// ---------------------------------------------------------------------------
async function renderDone(st) {
  show("done");
  el("d-name").textContent = st.original_filename || st.task_id;
  const base = "/api/tasks/" + encodeURIComponent(st.task_id) + "/result";
  el("dl-srt").href = base + "?format=srt";
  el("dl-txt").href = base + "?format=txt";
  try {
    const r = await fetch(base + "?format=srt");
    if (r.ok) renderPreview(await r.text());
  } catch (_) {}
}

function renderPreview(srt) {
  const box = el("preview");
  box.textContent = "";
  const blocks = srt.replace(/\r/g, "").trim().split(/\n\n+/);
  for (const b of blocks) {
    const lines = b.split("\n");
    if (lines.length < 2) continue;
    const tm = /(\d{2}:\d{2}:\d{2}),\d{3}\s*-->/.exec(lines[1]);
    const div = document.createElement("div");
    div.className = "cue";
    const t = document.createElement("span");
    t.className = "t"; t.textContent = tm ? tm[1] : "";
    const x = document.createElement("span");
    x.className = "x"; x.textContent = lines.slice(2).join("\n");  // textContent 防 XSS
    div.append(t, x);
    box.appendChild(div);
  }
}

async function deleteResult() {
  if (!confirm("确定删除该结果？删除后不可恢复。")) return;
  try {
    await fetch("/api/tasks/" + encodeURIComponent(curTask), { method: "DELETE" });
  } catch (_) {}
  showError("结果已删除。", null, "已删除");
}

// ---------------------------------------------------------------------------
// 历史任务列表（服务端 /api/tasks，重开浏览器后据此找回）
// ---------------------------------------------------------------------------
const HIST_STATE = {
  queued: "排队中", preprocessing: "处理中", transcribing: "转写中",
  postprocessing: "生成字幕", done: "已完成", failed: "失败", cancelled: "已取消",
};
const ACTIVE_STATES = new Set(["queued", "preprocessing", "transcribing", "postprocessing"]);

async function loadHistory() {
  const box = el("history"), list = el("history-list");
  try {
    const r = await fetch("/api/tasks");
    if (!r.ok) { box.classList.add("hidden"); return; }
    const { tasks } = await r.json();
    if (!tasks || !tasks.length) { box.classList.add("hidden"); return; }
    list.textContent = "";
    for (const t of tasks) list.appendChild(historyItem(t));
    box.classList.remove("hidden");
  } catch (_) { box.classList.add("hidden"); }
}

function historyItem(t) {
  const li = document.createElement("li");
  li.className = "hist-item";
  li.onclick = () => openTask(t.task_id);

  const main = document.createElement("div");
  main.className = "hist-main";
  const name = document.createElement("div");
  name.className = "hist-name"; name.textContent = t.original_filename || t.task_id;
  const meta = document.createElement("div");
  meta.className = "hist-meta";
  const parts = [];
  if (t.created_at) parts.push(fmtWhen(t.created_at));
  if (t.audio_duration_sec) parts.push(fmtSec(t.audio_duration_sec));
  if (t.state === "queued" && t.queue_position) parts.push(`第 ${t.queue_position} 位`);
  meta.textContent = parts.join(" · ");
  main.append(name, meta);

  const badge = document.createElement("span");
  const active = ACTIVE_STATES.has(t.state);
  badge.className = "hist-badge " + (t.state === "done" ? "ok" : active ? "run" : "muted");
  badge.textContent = HIST_STATE[t.state] || t.state;

  const del = document.createElement("button");
  del.className = "hist-del"; del.title = "删除"; del.textContent = "✕";
  del.onclick = (ev) => deleteHistory(t.task_id, ev);

  li.append(main, badge, del);
  return li;
}

// 从历史/URL 恢复某任务：复用轮询渲染（done/failed/… 均由 renderStatus 处理）。
function openTask(id) {
  curTask = id;
  history.replaceState(null, "", "?task=" + encodeURIComponent(id));
  show("progress");
  el("p-name").textContent = "（恢复的任务）";
  el("p-detail").textContent = "";
  el("p-taskid-row").classList.remove("hidden");
  el("p-taskid").textContent = id;
  el("btn-cancel").classList.add("hidden");
  el("btn-retry-up").classList.add("hidden");
  startPolling();
}

async function deleteHistory(id, ev) {
  ev.stopPropagation();  // 不触发条目的“恢复”点击
  if (!confirm("删除该历史任务？删除后不可恢复。")) return;
  try { await fetch("/api/tasks/" + encodeURIComponent(id), { method: "DELETE" }); } catch (_) {}
  loadHistory();
}

// ---------------------------------------------------------------------------
// 渲染辅助
// ---------------------------------------------------------------------------
function show(name) {
  for (const k in S) S[k].classList.toggle("hidden", k !== name);
}
function setState(text, pct) {
  el("p-state").textContent = text;
  if (typeof pct === "number") setBar(pct);
}
function setBar(pct) { el("p-bar").style.width = Math.max(0, Math.min(100, pct)) + "%"; }
function showError(msg, hint, icon) {
  clearTimeout(pollTimer);
  show("error");
  el("e-msg").textContent = msg;
  const h = el("e-hint");
  if (hint) { h.textContent = hint; h.classList.remove("hidden"); } else h.classList.add("hidden");
  if (icon) document.querySelector(".err-icon").textContent = icon === "已删除" ? "🗑️" : "⚠️";
}
function resetToIdle() {
  clearTimeout(pollTimer);
  curTask = null; audioBlob = null;
  history.replaceState(null, "", location.pathname);
  el("file").value = "";
  el("preview").textContent = "";
  document.querySelector(".err-icon").textContent = "⚠️";
  show("idle");
  loadHistory();
}
function fmtSec(s) {
  s = Math.round(s || 0);
  const m = (s / 60) | 0, r = s % 60;
  return m ? `${m}分${r}秒` : `${r}秒`;
}
// ISO 时间 → 本地「MM-DD HH:mm」；解析失败原样返回。
function fmtWhen(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
