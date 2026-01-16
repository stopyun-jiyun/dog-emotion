// web/app.js (완성본: 자동예측 폭주/무한대기 방지 + 동시요청 락 + 안정화)

// ===== DOM =====
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let overlay = document.getElementById('overlay');

if (!overlay) {
  console.warn("overlay canvas not found — creating one dynamically");
  const frame = document.querySelector('.videoFrame');
  overlay = document.createElement('canvas');
  overlay.id = 'overlay';
  overlay.className = 'overlay';
  frame.appendChild(overlay);
}

const octx = overlay.getContext('2d');

const emotionEl = document.getElementById('emotion');
const confEl = document.getElementById('conf');
const warnEl = document.getElementById('warn');
const guideEl = document.getElementById('guideText');
const emojiEl = document.getElementById('emotionEmoji');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const autoChk = document.getElementById('autoChk');

const shotBtn = document.getElementById('shotBtn');
const saveBtn = document.getElementById('saveBtn');
const captionInput = document.getElementById('captionInput');
const previewImg = document.getElementById('previewImg');
const previewMeta = document.getElementById('previewMeta');
const galleryEl = document.getElementById('gallery');

// ===== State =====
let stream = null;
let timer = null;
const CONF_THRESHOLD = 0.7;

// ✅ 핵심 안정화 상태
let inFlight = false;             // /predict 동시 호출 방지 (폭주/무한대기 방지)
let autoEnabledByUser = false;    // 사용자가 체크박스를 "직접" 건드리기 전엔 auto 시작 금지
let consecutiveErrors = 0;        // 연속 에러 횟수 (자동 분석 잠시 멈추기)
const MAX_CONSEC_ERRORS = 3;

// ===== Emotion stability =====
let hist = [];
const HIST_N = 5;

function stableEmotion(newEmotion) {
  hist.push(newEmotion);
  if (hist.length > HIST_N) hist.shift();

  const count = {};
  for (const e of hist) count[e] = (count[e] || 0) + 1;

  return Object.entries(count).sort((a, b) => b[1] - a[1])[0][0];
}

// ===== Guides & Emojis =====
const ACTION_GUIDE = {
  alert: '👀 주변을 경계하고 있어요.\n조용한 환경을 만들어주고 무엇에 반응하는지 살펴보세요.',
  happy: '😊 기분이 좋아 보여요!\n칭찬해 주거나 가볍게 놀아주면 좋아요.',
  angry: '⚠️ 스트레스 상태일 수 있어요.\n자극을 줄이고 잠시 거리를 두세요.',
  frown: '😟 불안하거나 우울할 수 있어요.\n부드럽게 말을 걸어 안정감을 주세요.',
  relax: '😌 편안한 상태예요.\n현재 환경을 유지해 주세요.',
};

const EMOJI = {
  alert: '👀',
  happy: '😄',
  angry: '😾',
  frown: '🥺',
  relax: '😌',
  '-': '🐾',
};

// Optional: Korean label display (theme still uses English)
const LABEL_KO = {
  alert: '경계',
  happy: '행복',
  angry: '화남',
  frown: '시무룩',
  relax: '편안',
};

// ===== Theme =====
function setTheme(emotion) {
  document.body.classList.remove('emotion-theme', 'alert', 'happy', 'angry', 'frown', 'relax');
  if (['alert', 'happy', 'angry', 'frown', 'relax'].includes(emotion)) {
    document.body.classList.add('emotion-theme', emotion);
  }
}

// ===== Overlay drawing =====
function clearOverlay() {
  if (!overlay.width || !overlay.height) return;
  octx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawOverlayBox(data, stable) {
  if (!overlay.width || !overlay.height) return;

  clearOverlay();

  const box = data?.box_xyxy;
  if (!box || box.length !== 4) return;

  const [x1, y1, x2, y2] = box;
  const w = x2 - x1;
  const h = y2 - y1;

  const conf = typeof data.confidence === 'number' ? Math.round(data.confidence * 100) : null;
  const label = `${stable.toUpperCase()}${conf !== null ? `  ${conf}%` : ''}`;

  octx.lineWidth = 6;
  octx.strokeStyle = 'rgba(255, 115, 182, 0.95)';
  octx.strokeRect(x1, y1, w, h);

  octx.font = 'bold 22px Arial';
  const padX = 12;
  const boxH = 34;
  const textW = octx.measureText(label).width;

  let lx = Math.max(0, x1);
  let ly = y1 - boxH - 8;
  if (ly < 0) ly = y1 + 8;

  octx.fillStyle = 'rgba(255, 255, 255, 0.78)';
  octx.fillRect(lx, ly, textW + padX * 2, boxH);

  octx.fillStyle = 'rgba(31, 36, 48, 0.88)';
  octx.fillText(label, lx + padX, ly + 24);
}

// ===== Helpers =====
function setServerErrorUI(msg = '서버 오류') {
  emotionEl.textContent = msg;
  confEl.textContent = '';
  warnEl.classList.add('hidden');
  guideEl.textContent = '-';
  emojiEl.textContent = '🐾';
  setTheme('');
  clearOverlay();
}

function resetUI() {
  hist = [];
  consecutiveErrors = 0;

  emotionEl.textContent = '-';
  confEl.textContent = '0%';
  warnEl.classList.add('hidden');
  guideEl.textContent = '-';
  emojiEl.textContent = EMOJI['-'];
  setTheme('');
  clearOverlay();
}

// ===== Webcam start/stop =====
async function startWebcam() {
  if (stream) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;

    // ensure overlay matches real video size
    video.addEventListener('loadedmetadata', async () => {
      try { await video.play(); } catch {}
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
    }, { once: true });

    startBtn.disabled = true;
    stopBtn.disabled = false;

    // reset UI state
    resetUI();

    // ✅ 중요: 페이지 로드시 autoChk가 체크되어 있어도 자동 시작 금지
    // 사용자가 체크박스를 "직접" 눌러(autoEnabledByUser=true)야만 자동 시작
    if (autoEnabledByUser && autoChk?.checked) startAuto();

  } catch (err) {
    console.error('getUserMedia error:', err);
    setServerErrorUI('카메라 오류');
    alert(`카메라 오류: ${err.name || err}`);
  }
}

function stopWebcam() {
  stopAuto();

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  video.srcObject = null;
  hist = [];
  inFlight = false;
  consecutiveErrors = 0;
  clearOverlay();

  startBtn.disabled = false;
  stopBtn.disabled = true;

  emotionEl.textContent = '-';
  confEl.textContent = '0%';
  warnEl.classList.add('hidden');
  guideEl.textContent = '-';
  emojiEl.textContent = EMOJI['-'];
  setTheme('');
}

// ===== Auto loop =====
function startAuto() {
  stopAuto();
  // ✅ 너무 잦으면 서버 부담 커질 수 있어서 1s 유지 (원하면 1500~2000 권장)
  timer = setInterval(captureAndPredict, 1000);
}

function stopAuto() {
  if (timer) clearInterval(timer);
  timer = null;
}

// ===== Predict =====
async function captureAndPredict() {
  if (!stream) return;
  if (!video.videoWidth || !video.videoHeight) return;

  // ✅ 가장 중요: 요청 겹침 방지
  if (inFlight) return;
  inFlight = true;

  try {
    // model input 224x224
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, 0, 0, 224, 224);

    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.9));
    if (!blob) throw new Error('Failed to create image blob');

    const form = new FormData();
    form.append('file', blob, 'frame.jpg');

    // ✅ fetch (상대경로 OK: Render 도메인에서 그대로 호출)
    const resp = await fetch('/predict', { method: 'POST', body: form });

    if (!resp.ok) {
      // 서버가 보내는 json 에러가 있으면 보이게
      let detail = '';
      try {
        const j = await resp.json();
        detail = j?.detail ? ` (${typeof j.detail === 'string' ? j.detail : 'detail'})` : '';
      } catch {}
      throw new Error(`HTTP ${resp.status}${detail}`);
    }

    const data = await resp.json();

    consecutiveErrors = 0;

    const predicted = data.emotion ?? data.class ?? '-';
    const stable = stableEmotion(predicted);

    // UI text (Korean display) + emoji
    emotionEl.textContent = LABEL_KO[stable] ?? stable;
    emojiEl.textContent = EMOJI[stable] ?? '🐾';
    setTheme(stable);

    // confidence
    const conf = typeof data.confidence === 'number' ? data.confidence : 0;
    const pct = Math.min(conf * 100, 99.9).toFixed(1);
    confEl.textContent = `${pct}%`;

    const low = conf < CONF_THRESHOLD;
    warnEl.classList.toggle('hidden', !low);

    // guide
    const guide = ACTION_GUIDE[stable] ?? '행동지침을 준비 중입니다.';
    guideEl.textContent = low ? `⚠️ 참고용 결과입니다.\n${guide}` : guide;

    // overlay (box + label) — 원하면 주석 해제
    // drawOverlayBox(data, stable);

  } catch (e) {
    console.error('predict error:', e);
    consecutiveErrors += 1;

    // UI
    setServerErrorUI('서버 오류');

    // ✅ 자동 분석 중 연속 에러가 나면 자동 멈춤 (무한 폭주 방지)
    if (timer && consecutiveErrors >= MAX_CONSEC_ERRORS) {
      stopAuto();
      if (autoChk) autoChk.checked = false;
      guideEl.textContent = '⚠️ 서버 응답이 불안정해서 자동 분석을 잠시 멈췄어요.\n다시 체크해서 재시도해 주세요.';
    }
  } finally {
    inFlight = false;
  }
}

// ===== Events =====
startBtn?.addEventListener('click', startWebcam);
stopBtn?.addEventListener('click', stopWebcam);

autoChk?.addEventListener('change', () => {
  // ✅ 사용자가 직접 토글했을 때만 auto 허용
  autoEnabledByUser = true;

  if (!stream) return;
  autoChk.checked ? startAuto() : stopAuto();
});

// =========================
// Screenshot / Local Gallery
// =========================
let lastShot = null;

function nowText() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
}

function loadPosts() {
  try { return JSON.parse(localStorage.getItem('dog_posts') || '[]'); }
  catch { return []; }
}

function savePosts(posts) {
  localStorage.setItem('dog_posts', JSON.stringify(posts));
}

function renderGallery() {
  const posts = loadPosts();
  galleryEl.innerHTML = '';

  if (posts.length === 0) {
    galleryEl.innerHTML = `<div style="opacity:.7;font-size:13px;">아직 기록이 없어요. 📸 스크린샷을 찍고 한 줄 기록을 남겨봐!</div>`;
    return;
  }

  for (const p of posts) {
    const card = document.createElement('div');
    card.className = 'card';

    const img = document.createElement('img');
    img.src = p.dataUrl;

    const meta = document.createElement('div');
    meta.className = 'meta';

    const top = document.createElement('div');
    top.className = 'top';
    top.textContent = `${p.time}  |  ${p.emotion}  |  ${p.conf}`;

    const cap = document.createElement('div');
    cap.className = 'caption';
    cap.textContent = p.caption || '(설명 없음)';

    const actions = document.createElement('div');
    actions.className = 'actions';

    const dl = document.createElement('button');
    dl.className = 'smallBtn';
    dl.textContent = '다운로드';
    dl.onclick = () => downloadWithCaption(p);

    const del = document.createElement('button');
    del.className = 'smallBtn';
    del.textContent = '삭제';
    del.onclick = () => {
      const posts2 = loadPosts().filter(x => x.id !== p.id);
      savePosts(posts2);
      renderGallery();
    };

    actions.appendChild(dl);
    actions.appendChild(del);

    meta.appendChild(top);
    meta.appendChild(cap);
    meta.appendChild(actions);

    card.appendChild(img);
    card.appendChild(meta);

    galleryEl.appendChild(card);
  }
}

function takeScreenshot() {
  if (!stream) {
    alert('웹캠을 먼저 시작해줘!');
    return;
  }
  if (!video.videoWidth || !video.videoHeight) return;

  // 16:9 screenshot
  const w = 960, h = 540;
  const temp = document.createElement('canvas');
  temp.width = w;
  temp.height = h;
  const tctx = temp.getContext('2d');
  tctx.drawImage(video, 0, 0, w, h);

  const dataUrl = temp.toDataURL('image/jpeg', 0.92);

  const emotion = emotionEl.textContent || '-';
  const conf = confEl.textContent || '0%';
  const time = nowText();

  lastShot = { dataUrl, emotion, conf, time };

  previewImg.src = dataUrl;
  previewImg.style.display = 'block';
  previewMeta.textContent = `${time}\n감정: ${emotion}\n신뢰도: ${conf}\n\n설명을 입력하고 저장을 누르세요.`;

  saveBtn.disabled = false;
}

function saveScreenshotPost() {
  if (!lastShot) return;

  const caption = captionInput.value.trim();

  const post = {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    dataUrl: lastShot.dataUrl,
    emotion: lastShot.emotion,
    conf: lastShot.conf,
    time: lastShot.time,
    caption,
  };

  const posts = loadPosts();
  posts.unshift(post);
  savePosts(posts);

  captionInput.value = '';
  saveBtn.disabled = true;
  previewMeta.textContent = '저장 완료 ✅ 아래 기록에서 확인하세요.';

  renderGallery();
}

async function downloadWithCaption(post) {
  const img = new Image();
  img.src = post.dataUrl;

  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = rej;
  });

  const srcW = img.naturalWidth;
  const srcH = img.naturalHeight;

  const pad = Math.round(srcW * 0.03);

  const titleSize = Math.max(18, Math.round(srcW * 0.028));
  const bodySize  = Math.max(16, Math.round(srcW * 0.024));

  const footerH = Math.round(srcH * 0.18);

  const c = document.createElement('canvas');
  c.width = srcW;
  c.height = srcH + footerH;
  const g = c.getContext('2d');

  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, c.width, c.height);

  g.drawImage(img, 0, 0, srcW, srcH);

  const y0 = srcH;

  g.fillStyle = 'rgba(0,0,0,0.06)';
  g.fillRect(0, y0, c.width, 2);

  function strokeFillText(text, x, y, font, fill = 'rgba(20,20,20,0.95)') {
    g.font = font;
    g.lineWidth = Math.max(4, Math.round(srcW * 0.004));
    g.strokeStyle = 'rgba(255,255,255,0.95)';
    g.fillStyle = fill;

    g.strokeText(text, x, y);
    g.fillText(text, x, y);
  }

  function drawWrapped(text, x, y, maxWidth, lineHeight, font) {
    g.font = font;
    const words = text.split(' ');
    let line = '';

    for (let i = 0; i < words.length; i++) {
      const test = line + words[i] + ' ';
      if (g.measureText(test).width > maxWidth && i > 0) {
        strokeFillText(line.trim(), x, y, font);
        line = words[i] + ' ';
        y += lineHeight;
      } else {
        line = test;
      }
    }
    strokeFillText(line.trim(), x, y, font);
    return y + lineHeight;
  }

  g.textBaseline = 'top';

  const title = `🐶 ${post.time}   |   ${post.emotion}   |   ${post.conf}`;
  const caption = (post.caption && post.caption.trim().length)
    ? `📝 ${post.caption.trim()}`
    : `📝 (메모 없음)`;

  const maxW = c.width - pad * 2;

  let y = y0 + pad;
  y = drawWrapped(title, pad, y, maxW, Math.round(titleSize * 1.25), `700 ${titleSize}px Arial`);
  drawWrapped(caption, pad, y, maxW, Math.round(bodySize * 1.45), `500 ${bodySize}px Arial`);

  const outUrl = c.toDataURL('image/jpeg', 0.92);
  const a = document.createElement('a');
  a.href = outUrl;
  a.download = `dog_${post.time.replace(/[: ]/g, '_')}.jpg`;
  a.click();
}

shotBtn?.addEventListener('click', takeScreenshot);
saveBtn?.addEventListener('click', saveScreenshotPost);

// initial gallery render
renderGallery();

// ✅ 선택: 브라우저가 체크 상태를 기억해도 시작 시 자동이 돌지 않게 "시작 전엔" auto를 무력화하고 싶으면 아래 한 줄을 켜도 됨
// if (autoChk) autoChk.checked = false;
