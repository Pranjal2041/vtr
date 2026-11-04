/* Visual Token Routing project page */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const EX = Object.fromEntries(EXAMPLES.map(e => [e.key, e]));
const fmt = n => Math.round(n).toLocaleString('en-US');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const mulberry32 = a => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const COL = { full: '#1f5fa8', sparse: '#e07a1f', blind: '#6b7a8c', lam: '#b4531a', ok: '#16a34a', kept: '#34d399' };

/* ───────────── Simulated desktop shared by the interactive figures ───────────── */
const TOY = (() => {
  const W = 576, H = 336, T = 24, GW = 24, GH = 14, N = GW * GH;
  const NAME = 'player_voice_full.mp3';
  const ALL = { typing: true, video: true, clock: true };
  const cache = new Map();
  const typedLen = (t, on) => on.typing ? Math.min(NAME.length, t % (NAME.length + 4)) : 9;
  const ballPos = ph => [56 + (ph * 53) % 240, 86 + 40 * Math.sin(ph * 0.9)];

  function draw(ctx, t, on) {
    const g = ctx.createLinearGradient(0, 0, W, H); g.addColorStop(0, '#334155'); g.addColorStop(1, '#1e293b');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 312, W, 24);
    ctx.fillStyle = '#38bdf8'; ctx.fillRect(8, 317, 14, 14);
    const sec = on.clock ? 7 + t : 7;
    ctx.fillStyle = '#e2e8f0'; ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText(`10:${42 + Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`, 500, 328);
    // media player
    ctx.fillStyle = '#e5e7eb'; ctx.fillRect(12, 12, 316, 236);
    ctx.fillStyle = '#374151'; ctx.font = '600 12px Inter, sans-serif'; ctx.fillText('Media Player', 22, 27);
    const vx = 16, vy = 36, vw = 308, vh = 184, ph = on.video ? t : 0;
    ['#f8fafc', '#facc15', '#22d3ee', '#22c55e', '#d946ef', '#ef4444', '#2563eb'].forEach((c, i) => { ctx.fillStyle = c; ctx.fillRect(vx + i * vw / 7, vy, vw / 7 + 1, vh); });
    ctx.save(); ctx.beginPath(); ctx.rect(vx, vy, vw, vh); ctx.clip();
    ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 10;
    ctx.beginPath(); ctx.moveTo(vx - 60 + (ph * 37) % 420, vy + vh); ctx.lineTo(vx + 60 + (ph * 37) % 420, vy); ctx.stroke();
    const [bx, by] = ballPos(ph); ctx.fillStyle = '#111827'; ctx.beginPath(); ctx.arc(bx, by, 16, 0, 7); ctx.fill();
    ctx.restore();
    ctx.fillStyle = '#d1d5db'; ctx.fillRect(16, 224, 308, 18);
    ctx.fillStyle = '#0891b2'; ctx.fillRect(40, 231, 30 + (ph * 9) % 270, 4);
    ctx.fillStyle = '#374151'; ctx.beginPath(); ctx.moveTo(22, 228); ctx.lineTo(32, 233); ctx.lineTo(22, 238); ctx.fill();
    // save dialog
    ctx.fillStyle = '#f8fafc'; ctx.fillRect(348, 60, 200, 172);
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1; ctx.strokeRect(348.5, 60.5, 199, 171);
    ctx.fillStyle = '#e2e8f0'; ctx.fillRect(349, 61, 198, 22);
    ctx.fillStyle = '#1f2937'; ctx.font = '600 12px Inter, sans-serif'; ctx.fillText('Save As', 358, 76);
    ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = '#475569'; ctx.fillText('File name:', 360, 106);
    ctx.fillStyle = '#fff'; ctx.fillRect(360, 112, 176, 26);
    ctx.strokeStyle = '#0891b2'; ctx.strokeRect(360.5, 112.5, 175, 25);
    ctx.fillStyle = '#0f172a'; ctx.font = '11px "JetBrains Mono", monospace';
    const txt = NAME.slice(0, typedLen(t, on)); ctx.fillText(txt, 365, 129);
    ctx.fillRect(365 + ctx.measureText(txt).width + 1, 117, 1, 16);
    ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = '#475569'; ctx.fillText('Save as type:', 360, 160);
    ctx.fillStyle = '#fff'; ctx.fillRect(360, 166, 176, 22); ctx.strokeStyle = '#cbd0d8'; ctx.strokeRect(360.5, 166.5, 175, 21);
    ctx.fillStyle = '#334155'; ctx.fillText('Audio (*.mp3)', 366, 181);
    ctx.fillStyle = '#0891b2'; ctx.fillRect(392, 200, 66, 22); ctx.fillStyle = '#e2e8f0'; ctx.fillRect(468, 200, 66, 22);
    ctx.fillStyle = '#fff'; ctx.font = '600 11px Inter, sans-serif'; ctx.fillText('Save', 412, 215);
    ctx.fillStyle = '#334155'; ctx.fillText('Cancel', 482, 215);
  }

  function frame(t, on = ALL) {
    const key = `${t}|${+on.typing}${+on.video}${+on.clock}`;
    if (cache.has(key)) return cache.get(key);
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    draw(ctx, t, on);
    const f = { cv, data: ctx.getImageData(0, 0, W, H).data };
    if (cache.size > 300) cache.delete(cache.keys().next().value);
    cache.set(key, f);
    return f;
  }

  function changed(t, on = ALL) {
    const a = frame(t - 1, on).data, b = frame(t, on).data, m = new Uint8Array(N);
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
      outer: for (let y = r * T; y < (r + 1) * T; y++) for (let x = c * T; x < (c + 1) * T; x++) {
        const i = (y * W + x) * 4;
        if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 24) { m[r * GW + c] = 1; break outer; }
      }
    }
    return m;
  }

  const cat = (c, r) => (r >= 4 && r <= 5 && c >= 15 && c <= 22) ? 'field' : (c <= 13 && r <= 10) ? 'video' : (r === 13 && c >= 20) ? 'clock' : 'static';
  // Illustrative keep probabilities: high only for changes in the field the agent types into.
  const keepProb = (t, on = ALL) => {
    const ch = changed(t, on), p = new Float32Array(N);
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
      const k = cat(c, r), x = ch[r * GW + c];
      p[r * GW + c] = k === 'field' ? (x ? 0.93 : 0.06) : x ? (k === 'video' ? 0.07 : k === 'clock' ? 0.05 : 0.1) : 0.01;
    }
    return p;
  };
  const gates = (p, seed) => { const rng = mulberry32(seed); return Uint8Array.from(p, v => rng() < v ? 1 : 0); };

  // regions an object segmenter would return on this screen
  function objects(t, on = ALL) {
    const [bx, by] = ballPos(on.video ? t : 0);
    const rects = [[360, 112, 176, 26], [392, 200, 66, 22], [468, 200, 66, 22], [496, 316, 66, 18], [18, 224, 18, 18], [bx - 16, by - 16, 32, 32]];
    const m = new Uint8Array(N);
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++)
      for (const [x, y, w, h] of rects) if (x < (c + 1) * T && x + w > c * T && y < (r + 1) * T && y + h > r * T) m[r * GW + c] = 1;
    return m;
  }

  function caretCell(t, on = ALL) {
    const ctx = document.createElement('canvas').getContext('2d'); ctx.font = '11px "JetBrains Mono", monospace';
    return [Math.min(22, Math.floor((366 + ctx.measureText(NAME.slice(0, typedLen(t, on))).width) / T)), 5];
  }

  // draw a frame with dropped tokens dimmed and kept tokens outlined
  function paint(ctx, t, mask, opt = {}) {
    const on = opt.on || ALL, sx = ctx.canvas.width / W, sy = ctx.canvas.height / H;
    ctx.save(); ctx.scale(sx, sy);
    ctx.drawImage(frame(t, on).cv, 0, 0);
    if (mask) {
      for (let i = 0; i < N; i++) if (!mask[i]) { ctx.fillStyle = 'rgba(2,6,23,0.72)'; ctx.fillRect((i % GW) * T, Math.floor(i / GW) * T, T, T); }
      ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1 / sx;
      for (let c = 1; c < GW; c++) { ctx.beginPath(); ctx.moveTo(c * T, 0); ctx.lineTo(c * T, H); ctx.stroke(); }
      for (let r = 1; r < GH; r++) { ctx.beginPath(); ctx.moveTo(0, r * T); ctx.lineTo(W, r * T); ctx.stroke(); }
      ctx.strokeStyle = opt.color || COL.kept; ctx.lineWidth = Math.max(2, 2.5 / sx);
      for (let i = 0; i < N; i++) if (mask[i]) ctx.strokeRect((i % GW) * T + 1.5, Math.floor(i / GW) * T + 1.5, T - 3, T - 3);
    }
    if (opt.sel) { ctx.strokeStyle = '#f97316'; ctx.lineWidth = 3.5; ctx.strokeRect(opt.sel[0] * T + 1, opt.sel[1] * T + 1, T - 2, T - 2); }
    ctx.restore();
  }

  return { W, H, T, GW, GH, N, ALL, ballPos, frame, changed, cat, keepProb, gates, objects, caretCell, paint };
})();

/* ───────────── Routed frames: before / after slider ───────────── */
(function sees() {
  const order = ['dota', 'minecraft', 'candy', 'gta', 'gpredict', 'psychopy', 'librecad'];
  const tabs = $('#sees-tabs'), box = $('#compare'), top = $('#cmp-top'), handle = $('#cmp-handle');
  let drag = false;
  const setPos = p => { p = Math.max(0, Math.min(100, p)); top.style.clipPath = `inset(0 0 0 ${p}%)`; handle.style.left = p + '%'; };
  const show = key => {
    const e = EX[key];
    $('#cmp-base').src = `assets/examples/${key}_prev.jpg`;
    $('#cmp-kept').src = `assets/examples/${key}_kept.jpg`;
    $('#cmp-count').innerHTML = `<b>${fmt(e.kept)}</b> of ${fmt(e.grid[0] * e.grid[1])} tokens kept`;
    $('#cmp-caption').textContent = `${e.name}. ${e.setup} ${e.note}`;
    $$('.tab', tabs).forEach(t => t.classList.toggle('active', t.dataset.key === key));
  };
  order.forEach(k => { const b = document.createElement('button'); b.className = 'tab'; b.dataset.key = k; b.textContent = EX[k].name; b.onclick = () => show(k); tabs.appendChild(b); });
  const move = ev => { const r = box.getBoundingClientRect(); setPos((ev.clientX - r.left) / r.width * 100); };
  box.addEventListener('pointerdown', ev => { drag = true; box.setPointerCapture(ev.pointerId); move(ev); });
  box.addEventListener('pointermove', ev => { if (drag) move(ev); });
  box.addEventListener('pointerup', () => { drag = false; });
  show('dota');
  const t0 = performance.now() + 500;
  const sweep = now => { const u = Math.max(0, Math.min(1, (now - t0) / 1800)); setPos(88 - 58 * (1 - Math.pow(1 - u, 3))); if (u < 1 && !drag) requestAnimationFrame(sweep); };
  setPos(88); requestAnimationFrame(sweep);
})();

/* ───────────── Visual tokens in context ───────────── */
(function context() {
  const MODES = {
    desktop: { n: 100, dense: 2040, vtr: 807.9, unit: i => `Step ${i} of 100`, result: ['2.5× fewer visual tokens', '2.4× lower inference cost', '2.3× faster prefill'] },
    game: { n: 1800, dense: 220, vtr: 38.2, unit: i => `0:${String(Math.floor(i / 30)).padStart(2, '0')} of 1:00 at 30 fps`, result: ['5.8× fewer visual tokens', '3.3× lower inference cost', '2.9× faster prefill'] },
  };
  const B = 10, cvD = $('#race-dense'), cvV = $('#race-vtr');
  let mode = 'desktop', raf = null, started = false;
  const vtrAt = (m, i) => (i % B === 0 ? m.dense : (m.vtr * B - m.dense) / (B - 1));
  const fit = cv => { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; return cv.getContext('2d'); };
  const draw = k => {
    const m = MODES[mode];
    for (const [cv, fn, col] of [[cvD, () => m.dense, '#9ca3af'], [cvV, i => vtrAt(m, i), '#0891b2']]) {
      const ctx = fit(cv), W = cv.width, H = cv.height, w = W / m.n;
      ctx.fillStyle = '#f0f0f2'; ctx.fillRect(0, 0, W, H); ctx.fillStyle = col;
      for (let i = 0; i < k; i++) { const h = H * fn(i) / m.dense; ctx.fillRect(i * w, H - h, Math.max(1, w - (m.n <= 200 ? 1.5 : 0)), h); }
    }
    let vsum = 0; for (let i = 0; i < k; i++) vsum += vtrAt(m, i);
    $('#race-dense-total').firstChild.nodeValue = fmt(m.dense * k);
    $('#race-vtr-total').firstChild.nodeValue = fmt(vsum);
    $('#race-clock').textContent = m.unit(k);
    $('#race-result').innerHTML = k >= m.n ? m.result.map(s => `<span><b>${s.split(' ')[0]}</b> ${s.split(' ').slice(1).join(' ')}</span>`).join('') : '';
  };
  const run = () => {
    cancelAnimationFrame(raf);
    const m = MODES[mode], t0 = performance.now();
    const tick = now => { const u = Math.min(1, (now - t0) / 5000); draw(Math.round(u * m.n)); if (u < 1) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  };
  $$('#race-tabs .tab').forEach(b => b.onclick = () => { mode = b.dataset.mode; $$('#race-tabs .tab').forEach(t => t.classList.toggle('active', t === b)); run(); });
  $('#race-replay').onclick = run;
  draw(0);
  new IntersectionObserver((es, ob) => { if (es[0].isIntersecting && !started) { started = true; run(); ob.disconnect(); } }, { threshold: 0.4 }).observe($('.race'));
  addEventListener('resize', () => draw(MODES[mode].n));
})();

/* ───────────── Spatio-temporal router ───────────── */
(function router() {
  const { GW, T, N } = TOY, K = 3, NS = 'http://www.w3.org/2000/svg';
  const svg = $('#rt-svg');
  const S = (tag, a = {}, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
  const X = (s, a, p = svg) => { const e = S('text', a, p); e.textContent = s; return e; };
  // italic math with sub- and superscripts, e.g. 'v_{t−3,j}'
  function M(src, a, p = svg, size = 16) {
    const el = S('text', { 'font-size': size, ...a, class: 'm ' + (a.class || '') }, p);
    let shift = 0;
    src.replace(/([_^])\{([^}]*)\}|([^_^]+)/g, (m, op, inner, plain) => {
      const to = op === '_' ? size * 0.32 : op === '^' ? -size * 0.4 : 0;
      const ts = S('tspan', op ? { dy: to - shift, 'font-size': Math.round(size * 0.68) } : { dy: to - shift }, el);
      ts.textContent = op ? inner : plain; shift = to;
    });
    return el;
  }

  // layout
  const FX = 64, FW = 180, FH = 105, CS = 7.5, RY = [62, 182, 302, 422];
  const EX = 266, EW = 86, GX = 376, BX = 632, BW = 24, SEG = 80, SY = k => 120 + k * 88, CY = 292;
  const L = [{ x: 772, h: 200, n: '1,024' }, { x: 832, h: 100, n: '256' }, { x: 886, h: 16, n: '1' }], LW = 14;
  const SIG = 936, GA = { x: 972, y0: 200, y1: 384, w: 18 }, ND = { x: 1080, y: CY }, GB = { x: 1022, y: 140, w: 116, h: 62 };

  S('defs').innerHTML =
    '<marker id="rt-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#b6bcc6"/></marker>' +
    '<marker id="rt-ah-on" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e07a1f"/></marker>';
  [['Observations', FX + FW / 2], ['Vision encoder', EX + EW / 2], ['Concatenate', BX + BW / 2 + 14],
   ['Router MLP', 832], ['Keep probability', GA.x + GA.w / 2], ['Gate', ND.x]].forEach(([s, x]) => X(s, { x, y: 30, class: 'hd', 'text-anchor': 'middle' }));

  S('text', { x: GX + FW / 2, y: 30, class: 'hd', 'text-anchor': 'middle' }).innerHTML = 'Visual tokens <tspan class="m" font-size="16">v</tspan> ∈ ℝ<tspan class="m" font-size="11" dy="-6">D</tspan>';
  // observations, vision encoder, token grids
  const rows = [];
  for (let k = 0; k <= K; k++) {
    const y = RY[k], cy = y + FH / 2, cur = k === K;
    const g = S('g', { class: 'row' + (cur ? ' cur' : '') });
    M(cur ? 'o_{t}' : `o_{t−${K - k}}`, { x: FX - 10, y: cy + 6, 'text-anchor': 'end', class: cur ? 'curlab' : '' }, g, 20);
    const fimg = S('image', { x: FX, y, width: FW, height: FH, preserveAspectRatio: 'none', class: 'pick' }, g);
    S('rect', { x: FX, y, width: FW, height: FH, class: 'fbox' }, g);
    const a1 = S('line', { x1: FX + FW + 4, y1: cy, x2: EX - 4, y2: cy, class: 'flow' }, g);
    const enc = S('polygon', { points: `${EX},${y + 6} ${EX + EW},${y + 24} ${EX + EW},${y + FH - 24} ${EX},${y + FH - 6}`, class: 'enc' }, g);
    X('Vision', { x: EX + EW / 2 - 3, y: cy - 4, class: 'enc-t', 'text-anchor': 'middle' }, g);
    X('encoder', { x: EX + EW / 2 - 3, y: cy + 12, class: 'enc-t', 'text-anchor': 'middle' }, g);
    const a2 = S('line', { x1: EX + EW + 4, y1: cy, x2: GX - 4, y2: cy, class: 'flow' }, g);
    const gimg = S('image', { x: GX, y, width: FW, height: FH, preserveAspectRatio: 'none', class: 'pick grid' }, g);
    const hf = S('rect', { width: CS + 2, height: CS + 2, class: 'hl' }, g), hg = S('rect', { width: CS + 2, height: CS + 2, class: 'hl' }, g);
    rows.push({ y, fimg, gimg, a1, a2, enc, hf, hg });
  }
  X('shared weights', { x: EX + EW / 2 - 3, y: RY[K] + FH + 22, class: 'sm', 'text-anchor': 'middle' });

  // router MLP: concatenated input, two hidden layers, one output
  const fpoly = (x0, a0, b0, x1, a1, b1) => S('polygon', { points: `${x0},${a0} ${x1},${a1} ${x1},${b1} ${x0},${b0}`, class: 'funnel' });
  const fun = [fpoly(BX + BW, SY(0), SY(K) + SEG, L[0].x, CY - L[0].h / 2, CY + L[0].h / 2)];
  for (let i = 1; i < 3; i++) fun.push(fpoly(L[i - 1].x + LW, CY - L[i - 1].h / 2, CY + L[i - 1].h / 2, L[i].x, CY - L[i].h / 2, CY + L[i].h / 2));
  rows.forEach(R => { R.mp = S('path', { class: 'flow merge' }); });
  const segs = [];
  for (let k = 0; k <= K; k++) {
    const g = S('g', { class: 'seg' + (k === K ? ' cur' : '') });
    const cells = Array.from({ length: 10 }, (_, i) => S('rect', { x: BX, y: SY(k) + i * 8, width: BW, height: 8 }, g));
    S('rect', { x: BX, y: SY(k), width: BW, height: SEG, class: 'sbox' }, g);
    M(k === K ? 'v_{t,j}' : `v_{t−${K - k},j}`, { x: BX + BW + 6, y: SY(k) + SEG / 2 + 5 }, g, 15);
    segs.push({ g, cells });
  }
  M('x_{t,j}', { x: BX + BW / 2, y: SY(K) + SEG + 26, 'text-anchor': 'middle' }, svg, 19);
  S('text', { x: BX + BW / 2, y: SY(K) + SEG + 50, 'text-anchor': 'middle', class: 'sm' }).innerHTML = '(<tspan class="m">K</tspan> + 1) × <tspan class="m">D</tspan> values';
  const lay = L.map(l => {
    X(l.n, { x: l.x + LW / 2, y: CY + L[0].h / 2 + 22, class: 'sm', 'text-anchor': 'middle' });
    return S('rect', { x: l.x, y: CY - l.h / 2, width: LW, height: l.h, rx: 4, class: 'layer' });
  });
  X('GELU', { x: (L[0].x + LW + L[1].x) / 2, y: CY - L[0].h / 2 - 2, class: 'sm', 'text-anchor': 'middle' });
  X('GELU', { x: (L[1].x + LW + L[2].x) / 2, y: CY - L[1].h / 2 - 8, class: 'sm', 'text-anchor': 'middle' });
  M('f_{φ}', { x: 832, y: CY + L[0].h / 2 + 48, 'text-anchor': 'middle' }, svg, 19);

  // keep probability, gate, output
  const late = S('g', { class: 'late' });
  const ls = S('line', { x1: L[2].x + LW + 2, y1: CY, x2: SIG - 18, y2: CY, class: 'flow' }, late);
  const sig = S('circle', { cx: SIG, cy: CY, r: 15, class: 'op' }, late); X('σ', { x: SIG, y: CY + 6, class: 'op-t', 'text-anchor': 'middle' }, late);
  const lg = S('line', { x1: SIG + 16, y1: CY, x2: GA.x - 4, y2: CY, class: 'flow' }, late);
  S('rect', { x: GA.x, y: GA.y0, width: GA.w, height: GA.y1 - GA.y0, rx: 4, class: 'gbg' }, late);
  const gfill = S('rect', { x: GA.x, y: GA.y0, width: GA.w, height: GA.y1 - GA.y0, rx: 4, class: 'gfill' }, late);
  M('p_{t,j}', { x: GA.x + GA.w / 2, y: GA.y0 - 12, 'text-anchor': 'middle' }, late, 18);
  const pval = X('', { x: GA.x + GA.w / 2, y: GA.y1 + 26, 'text-anchor': 'middle', class: 'pval' }, late);
  const lp = S('path', { d: `M${GA.x + GA.w + 3},${CY} H${ND.x - 19}`, class: 'flow' }, late);
  const lb = S('path', { d: `M${GA.x + GA.w + 16},${CY} V${GB.y + GB.h / 2} H${GB.x - 3}`, class: 'flow' }, late);
  const gb = S('rect', { x: GB.x, y: GB.y, width: GB.w, height: GB.h, rx: 10, class: 'gb' }, late);
  S('text', { x: GB.x + GB.w / 2, y: GB.y + 24, 'text-anchor': 'middle' }, late).innerHTML = '<tspan class="m">g</tspan> ∼ Bernoulli(<tspan class="m">p</tspan>)';
  const gv = X('', { x: GB.x + GB.w / 2, y: GB.y + 48, 'text-anchor': 'middle', class: 'gv' }, late);
  const lgn = S('path', { d: `M${ND.x},${GB.y + GB.h + 3} V${ND.y - 19}`, class: 'flow' }, late);
  M('g_{t,j}', { x: ND.x + 8, y: (GB.y + GB.h + ND.y) / 2 + 4 }, late, 16);
  const node = S('circle', { cx: ND.x, cy: ND.y, r: 16, class: 'op' }, late); X('×', { x: ND.x, y: ND.y + 6, class: 'op-t', 'text-anchor': 'middle' }, late);
  const lv = S('path', { class: 'flow' }, late);
  M('v_{t,j}', { x: 860, y: 572, 'text-anchor': 'middle' }, late, 17);
  X('the token being routed', { x: 912, y: 572, class: 'sm' }, late);
  const lo = S('line', { x1: ND.x + 18, y1: ND.y, x2: ND.x + 44, y2: ND.y, class: 'flow' }, late);
  const out1 = X('', { x: ND.x + 52, y: ND.y - 2, class: 'out1' }, late), out2 = S('text', { x: ND.x + 52, y: ND.y + 18, class: 'out2' }, late);

  // token grids and token vectors computed from each simulated frame
  const MIX = (() => { const rng = mulberry32(5); return Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => (rng() * 2 - 1) * 0.9)); })();
  const cache = new Map();
  function enc(t) {
    if (cache.has(t)) return cache.get(t);
    const f = TOY.frame(t), d = f.data, W = TOY.W, lum = i => 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2];
    const gc = document.createElement('canvas'); gc.width = FW * 2; gc.height = FH * 2; const g = gc.getContext('2d');
    g.fillStyle = '#0b1220'; g.fillRect(0, 0, gc.width, gc.height);
    const feats = [];
    for (let j = 0; j < N; j++) {
      const c = j % GW, r = Math.floor(j / GW), n = T * T;
      let R = 0, G = 0, B = 0, l2 = 0, gx = 0, gy = 0; const q = [0, 0, 0, 0];
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
        const i = ((r * T + y) * W + c * T + x) * 4, l = lum(i);
        R += d[i]; G += d[i + 1]; B += d[i + 2]; q[(y >= T / 2) * 2 + (x >= T / 2)] += l; l2 += l * l;
        if (x) gx += Math.abs(l - lum(i - 4));
        if (y) gy += Math.abs(l - lum(i - W * 4));
      }
      R /= n; G /= n; B /= n;
      const lm = (q[0] + q[1] + q[2] + q[3]) / n, sd = Math.sqrt(Math.max(0, l2 / n - lm * lm));
      g.fillStyle = `rgb(${R | 0},${G | 0},${B | 0})`; g.beginPath(); g.roundRect(c * CS * 2 + 1.5, r * CS * 2 + 1.5, CS * 2 - 3, CS * 2 - 3, 2.5); g.fill();
      const raw = [R / 127.5 - 1, G / 127.5 - 1, B / 127.5 - 1, ...q.map(v => v / (n / 4) / 127.5 - 1), sd / 50 - 0.5, gx / n / 20 - 0.5, gy / n / 20 - 0.5];
      feats.push(MIX.map(row => Math.tanh(1.6 * row.reduce((a, w, i) => a + w * raw[i], 0))));
    }
    const out = { frame: f.cv.toDataURL(), grid: gc.toDataURL(), feats };
    cache.set(t, out); if (cache.size > 12) cache.delete(cache.keys().next().value);
    return out;
  }
  const cmap = v => { const a = Math.min(1, Math.abs(v)), [r, g, b] = v < 0 ? [31, 95, 168] : [224, 122, 31]; return `rgb(${Math.round(248 + (r - 248) * a)},${Math.round(250 + (g - 250) * a)},${Math.round(252 + (b - 252) * a)})`; };

  let t = 8, seed = 1, sel = null, follow = 'field', playing = true, run = 0, running = false;
  const region = () => {
    if (follow === 'field') return TOY.caretCell(t);
    if (follow === 'video') { const [x, y] = TOY.ballPos(t); return [Math.floor(x / T), Math.floor(y / T)]; }
    if (follow === 'clock') return [22, 13];
    if (follow === 'static') return [18, 11];
    return sel;
  };
  function place(j) {
    const c = j % GW, r = Math.floor(j / GW);
    rows.forEach((R, k) => {
      for (const [h, x0] of [[R.hf, FX], [R.hg, GX]]) { h.setAttribute('x', x0 + c * CS - 1); h.setAttribute('y', R.y + r * CS - 1); }
      const x0 = GX + (c + 1) * CS + 1, y0 = R.y + (r + 0.5) * CS, x1 = BX - 3, y1 = SY(k) + SEG / 2;
      R.mp.setAttribute('d', `M${x0},${y0} C${x0 + 60},${y0} ${x1 - 60},${y1} ${x1},${y1}`);
      enc(t - K + k).feats[j].forEach((v, i) => segs[k].cells[i].setAttribute('fill', cmap(v)));
    });
    lv.setAttribute('d', `M${GX + (c + 0.5) * CS},${RY[K] + (r + 1) * CS + 1} V580 H${ND.x} V${ND.y + 19}`);
  }
  const on = (els, v = true) => [].concat(els).forEach(e => e.classList.toggle('on', v));

  function tokens(g, j) {
    const ids = []; g.forEach((v, i) => v && ids.push(i));
    $('#rt-count').textContent = `${ids.length} of ${N}`;
    const shown = ids.slice(0, 26);
    $('#rt-tokens').innerHTML = `<span class="tok text">instruction</span><span class="tok text">o<sub>t−1</sub>, a<sub>t−1</sub> …</span>` +
      shown.map((i, n) => `<span class="tok vis${i === j ? ' sel' : ''}" style="animation-delay:${n * 25}ms">pos ${i}</span>`).join('') +
      (ids.length > shown.length ? `<span class="tok gap">+${ids.length - shown.length} more</span>` : '') +
      `<span class="tok gap">${N - ids.length} dropped</span>`;
  }

  async function cycle() {
    const my = ++run; running = true;
    const w = ms => sleep(ms).then(() => { if (my !== run) throw 0; });
    try {
      const [c, r] = region(), j = r * GW + c;
      const p = TOY.keepProb(t), g = TOY.gates(p, seed * 1000 + t), kept = g[j] === 1;
      $$('.on', svg).forEach(e => e.classList.remove('on'));
      svg.classList.remove('sweeping'); gb.classList.remove('kept');
      rows.forEach((R, k) => { const e = enc(t - K + k); R.fimg.setAttribute('href', e.frame); R.gimg.setAttribute('href', e.grid); });
      place(j);
      gfill.style.transform = 'scaleY(0)'; pval.textContent = ''; gv.textContent = ''; out1.textContent = ''; out2.innerHTML = '';
      $$('#rt-chips button').forEach(b => b.classList.toggle('active', b.dataset.r === follow));
      await w(300);
      rows.forEach(R => on([R.a1, R.enc])); await w(600);
      rows.forEach(R => on([R.a2, R.gimg])); await w(700);
      rows.forEach(R => on(R.mp));
      for (const s of segs) { s.g.classList.add('on'); await w(170); }
      await w(150);
      for (let i = 0; i < 3; i++) { on([fun[i], lay[i]]); await w(280); }
      on([ls, sig]); await w(250); on(lg);
      gfill.style.transform = `scaleY(${Math.max(0.015, p[j])})`; pval.textContent = p[j].toFixed(2); await w(750);
      on([lp, lb, gb]);
      for (let i = 0; i < 6; i++) { gv.textContent = `g = ${i % 2}`; await w(90); }
      gv.textContent = `g = ${g[j]}`; gb.classList.toggle('kept', kept); on([lgn, lv, node]); await w(550);
      on(lo); out1.textContent = kept ? 'kept' : 'dropped'; out1.setAttribute('class', 'out1 ' + (kept ? 'kept' : 'dropped'));
      out2.innerHTML = kept ? '<tspan class="m">p</tspan> · <tspan class="m">v</tspan>' : 'removed';
      await w(1100);
      // the same router on every position of o_t
      svg.classList.add('sweeping');
      const mctx = $('#rt-map').getContext('2d'), mask = new Uint8Array(N);
      for (let i = 0; i < N; i += 6) {
        for (let q = i; q < Math.min(N, i + 6); q++) mask[q] = g[q];
        const jj = Math.min(N - 1, i + 5); place(jj);
        TOY.paint(mctx, t, mask, { sel: [jj % GW, Math.floor(jj / GW)] });
        await w(18);
      }
      place(j); svg.classList.remove('sweeping');
      TOY.paint(mctx, t, g, { sel: [c, r] }); tokens(g, j);
      await w(2200);
      running = false;
      if (playing) { t++; cycle(); }
    } catch (e) { if (e !== 0) throw e; }
  }

  svg.addEventListener('click', ev => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM().inverse());
    for (let k = 0; k <= K; k++) for (const x0 of [FX, GX])
      if (q.x >= x0 && q.x < x0 + FW && q.y >= RY[k] && q.y < RY[k] + FH) {
        sel = [Math.floor((q.x - x0) / CS), Math.floor((q.y - RY[k]) / CS)]; follow = null; cycle(); return;
      }
  });
  $$('#rt-chips button').forEach(b => b.onclick = () => { follow = b.dataset.r; cycle(); });
  $('#rt-play').onclick = () => {
    playing = !playing; $('#rt-play').textContent = playing ? '❚❚ Pause' : '▶ Play';
    if (playing && !running) { t++; cycle(); }
  };
  $('#rt-next').onclick = () => { t++; cycle(); };
  $('#rt-resample').onclick = () => { seed++; cycle(); };
  // first frame drawn statically, animation starts when the figure is on screen
  rows.forEach((R, k) => { const e = enc(t - K + k); R.fimg.setAttribute('href', e.frame); R.gimg.setAttribute('href', e.grid); });
  place(TOY.caretCell(t)[1] * GW + TOY.caretCell(t)[0]);
  TOY.paint($('#rt-map').getContext('2d'), t, null);
  new IntersectionObserver((es, ob) => { if (es[0].isIntersecting) { cycle(); ob.disconnect(); } }, { threshold: 0.3 }).observe(svg);
})();

/* ───────────── Anchors ───────────── */
(function anchors() {
  const n = 12, B = 6, { N } = TOY;
  const box = $('#an-frames');
  let mode = 'anchors';
  const cells = Array.from({ length: n }, (_, i) => {
    const d = document.createElement('div'); d.className = 'an-frame';
    const cv = document.createElement('canvas'); cv.width = 288; cv.height = 168;
    const s = document.createElement('small');
    d.append(cv, s); box.appendChild(d); return { d, cv, s };
  });
  function render() {
    const counts = [];
    cells.forEach(({ d, cv, s }, i) => {
      const t = 20 + i, anchor = mode === 'anchors' ? i % B === 0 : i === 0;
      const g = anchor ? null : TOY.gates(TOY.keepProb(t), 77 + t);
      TOY.paint(cv.getContext('2d'), t, g);
      const k = anchor ? N : g.reduce((a, b) => a + b, 0);
      counts.push([k, anchor]);
      d.classList.toggle('anchor', anchor);
      s.textContent = anchor ? `anchor · ${N}` : `${k} kept`;
      d.classList.remove('show');
      setTimeout(() => d.classList.add('show'), 80 + i * 110);
    });
    const total = n * N, sum = counts.reduce((a, [k]) => a + k, 0);
    $('#an-full').style.width = '100%';
    $('#an-full-n').textContent = fmt(total);
    $('#an-vtr').innerHTML = counts.map(([k, a]) => `<div style="width:${(k / total * 100).toFixed(2)}%;background:${a ? '#1f5fa8' : '#e07a1f'};border-right:1px solid #fff"></div>`).join('');
    $('#an-vtr-n').textContent = fmt(sum);
  }
  $$('#an-tabs .tab').forEach(b => b.onclick = () => { mode = b.dataset.m; $$('#an-tabs .tab').forEach(x => x.classList.toggle('active', x === b)); render(); });
  render();
})();

/* ───────────── Pixel-diff and VTR on the simulated desktop ───────────── */
(function comparison() {
  const on = { typing: true, video: true, clock: false };
  let t = 30;
  function step() {
    t++;
    const diff = TOY.changed(t, on);
    const vtr = diff.map((v, i) => v && TOY.cat(i % TOY.GW, Math.floor(i / TOY.GW)) === 'field' ? 1 : 0);
    TOY.paint($('#cv-diff').getContext('2d'), t, diff, { on });
    TOY.paint($('#cv-vtr').getContext('2d'), t, vtr, { on });
    $('#cnt-diff').innerHTML = `<b>${diff.reduce((a, b) => a + b, 0)}</b> of ${TOY.N} tokens kept`;
    $('#cnt-vtr').innerHTML = `<b>${vtr.reduce((a, b) => a + b, 0)}</b> of ${TOY.N} tokens kept`;
  }
  $$('#switches .switch').forEach(s => s.onclick = () => { on[s.dataset.key] = !on[s.dataset.key]; s.classList.toggle('on', on[s.dataset.key]); });
  step(); setInterval(step, 900);
})();

/* ───────────── How it works: step tabs ───────────── */
$$('#step-tabs .step-tab').forEach(b => b.onclick = () => {
  $$('#step-tabs .step-tab').forEach(x => x.classList.toggle('active', x === b));
  $$('.step-panel').forEach(p => p.classList.toggle('active', p.dataset.step === b.dataset.step));
  document.dispatchEvent(new CustomEvent('step', { detail: +b.dataset.step }));
});

/* canvas text with sub- and superscripts: 'ℓ^{full}', 'π_{θ}' */
function math(ctx, src, x, y, size, color, align = 'center') {
  const parts = [];
  src.replace(/([_^])\{([^}]*)\}|([^_^]+)/g, (m, op, inner, plain) => { parts.push(op ? [inner, op] : [plain, '']); });
  const font = (s, k) => `${k && s.length > 1 ? '' : 'italic '}${k ? Math.round(size * 0.66) : size}px "Times New Roman", serif`;
  let w = 0; for (const [s, k] of parts) { ctx.font = font(s, k); w += ctx.measureText(s).width; }
  let cx = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  for (const [s, k] of parts) { ctx.font = font(s, k); ctx.fillText(s, cx, y + (k === '_' ? size * 0.3 : k === '^' ? -size * 0.38 : 0)); cx += ctx.measureText(s).width; }
}

/* ───────────── 01 Joint training ───────────── */
(function jointTraining() {
  const cv = $('#jt-canvas'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  const TW = 120, TH = 70, TX = [20, 150, 280, 410], CY = { full: 110, blind: 290, sparse: 470 };
  const BX = 575, BW = 200, BH = 84, LX = 835, RX = 900, RW = 440;
  const DG = { x: RX, y: 46, w: RW, h: 150 }, LS = { x: RX, y: 290, w: 190, h: 104 }, LM = { x: RX + RW - 190, y: 290, w: 190, h: 104 };
  const T0 = 40;
  let phase = 0, mode = 'train', seed = 3, lam = 0.5, t0 = performance.now();
  const thumbs = {};
  function makeThumbs() {
    for (const key of ['full', 'blind', 'sparse']) thumbs[key] = TX.map((_, k) => {
      const c = document.createElement('canvas'); c.width = TW * 2; c.height = TH * 2;
      const t = T0 + k;
      let mask = null;
      if (k > 0 && key === 'blind') mask = new Uint8Array(TOY.N);
      if (k > 0 && key === 'sparse') mask = TOY.gates(TOY.keepProb(t), seed * 97 + t);
      TOY.paint(c.getContext('2d'), t, mask);
      return c;
    });
  }
  const text = (s, x, y, o = {}) => { ctx.font = o.font || '500 22px Inter, sans-serif'; ctx.fillStyle = o.color || '#1f2937'; ctx.textAlign = o.align || 'center'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); };
  const flow = (pts, color, on, w = 3) => {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.globalAlpha *= on ? 1 : 0.22;
    if (on) { ctx.setLineDash([10, 8]); ctx.lineDashOffset = -(performance.now() / 25) % 18; }
    ctx.beginPath(); ctx.moveTo(...pts[0]); for (const p of pts.slice(1)) ctx.lineTo(...p); ctx.stroke();
    const [a, b] = pts.slice(-2), ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    ctx.setLineDash([]); ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - 14 * Math.cos(ang - 0.45), b[1] - 14 * Math.sin(ang - 0.45)); ctx.lineTo(b[0] - 14 * Math.cos(ang + 0.45), b[1] - 14 * Math.sin(ang + 0.45)); ctx.fill();
    ctx.restore();
  };
  const box = (x, y, w, h, color, glow) => {
    ctx.save(); if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 24; }
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 14); ctx.fillStyle = glow ? '#fff7ed' : '#fff'; ctx.fill();
    ctx.lineWidth = glow ? 4 : 2.2; ctx.strokeStyle = color; ctx.stroke(); ctx.restore();
  };

  function draw() {
    const infer = mode === 'infer', tr = !infer, p = phase;
    ctx.clearRect(0, 0, W, H);
    for (const [s, x] of [['Inputs', (TX[0] + TX[3] + TW) / 2], ['Models', BX + BW / 2], ['Losses', LX]]) text(s, x, 20, { font: '600 19px Inter, sans-serif', color: '#9ca3af' });
    for (const key of ['full', 'blind', 'sparse']) {
      const cy = CY[key], ref = key !== 'sparse', by = cy - BH / 2;
      ctx.save(); ctx.globalAlpha = infer && ref ? 0.13 : 1;
      thumbs[key].forEach((c, k) => {
        ctx.drawImage(c, TX[k], cy - TH / 2, TW, TH);
        ctx.lineWidth = k === 0 ? 4 : 1.5; ctx.strokeStyle = k === 0 ? COL.full : '#9aa3ae'; ctx.strokeRect(TX[k], cy - TH / 2, TW, TH);
      });
      const fwd = p === 0 || infer;
      flow([[TX[3] + TW + 8, cy], [BX - 6, cy]], '#64748b', fwd);
      if (key === 'sparse') {
        box(BX, by, BW / 2, BH, COL.sparse, tr && (p === 1 || p === 2));
        box(BX + BW / 2, by, BW / 2, BH, COL.sparse, tr && p === 1);
        text('Router', BX + BW / 4, cy - 13, { font: '600 22px Inter, sans-serif' });
        math(ctx, 'φ', BX + BW / 4, cy + 18, 24, '#6b7280');
        text('VLM', BX + 3 * BW / 4, cy - 13, { font: '600 22px Inter, sans-serif' });
        math(ctx, 'π_{θ}', BX + 3 * BW / 4, cy + 16, 24, '#6b7280');
        if (tr && p === 2) text('fixed', BX + 3 * BW / 4, by + BH + 20, { font: '700 18px Inter, sans-serif', color: '#b91c1c' });
      } else {
        box(BX, by, BW, BH, COL[key], tr && p === 1);
        text(key === 'full' ? 'Full model' : 'Blind model', BX + BW / 2, cy - 13, { font: '600 22px Inter, sans-serif' });
        text(key === 'full' ? 'all tokens' : 'anchors only', BX + BW / 2, cy + 17, { font: '18px Inter, sans-serif', color: '#6b7280' });
      }
      flow([[BX + BW + 6, cy], [LX - 16, cy]], '#64748b', fwd);
      ctx.fillStyle = COL[key]; ctx.beginPath(); ctx.arc(LX, cy, 11, 0, 7); ctx.fill();
      if (infer && !ref) text('action', LX + 6, cy - 30, { font: '600 20px Inter, sans-serif', color: COL.sparse, align: 'right' });
      else math(ctx, `ℓ^{${key}}`, LX + 6, cy - 32, 28, COL[key], 'right');
      if (tr && p === 1) {
        const yb = by + BH + (key === 'sparse' ? 40 : 24);
        flow([[LX, cy + 14], [LX, yb], [BX + BW / 2, yb], [BX + BW / 2, by + BH + 4]], COL[key], true);
        text('CE', LX + 26, yb, { font: '700 18px Inter, sans-serif', color: COL[key] });
      }
      ctx.restore();
    }
    // degradation, router loss, adaptive weight
    ctx.save(); ctx.globalAlpha = infer ? 0.07 : 1;
    const deg = tr && (p === 2 || p === 3);
    for (const key of ['full', 'blind', 'sparse']) flow([[LX + 14, CY[key]], [DG.x - 6, DG.y + 70 + (key === 'blind' ? 12 : key === 'sparse' ? 24 : 0)]], '#94a3b8', deg, 2);
    box(DG.x, DG.y, DG.w, DG.h, COL.sparse, deg);
    text('Measure degradation', DG.x + DG.w / 2, DG.y + 30, { font: '600 22px Inter, sans-serif' });
    math(ctx, 'i = ℓ^{blind} − ℓ^{full}', DG.x + DG.w * 0.27, DG.y + 76, 25, '#1f2937');
    math(ctx, 'd = ℓ^{sparse} − ℓ^{full}', DG.x + DG.w * 0.73, DG.y + 76, 25, '#1f2937');
    math(ctx, 'ℒ_{deg} = Σ[i]_{+}d / Σ[i]_{+}', DG.x + DG.w * 0.27, DG.y + 120, 23, '#1f2937');
    math(ctx, 'r̄ = Σd / Σi', DG.x + DG.w * 0.73, DG.y + 120, 23, '#1f2937');
    const mx = LS.x + LS.w / 2, lx = LM.x + LM.w / 2, rb = DG.y + DG.h;
    flow([[mx, rb + 4], [mx, LS.y - 6]], COL.lam, tr && p === 2);
    math(ctx, 'ℒ_{deg}', mx + 12, (rb + LS.y) / 2, 23, COL.lam, 'left');
    flow([[lx, rb + 4], [lx, LM.y - 6]], COL.lam, tr && p === 3);
    math(ctx, 'r̄', lx + 12, (rb + LM.y) / 2, 24, COL.lam, 'left');
    flow([[LM.x - 4, LS.y + LS.h / 2], [LS.x + LS.w + 6, LS.y + LS.h / 2]], COL.lam, tr && p === 2);
    math(ctx, 'λ', (LM.x + LS.x + LS.w) / 2, LS.y + LS.h / 2 - 18, 24, COL.lam);
    box(LS.x, LS.y, LS.w, LS.h, COL.lam, tr && p === 2);
    text('Router loss', mx, LS.y + 30, { font: '600 21px Inter, sans-serif' });
    math(ctx, 'ℒ_{tok} + λℒ_{deg}', mx, LS.y + 70, 25, '#1f2937');
    box(LM.x, LM.y, LM.w, LM.h, COL.lam, tr && p === 3);
    text('Adaptive weight', lx, LM.y + 30, { font: '600 21px Inter, sans-serif' });
    text(`λ = ${lam.toFixed(2)}`, lx, LM.y + 70, { font: 'italic 25px "Times New Roman", serif' });
    const rx = BX + BW / 4, py = 612;
    flow([[mx, LS.y + LS.h + 4], [mx, py], [rx, py], [rx, CY.sparse + BH / 2 + 6]], COL.lam, tr && p === 2);
    text('updates only the router', (mx + rx) / 2, py + 22, { font: '18px Inter, sans-serif', color: tr && p === 2 ? COL.lam : '#9ca3af' });
    ctx.restore();
    if (infer) {
      text('At inference, only the router', RX + RW / 2, 486, { font: '600 24px Inter, sans-serif', color: '#15803d' });
      text('and the VLM run.', RX + RW / 2, 520, { font: '600 24px Inter, sans-serif', color: '#15803d' });
    }
    ctx.fillStyle = COL.full; ctx.fillRect(20, 632, 16, 16);
    text('anchor, kept in full', 46, 641, { font: '18px Inter, sans-serif', color: '#4b5563', align: 'left' });
  }

  function setPhase(p) {
    phase = p;
    if (p === 0 || p === 2) { seed++; makeThumbs(); }
    if (p === 3) lam = Math.max(0, lam + 0.2 * ((Math.random() * 0.12 + 0.04) - 0.1));
    $$('#jt-phases li').forEach((li, i) => li.classList.toggle('on', mode === 'train' && i === p));
  }
  $$('#jt-mode .tab').forEach(b => b.onclick = () => {
    mode = b.dataset.m; $$('#jt-mode .tab').forEach(x => x.classList.toggle('active', x === b));
    $('#jt-phases').classList.toggle('off', mode === 'infer');
    $('#jt-note').textContent = mode === 'infer' ? 'At inference, only the router and the VLM run, so the reference models add no cost.' : 'The reference models are used only during training. Click a step to show it.';
    setPhase(0); t0 = performance.now();
  });
  $$('#jt-phases li').forEach((li, i) => { li.onclick = () => { if (mode === 'train') { setPhase(i); t0 = performance.now(); } }; });
  makeThumbs(); setPhase(0);
  const loop = now => {
    if (mode === 'train' && now - t0 > 3200) { setPhase((phase + 1) % 4); t0 = now; }
    if ($('.step-panel[data-step="0"]').classList.contains('active')) draw();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
})();

/* ───────────── 02 Measuring degradation ───────────── */
(function degradation() {
  const svg = $('#dg-axis'), NS = 'http://www.w3.org/2000/svg';
  let type = 'needs';
  const X = l => 60 + l / 2.8 * 640;
  const el = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt !== undefined) e.innerHTML = txt; svg.appendChild(e); return e; };
  function render() {
    const k = $('#dg-k').value / 100;
    $('#dg-kval').textContent = Math.round(k * 100) + '%';
    const lf = type === 'needs' ? 0.6 : 0.8, lb = type === 'needs' ? 2.4 : 0.72;
    const D = type === 'needs' ? (lb - lf) * 0.95 : 0.25;
    const ls = lf + D * Math.pow(1 - k, 6);
    const i = lb - lf, d = ls - lf, w = Math.max(0, i);
    svg.innerHTML = '';
    const Y = 128;
    el('line', { x1: 40, y1: Y, x2: 730, y2: Y, stroke: '#4b5563', 'stroke-width': 2 });
    el('path', { d: `M730,${Y - 7} L742,${Y} L730,${Y + 7} Z`, fill: '#4b5563' });
    el('text', { x: 742, y: Y + 30, 'text-anchor': 'end', 'font-size': 15, fill: '#6b7280', 'font-family': 'Inter' }, 'loss');
    // brackets
    const xf = X(lf), xb = X(lb), xs = X(ls);
    if (Math.abs(xb - xf) > 2) {
      el('path', { d: `M${xf},62 L${xf},54 L${xb},54 L${xb},62`, fill: 'none', stroke: COL.full, 'stroke-width': 2 });
      el('text', { x: (xf + xb) / 2, y: 42, 'text-anchor': 'middle', 'font-size': 20, fill: COL.full, 'font-family': 'Times New Roman', 'font-style': 'italic' }, `i<tspan font-size="14" dy="5">n</tspan><tspan dy="-5"> = ${i.toFixed(2)}</tspan>`);
    }
    el('path', { d: `M${xf},${Y + 24} L${xf},${Y + 32} L${xs},${Y + 32} L${xs},${Y + 24}`, fill: 'none', stroke: COL.sparse, 'stroke-width': 2 });
    el('text', { x: Math.max(xf + 6, (xf + xs) / 2), y: Y + 58, 'text-anchor': 'middle', 'font-size': 20, fill: COL.sparse, 'font-family': 'Times New Roman', 'font-style': 'italic' }, `d<tspan font-size="14" dy="5">n</tspan><tspan dy="-5"> = ${d.toFixed(2)}</tspan>`);
    for (const [x, c, lab, dy] of [[xf, COL.full, 'full', -22], [xb, COL.blind, 'blind', type === 'needs' ? -22 : 44], [xs, COL.sparse, 'sparse', type === 'needs' ? -22 : 66]]) {
      el('circle', { cx: x, cy: Y, r: 9, fill: c, stroke: '#fff', 'stroke-width': 2 });
      el('text', { x, y: Y + dy, 'text-anchor': 'middle', 'font-size': 19, fill: c, 'font-family': 'Times New Roman', 'font-style': 'italic' }, `ℓ<tspan font-size="13" dy="-8" font-style="normal">${lab}</tspan>`);
    }
    $('#dg-read').innerHTML = [
      [i.toFixed(2), 'i<sub>n</sub>, help of the history'],
      [d.toFixed(2), 'd<sub>n</sub>, harm of dropping'],
      [i > 0 ? (d / i).toFixed(2) : 'n/a', 'd<sub>n</sub> / i<sub>n</sub> for this token'],
      [w.toFixed(2), 'weight [i<sub>n</sub>]<sub>+</sub> in ℒ<sub>deg</sub>'],
    ].map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join('');
  }
  $('#dg-k').oninput = render;
  $$('#dg-type .tab').forEach(b => b.onclick = () => { type = b.dataset.t; $$('#dg-type .tab').forEach(x => x.classList.toggle('active', x === b)); render(); });
  render();
})();

/* ───────────── 03 Adaptive weight ───────────── */
(function adaptiveWeight() {
  const cv = $('#aw-chart'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  // validation r_bar of the VTR training run (r* = 0.1)
  const REAL = [[50,0.5325],[100,0.4774],[150,0.2792],[200,0.3683],[250,0.2714],[300,0.2759],[350,0.19],[400,0.157],[450,0.0815],[500,0.0741],[550,0.0259],[600,0.2827],[650,0.1597],[700,0.3053],[750,0.3161],[800,0.2924],[850,0.2366],[900,0.1951],[950,0.1518],[1000,0.1054],[1050,0.1327],[1100,0.062],[1150,0.0714],[1200,0.0663],[1250,0.0631],[1300,0.1126],[1350,0.1719],[1400,0.072],[1450,0.05],[1500,-0.0362],[1550,-0.0475],[1600,0.0115],[1650,0.0962],[1700,0.0428],[1750,0.0181],[1800,0.033],[1850,0.0813],[1900,0.1139],[1950,0.1564],[2000,0.0754]];
  const STEPS = 2000, LMAX = 3;
  let sim = [], raf = null;
  // toy degradation curve r(k) = 0.95 (1 − k)^6 for kept fraction k, with the paper's dual ascent on λ
  function simulate(rs) {
    const rng = mulberry32(11); let k = 0.09, lam = 0.5, ema = null; const out = [];
    for (let s = 0; s < STEPS; s++) {
      const r = 0.95 * Math.pow(1 - k, 6), dr = -5.7 * Math.pow(1 - k, 5);
      k = Math.min(1, Math.max(0.01, k - 0.0008 * (1 + lam * dr)));
      const rn = r + (rng() + rng() + rng() - 1.5) * 0.04;
      lam = Math.max(0, lam + 0.2 * (rn < rs ? 10 : 1) * (rn - rs));
      ema = ema === null ? rn : 0.95 * ema + 0.05 * rn;
      out.push([s, ema, k, lam]);
    }
    return out;
  }
  const P1 = { x0: 86, x1: 830, y0: 52, y1: 262 }, P2 = { x0: 86, x1: 830, y0: 340, y1: 530 };
  const px = s => P1.x0 + s / STEPS * (P1.x1 - P1.x0);
  const py = (v, P, hi) => P.y1 - Math.max(0, Math.min(hi, v)) / hi * (P.y1 - P.y0);
  const T = (s, x, y, f, c, a = 'center') => { ctx.font = f; ctx.fillStyle = c; ctx.textAlign = a; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); };
  function frame(P, ticks, hi) {
    for (const v of ticks) {
      const y = py(v, P, hi);
      ctx.strokeStyle = '#efeff3'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(P.x0, y); ctx.lineTo(P.x1, y); ctx.stroke();
      T(v.toFixed(1), P.x0 - 12, y, '19px Inter, sans-serif', '#6b7280', 'right');
    }
    ctx.strokeStyle = '#c9c9d2'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P.x0, P.y0); ctx.lineTo(P.x0, P.y1); ctx.lineTo(P.x1, P.y1); ctx.stroke();
  }
  function curve(n, P, hi, idx, color, w, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath();
    for (let i = 0; i < n; i += 3) { const x = px(sim[i][0]), y = py(sim[i][idx], P, hi); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  function draw(n) {
    const rs = $('#aw-r').value / 100, last = sim[n - 1];
    ctx.clearRect(0, 0, W, H);
    // top: degradation ratio
    frame(P1, [0, 0.2, 0.4, 0.6], 0.6);
    const ys = py(rs, P1, 0.6);
    ctx.strokeStyle = COL.ok; ctx.lineWidth = 2.5; ctx.setLineDash([10, 7]); ctx.beginPath(); ctx.moveTo(P1.x0, ys); ctx.lineTo(P1.x1, ys); ctx.stroke(); ctx.setLineDash([]);
    math(ctx, `r* = ${rs.toFixed(2)}`, P1.x1 + 12, ys, 23, COL.ok, 'left');
    for (const [s, v] of REAL) { if (s > n) break; ctx.fillStyle = 'rgba(107,114,128,0.55)'; ctx.beginPath(); ctx.arc(px(s), py(v, P1, 0.6), 6, 0, 7); ctx.fill(); }
    curve(n, P1, 0.6, 1, COL.sparse, 3.5);
    math(ctx, 'r̄', 34, (P1.y0 + P1.y1) / 2, 30, COL.sparse);
    ctx.fillStyle = COL.sparse; ctx.fillRect(P1.x0 + 16, 16, 26, 4);
    T(`simulated r̄ = ${last[1].toFixed(2)}`, P1.x0 + 50, 18, '600 19px Inter, sans-serif', COL.sparse, 'left');
    ctx.fillStyle = 'rgba(107,114,128,0.7)'; ctx.beginPath(); ctx.arc(P1.x0 + 330, 18, 6, 0, 7); ctx.fill();
    T('VTR training run, r* = 0.1', P1.x0 + 344, 18, '19px Inter, sans-serif', '#6b7280', 'left');
    // bottom: kept fraction and λ
    frame(P2, [0, 0.2, 0.4, 0.6], 0.6);
    for (const v of [0, 1, 2, 3]) T(String(v), P2.x1 + 12, py(v, P2, LMAX), '19px Inter, sans-serif', COL.lam, 'left');
    ctx.strokeStyle = COL.lam; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P2.x1, P2.y0); ctx.lineTo(P2.x1, P2.y1); ctx.stroke();
    curve(n, P2, LMAX, 3, COL.lam, 2.5, [3, 6]);
    curve(n, P2, 0.6, 2, '#0891b2', 4);
    ctx.save(); ctx.translate(30, (P2.y0 + P2.y1) / 2); ctx.rotate(-Math.PI / 2); T('kept fraction', 0, 0, '600 20px Inter, sans-serif', '#0891b2'); ctx.restore();
    math(ctx, 'λ', W - 24, (P2.y0 + P2.y1) / 2, 30, COL.lam);
    ctx.fillStyle = '#0891b2'; ctx.fillRect(P2.x0 + 16, P2.y0 - 32, 26, 5);
    T(`kept fraction = ${last[2].toFixed(2)}`, P2.x0 + 50, P2.y0 - 30, '600 19px Inter, sans-serif', '#0891b2', 'left');
    ctx.strokeStyle = COL.lam; ctx.lineWidth = 2.5; ctx.setLineDash([3, 6]); ctx.beginPath(); ctx.moveTo(P2.x0 + 330, P2.y0 - 30); ctx.lineTo(P2.x0 + 356, P2.y0 - 30); ctx.stroke(); ctx.setLineDash([]);
    T(`λ = ${last[3].toFixed(2)}, right axis`, P2.x0 + 364, P2.y0 - 30, '600 19px Inter, sans-serif', COL.lam, 'left');
    for (const s of [0, 500, 1000, 1500, 2000]) T(s.toLocaleString('en-US'), px(s), P2.y1 + 24, '19px Inter, sans-serif', '#6b7280');
    T('training step', (P2.x0 + P2.x1) / 2, P2.y1 + 54, '20px Inter, sans-serif', '#4b5563');
  }
  function run() {
    cancelAnimationFrame(raf); sim = simulate($('#aw-r').value / 100);
    const t0 = performance.now();
    const tick = now => { const u = Math.min(1, (now - t0) / 3500); draw(Math.max(2, Math.round(u * STEPS))); if (u < 1) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  }
  $('#aw-r').oninput = () => { $('#aw-rval').textContent = ($('#aw-r').value / 100).toFixed(2); run(); };
  $('#aw-run').onclick = run;
  sim = simulate(0.1); draw(STEPS);
  document.addEventListener('step', e => { if (e.detail === 2) run(); });
})();

/* ───────────── 04 Initialization ───────────── */
(function initialization() {
  const t = 45, ch = TOY.changed(t), ob = TOY.objects(t);
  const label = ch.map((v, i) => v && ob[i] ? 1 : 0);
  const vtr = ch.map((v, i) => v && TOY.cat(i % TOY.GW, Math.floor(i / TOY.GW)) === 'field' ? 1 : 0);
  TOY.paint($('#in-changed').getContext('2d'), t, ch, { color: '#f59e0b' });
  TOY.paint($('#in-objects').getContext('2d'), t, ob, { color: '#60a5fa' });
  TOY.paint($('#in-label').getContext('2d'), t, label);
  TOY.paint($('#in-vtr').getContext('2d'), t, vtr);
})();

/* ───────────── Results charts ───────────── */
(function charts() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.color = '#5a5a63';
  const C = { dense: '#9ca3af', vtr: '#0891b2', h264: '#d97706', pdiff: '#7c3aed', frozen: '#dc2626' };
  const NAMES = { dense: 'Dense', vtr: 'VTR', h264: 'H.264', pdiff: 'Pixel-diff', frozen: 'Frozen router' };
  const P = (...a) => a.map(([x, y]) => ({ x, y }));
  const D = {
    osworld: { y: 'Score', ymax: 20,
      tok: { dense: P([2040, 18.56], [1008, 15.19], [510, 13.39], [252, 6.17]), vtr: P([807.9, 18.10], [391, 13.73], [171.8, 5.76]), h264: P([1386.5, 9.44]), pdiff: P([485.9, 5.25]), frozen: P([421.5, 4.91]) },
      cost: { dense: P([1.0, 18.56], [0.523, 15.19], [0.55, 13.39], [0.349, 6.17]), vtr: P([0.413, 18.10], [0.351, 13.73], [0.354, 5.76]), h264: P([0.753, 9.44]), pdiff: P([0.421, 5.25]), frozen: P([0.276, 4.91]) } },
    cuaworld: { y: 'Average score', ymax: 20,
      tok: { dense: P([1946.2, 16.61], [966.5, 15.33], [510, 11.75], [252, 8.98]), vtr: P([674.3, 16.34], [328.8, 12.92], [148.1, 9.25]), h264: P([1367, 12.43]), pdiff: P([406.8, 9.33]) },
      cost: { dense: P([1.0, 16.61], [0.643, 15.33], [0.55, 11.75], [0.274, 8.98]), vtr: P([0.55, 16.34], [0.454, 12.92], [0.394, 9.25]), h264: P([0.587, 12.43]), pdiff: P([0.368, 9.33]) } },
    games: { y: 'Click accuracy (%)', ymax: 6,
      tok: { dense: P([220, 4.74], [112, 3.82], [55, 3.63], [28, 3.44]), vtr: P([38.2, 4.90], [16.4, 4.54], [9.5, 4.31]), h264: P([150.3, 2.54]), pdiff: P([67.3, 4.59]), frozen: P([51.6, 4.14]) },
      cost: { dense: P([1.0, 4.74], [0.586, 3.82], [0.368, 3.63], [0.264, 3.44]), vtr: P([0.303, 4.90], [0.22, 4.54], [0.193, 4.31]), h264: P([0.733, 2.54]), pdiff: P([0.415, 4.59]), frozen: P([0.355, 4.14]) } },
  };
  const sets = data => Object.entries(data).map(([k, pts]) => ({
    label: NAMES[k], data: pts, showLine: k === 'dense' || k === 'vtr', borderColor: C[k], backgroundColor: C[k],
    pointRadius: k === 'vtr' ? 6 : 5, pointStyle: { dense: 'circle', vtr: 'rect', h264: 'triangle', pdiff: 'rectRot', frozen: 'star' }[k],
    borderWidth: k === 'vtr' ? 3 : 2, order: k === 'vtr' ? 0 : 1,
  }));
  const scatter = (id, xTitle, log) => new Chart($(id), {
    type: 'scatter', data: { datasets: [] },
    options: { maintainAspectRatio: false, animation: { duration: 400 },
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, font: { size: 11 } } } },
      scales: { x: { type: log ? 'logarithmic' : 'linear', title: { display: true, text: xTitle }, ticks: log ? { callback: v => [10, 30, 100, 300, 1000, 2000].includes(v) ? v : '' } : {} },
                y: { min: 0, title: { display: true, text: '' } } } },
  });
  const cTok = scatter('#chart-tokens', 'Visual tokens per image', true);
  const cCost = scatter('#chart-cost', 'Inference cost relative to full resolution', false);
  const setD = d => {
    for (const [c, key] of [[cTok, 'tok'], [cCost, 'cost']]) { c.data.datasets = sets(D[d][key]); c.options.scales.y.max = D[d].ymax; c.options.scales.y.title.text = D[d].y; c.update(); }
    $$('#res-tabs .tab').forEach(t => t.classList.toggle('active', t.dataset.d === d));
  };
  $$('#res-tabs .tab').forEach(t => t.onclick = () => setD(t.dataset.d));
  setD('osworld');
  new Chart($('#chart-keep'), {
    type: 'bar',
    data: { labels: ['Desktop', ['Low-motion', 'games'], ['Medium-motion', 'games'], ['High-motion', 'games']],
      datasets: [
        { label: 'VTR', data: [26.3, 2.7, 5.0, 5.9], backgroundColor: C.vtr },
        { label: 'H.264', data: [64.3, 32, 64, 70], backgroundColor: C.h264 },
        { label: 'Pixel-diff', data: [9.6, 5.9, 16, 26], backgroundColor: C.pdiff },
      ] },
    options: { maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { y: { min: 0, max: 80, title: { display: true, text: 'Visual tokens kept (%)' } } } },
  });
})();

/* ───────────── BibTeX copy ───────────── */
$('#copy-bib').onclick = () => {
  navigator.clipboard.writeText($('#bibtex').textContent);
  $('#copy-bib').textContent = 'Copied'; setTimeout(() => ($('#copy-bib').textContent = 'Copy'), 1500);
};
