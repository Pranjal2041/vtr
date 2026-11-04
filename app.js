/* Visual Token Routing project page */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const EX = Object.fromEntries(EXAMPLES.map(e => [e.key, e]));
const fmt = n => Math.round(n).toLocaleString('en-US');

/* ───────────── What the model sees: before / after slider ───────────── */
(function sees() {
  const order = ['dota', 'minecraft', 'candy', 'gta', 'gpredict', 'psychopy', 'librecad'];
  const tabs = $('#sees-tabs'), box = $('#compare'), top = $('#cmp-top'), handle = $('#cmp-handle');
  let pos = 50;
  const setPos = p => {
    pos = Math.max(0, Math.min(100, p));
    top.style.clipPath = `inset(0 0 0 ${pos}%)`;
    handle.style.left = pos + '%';
  };
  const show = key => {
    const e = EX[key];
    $('#cmp-base').src = `assets/examples/${key}_prev.jpg`;
    $('#cmp-kept').src = `assets/examples/${key}_kept.jpg`;
    $('#cmp-count').innerHTML = `<b>${fmt(e.kept)}</b> of ${fmt(e.grid[0] * e.grid[1])} tokens kept`;
    $('#cmp-caption').textContent = `${e.name}. ${e.setup} ${e.note}`;
    $$('.tab', tabs).forEach(t => t.classList.toggle('active', t.dataset.key === key));
  };
  order.forEach(k => {
    const b = document.createElement('button');
    b.className = 'tab'; b.dataset.key = k; b.textContent = EX[k].name;
    b.onclick = () => show(k);
    tabs.appendChild(b);
  });
  let drag = false;
  const move = ev => { const r = box.getBoundingClientRect(); setPos((ev.clientX - r.left) / r.width * 100); };
  box.addEventListener('pointerdown', ev => { drag = true; box.setPointerCapture(ev.pointerId); move(ev); });
  box.addEventListener('pointermove', ev => { if (drag) move(ev); });
  box.addEventListener('pointerup', () => { drag = false; });
  show('dota');
  // a short sweep on load shows that the divider can be dragged
  const t0 = performance.now();
  const sweep = now => {
    const u = Math.min(1, (now - t0) / 1800), e = 1 - Math.pow(1 - u, 3);
    setPos(88 - 58 * e);
    if (u < 1 && !drag) requestAnimationFrame(sweep);
  };
  setPos(88);
  setTimeout(() => requestAnimationFrame(sweep), 500);
})();

/* ───────────── Token race ───────────── */
(function race() {
  // Averages from the paper: OSWorld (desktop) and video game click prediction.
  const MODES = {
    desktop: { n: 100, dense: 2040, vtr: 807.9, unit: i => `Step ${i} of 100`,
      result: ['2.5× fewer tokens', '2.4× lower cost', '2.3× faster prefill'] },
    game: { n: 1800, dense: 220, vtr: 38.2, unit: i => `0:${String(Math.floor(i / 30)).padStart(2, '0')} of 1:00 at 30 fps`,
      result: ['5.8× fewer tokens', '3.3× lower cost', '2.9× faster prefill'] },
  };
  const B = 10; // anchor every 10 observations
  const cvD = $('#race-dense'), cvV = $('#race-vtr');
  let mode = 'desktop', raf = null, started = false;
  const vtrAt = (m, i) => (i % B === 0 ? m.dense : (m.vtr * B - m.dense) / (B - 1));
  const fit = cv => { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; return cv.getContext('2d'); };
  const draw = (k) => {
    const m = MODES[mode];
    for (const [cv, fn, col] of [[cvD, () => m.dense, '#9ca3af'], [cvV, i => vtrAt(m, i), '#0891b2']]) {
      const ctx = fit(cv), W = cv.width, H = cv.height, w = W / m.n;
      ctx.fillStyle = '#f0f0f2'; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = col;
      for (let i = 0; i < k; i++) {
        const h = H * fn(i) / m.dense;
        ctx.fillRect(i * w, H - h, Math.max(1, w - (m.n <= 200 ? 1.5 : 0)), h);
      }
    }
    let dsum = m.dense * k, vsum = 0;
    for (let i = 0; i < k; i++) vsum += vtrAt(m, i);
    $('#race-dense-total').firstChild.nodeValue = fmt(dsum);
    $('#race-vtr-total').firstChild.nodeValue = fmt(vsum);
    $('#race-clock').textContent = m.unit(k);
    $('#race-result').innerHTML = k >= m.n ? m.result.map(s => `<span><b>${s.split(' ')[0]}</b> ${s.split(' ').slice(1).join(' ')}</span>`).join('') : '';
  };
  const run = () => {
    cancelAnimationFrame(raf);
    const m = MODES[mode], t0 = performance.now(), dur = 5000;
    const tick = now => {
      const u = Math.min(1, (now - t0) / dur);
      draw(Math.round(u * m.n));
      if (u < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  };
  $$('#race-tabs .tab').forEach(b => b.onclick = () => {
    mode = b.dataset.mode;
    $$('#race-tabs .tab').forEach(t => t.classList.toggle('active', t === b));
    run();
  });
  $('#race-replay').onclick = run;
  draw(0);
  new IntersectionObserver((es, ob) => {
    if (es[0].isIntersecting && !started) { started = true; run(); ob.disconnect(); }
  }, { threshold: 0.4 }).observe($('.race'));
  addEventListener('resize', () => draw(MODES[mode].n));
})();

/* ───────────── Playground: a toy desktop ───────────── */
(function playground() {
  const W = 576, H = 336, T = 24, GW = W / T, GH = H / T;
  const NAME = 'player_voice_full.mp3';
  const FIELD = { c0: 15, c1: 22, r0: 4, r1: 5 }; // tokens of the file-name field
  const on = { typing: true, video: true, clock: false };
  const st = { t: 0, k: 6, sec: 7 };
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const octx = off.getContext('2d', { willReadFrequently: true });
  let prev = null;

  function scene(ctx) {
    // desktop
    const g = ctx.createLinearGradient(0, 0, W, H); g.addColorStop(0, '#334155'); g.addColorStop(1, '#1e293b');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // taskbar and clock
    ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 312, W, 24);
    ctx.fillStyle = '#38bdf8'; ctx.fillRect(8, 317, 14, 14);
    ctx.fillStyle = '#e2e8f0'; ctx.font = '12px "JetBrains Mono", monospace';
    const s = st.sec % 60, m = 42 + Math.floor(st.sec / 60);
    ctx.fillText(`10:${m}:${String(s).padStart(2, '0')}`, 500, 328);
    // media player window
    ctx.fillStyle = '#e5e7eb'; ctx.fillRect(12, 12, 316, 236);
    ctx.fillStyle = '#374151'; ctx.font = '600 12px Inter, sans-serif'; ctx.fillText('Media Player', 22, 27);
    const vx = 16, vy = 36, vw = 308, vh = 184;
    const bars = ['#f8fafc', '#facc15', '#22d3ee', '#22c55e', '#d946ef', '#ef4444', '#2563eb'];
    bars.forEach((c, i) => { ctx.fillStyle = c; ctx.fillRect(vx + i * vw / 7, vy, vw / 7 + 1, vh); });
    const ph = st.v || 0;
    ctx.save(); ctx.beginPath(); ctx.rect(vx, vy, vw, vh); ctx.clip();
    ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 10;
    ctx.beginPath(); ctx.moveTo(vx - 60 + (ph * 37) % 420, vy + vh); ctx.lineTo(vx + 60 + (ph * 37) % 420, vy); ctx.stroke();
    ctx.fillStyle = '#111827';
    ctx.beginPath(); ctx.arc(vx + 40 + (ph * 53) % 240, vy + 50 + 40 * Math.sin(ph * 0.9), 16, 0, 7); ctx.fill();
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
    ctx.fillStyle = '#ffffff'; ctx.fillRect(360, 112, 176, 26);
    ctx.strokeStyle = '#0891b2'; ctx.strokeRect(360.5, 112.5, 175, 25);
    ctx.fillStyle = '#0f172a'; ctx.font = '11px "JetBrains Mono", monospace';
    const txt = NAME.slice(0, st.k); ctx.fillText(txt, 365, 129);
    const cx = 365 + ctx.measureText(txt).width + 1; ctx.fillRect(cx, 117, 1, 16);
    ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = '#475569'; ctx.fillText('Save as type:', 360, 160);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(360, 166, 176, 22); ctx.strokeStyle = '#cbd5e1'; ctx.strokeRect(360.5, 166.5, 175, 21);
    ctx.fillStyle = '#334155'; ctx.fillText('Audio (*.mp3)', 366, 181);
    ctx.fillStyle = '#0891b2'; ctx.fillRect(392, 200, 66, 22); ctx.fillStyle = '#e2e8f0'; ctx.fillRect(468, 200, 66, 22);
    ctx.fillStyle = '#fff'; ctx.font = '600 11px Inter, sans-serif'; ctx.fillText('Save', 412, 215);
    ctx.fillStyle = '#334155'; ctx.fillText('Cancel', 482, 215);
  }

  function step() {
    if (on.typing) st.k = st.k >= NAME.length ? 0 : st.k + 1;
    if (on.video) st.v = (st.v || 0) + 1;
    if (on.clock) st.sec += 1;
    scene(octx);
    const cur = octx.getImageData(0, 0, W, H).data;
    const diff = [], vtr = [];
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
      let ch = false;
      if (prev) for (let y = r * T; y < (r + 1) * T && !ch; y++) for (let x = c * T; x < (c + 1) * T; x++) {
        const i = (y * W + x) * 4;
        if (Math.abs(cur[i] - prev[i]) + Math.abs(cur[i + 1] - prev[i + 1]) + Math.abs(cur[i + 2] - prev[i + 2]) > 24) { ch = true; break; }
      }
      diff.push(ch);
      vtr.push(ch && c >= FIELD.c0 && c <= FIELD.c1 && r >= FIELD.r0 && r <= FIELD.r1);
    }
    prev = cur;
    paint($('#cv-diff'), diff); paint($('#cv-vtr'), vtr);
    const n = GW * GH;
    $('#cnt-diff').innerHTML = `<b>${diff.filter(Boolean).length}</b> of ${n} tokens kept`;
    $('#cnt-vtr').innerHTML = `<b>${vtr.filter(Boolean).length}</b> of ${n} tokens kept`;
  }

  function paint(cv, mask) {
    const ctx = cv.getContext('2d');
    ctx.drawImage(off, 0, 0);
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
      if (!mask[r * GW + c]) { ctx.fillStyle = 'rgba(2,6,23,0.74)'; ctx.fillRect(c * T, r * T, T, T); }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1;
    for (let c = 1; c < GW; c++) { ctx.beginPath(); ctx.moveTo(c * T + 0.5, 0); ctx.lineTo(c * T + 0.5, H); ctx.stroke(); }
    for (let r = 1; r < GH; r++) { ctx.beginPath(); ctx.moveTo(0, r * T + 0.5); ctx.lineTo(W, r * T + 0.5); ctx.stroke(); }
    ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2.5;
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) if (mask[r * GW + c]) ctx.strokeRect(c * T + 1.5, r * T + 1.5, T - 3, T - 3);
  }

  $$('#switches .switch').forEach(s => s.onclick = () => {
    on[s.dataset.key] = !on[s.dataset.key];
    s.classList.toggle('on', on[s.dataset.key]);
  });
  scene(octx); prev = octx.getImageData(0, 0, W, H).data;
  step();
  setInterval(step, 850);
})();

/* ───────────── Be the router ───────────── */
(function quiz() {
  const order = ['candy', 'dota', 'minecraft', 'gta', 'candyfall'];
  const box = $('#quiz-box'), grid = $('#quiz-grid'), tabs = $('#quiz-tabs');
  let cur = null, picks = new Set();
  const update = () => { $('#quiz-picked').textContent = picks.size ? `${picks.size} region${picks.size > 1 ? 's' : ''} selected` : 'Click the frame to select regions.'; };
  const load = key => {
    cur = EX[key]; picks = new Set();
    box.classList.remove('revealed');
    $('#quiz-img').src = `assets/examples/${key}_prev.jpg`;
    $('#quiz-name').textContent = cur.name;
    $('#quiz-setup').textContent = `Next frame: ${cur.setup.charAt(0).toLowerCase()}${cur.setup.slice(1)}`;
    const [gw, gh] = cur.grid;
    grid.style.gridTemplateColumns = `repeat(${gw}, 1fr)`; grid.style.gridTemplateRows = `repeat(${gh}, 1fr)`;
    grid.innerHTML = '';
    for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) {
      const d = document.createElement('div'); d.className = 'quiz-cell'; d.dataset.id = r * gw + c;
      d.onclick = () => {
        if (box.classList.contains('revealed')) return;
        const id = +d.dataset.id;
        picks.has(id) ? picks.delete(id) : picks.add(id);
        d.classList.toggle('pick'); update();
      };
      grid.appendChild(d);
    }
    $$('.tab', tabs).forEach(t => t.classList.toggle('active', t.dataset.key === key));
    update();
  };
  order.forEach((k, i) => {
    const b = document.createElement('button');
    b.className = 'tab'; b.dataset.key = k; b.textContent = `${i + 1}. ${EX[k].name}`;
    b.onclick = () => load(k);
    tabs.appendChild(b);
  });
  $('#quiz-reveal').onclick = () => {
    const [gw] = cur.grid; let both = 0;
    $('#quiz-img').src = `assets/examples/${cur.key}_kept.jpg`;
    $$('.quiz-cell', grid).forEach(d => {
      const id = +d.dataset.id, r = Math.floor(id / gw), c = id % gw;
      const v = cur.mask[r][c] === '1';
      d.classList.toggle('vtr', v);
      if (v && picks.has(id)) both++;
    });
    $('#qs-you').textContent = picks.size; $('#qs-vtr').textContent = cur.kept; $('#qs-both').textContent = both;
    $('#quiz-note').textContent = cur.note;
    box.classList.add('revealed');
  };
  $('#quiz-clear').onclick = () => load(cur.key);
  $('#quiz-next').onclick = () => load(order[(order.indexOf(cur.key) + 1) % order.length]);
  load('candy');
})();

/* ───────────── Results charts ───────────── */
(function charts() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.color = '#5a5a63';
  const COL = { dense: '#9ca3af', vtr: '#0891b2', h264: '#d97706', pdiff: '#7c3aed', frozen: '#dc2626' };
  const NAMES = { dense: 'Dense, lower resolution', vtr: 'VTR', h264: 'H.264', pdiff: 'Pixel-diff', frozen: 'Frozen router' };
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
    label: NAMES[k], data: pts, showLine: k === 'dense' || k === 'vtr', borderColor: COL[k], backgroundColor: COL[k],
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
    for (const [c, key] of [[cTok, 'tok'], [cCost, 'cost']]) {
      c.data.datasets = sets(D[d][key]);
      c.options.scales.y.max = D[d].ymax; c.options.scales.y.title.text = D[d].y;
      c.update();
    }
    $$('#res-tabs .tab').forEach(t => t.classList.toggle('active', t.dataset.d === d));
  };
  $$('#res-tabs .tab').forEach(t => t.onclick = () => setD(t.dataset.d));
  setD('osworld');
  new Chart($('#chart-keep'), {
    type: 'bar',
    data: { labels: ['Desktop', ['Low-motion', 'games'], ['Medium-motion', 'games'], ['High-motion', 'games']],
      datasets: [
        { label: 'VTR', data: [26.3, 2.7, 5.0, 5.9], backgroundColor: COL.vtr },
        { label: 'H.264', data: [64.3, 32, 64, 70], backgroundColor: COL.h264 },
        { label: 'Pixel-diff', data: [9.6, 5.9, 16, 26], backgroundColor: COL.pdiff },
      ] },
    options: { maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { y: { min: 0, max: 80, title: { display: true, text: 'Visual tokens kept (%)' } } } },
  });
})();

/* ───────────── BibTeX copy ───────────── */
$('#copy-bib').onclick = () => {
  navigator.clipboard.writeText($('#bibtex').textContent);
  $('#copy-bib').textContent = 'Copied'; setTimeout(() => ($('#copy-bib').textContent = 'Copy'), 1500);
};
