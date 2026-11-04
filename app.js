/* Visual Token Routing project page */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const EX = Object.fromEntries(EXAMPLES.map(e => [e.key, e]));
const fmt = n => Math.round(n).toLocaleString('en-US');

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
    desktop: { n: 100, dense: 2040, vtr: 807.9, unit: i => `Step ${i} of 100`, result: '2.5× fewer visual tokens' },
    game: { n: 1800, dense: 220, vtr: 38.2, unit: i => `${Math.floor(i / 30)} of 60 seconds`, result: '5.8× fewer visual tokens' },
  };
  let mode = 'desktop', raf = null, started = false;
  const short = n => n >= 1000 ? `${Math.round(n / 1000)}K` : `${Math.round(n)}`;
  const draw = k => {
    const m = MODES[mode], max = m.n * m.dense;
    $('#race-dense').style.width = `${100 * k * m.dense / max}%`;
    $('#race-vtr').style.width = `${100 * k * m.vtr / max}%`;
    $('#race-dense-total').textContent = `${short(k * m.dense)} tokens`;
    $('#race-vtr-total').textContent = `${short(k * m.vtr)} tokens`;
    $('#race-clock').textContent = m.unit(k);
    $('#race-result').innerHTML = k >= m.n ? `<span><b>${m.result.split(' ')[0]}</b> ${m.result.split(' ').slice(1).join(' ')}</span>` : '';
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
})();

/* ───────────── Router video: plays while on screen ───────────── */
new IntersectionObserver(es => { const v = es[0].target; if (es[0].isIntersecting) v.play(); else v.pause(); }, { threshold: 0.4 }).observe($('#router-video'));

/* ───────────── How it works: step tabs ───────────── */
$$('#step-tabs .step-tab').forEach(b => b.onclick = () => {
  $$('#step-tabs .step-tab').forEach(x => x.classList.toggle('active', x === b));
  $$('.step-panel').forEach(p => p.classList.toggle('active', p.dataset.step === b.dataset.step));
  document.dispatchEvent(new CustomEvent('step', { detail: +b.dataset.step }));
});

/* ───────────── Adaptive weight: validation data of the VTR training run ───────────── */
(function adaptiveWeight() {
  // share of the help from the extra screenshots that the VLM loses, on validation data (target 0.1)
  const REAL = [[50,0.5325],[100,0.4774],[150,0.2792],[200,0.3683],[250,0.2714],[300,0.2759],[350,0.19],[400,0.157],[450,0.0815],[500,0.0741],[550,0.0259],[600,0.2827],[650,0.1597],[700,0.3053],[750,0.3161],[800,0.2924],[850,0.2366],[900,0.1951],[950,0.1518],[1000,0.1054],[1050,0.1327],[1100,0.062],[1150,0.0714],[1200,0.0663],[1250,0.0631],[1300,0.1126],[1350,0.1719],[1400,0.072],[1450,0.05],[1500,-0.0362],[1550,-0.0475],[1600,0.0115],[1650,0.0962],[1700,0.0428],[1750,0.0181],[1800,0.033],[1850,0.0813],[1900,0.1139],[1950,0.1564],[2000,0.0754]];
  let chart = null;
  const make = () => {
    if (chart || !window.Chart) return;
    chart = new Chart($('#chart-rbar'), {
      type: 'line',
      data: { datasets: [
        { label: 'Share of the help lost', data: REAL.map(([x, y]) => ({ x, y: 100 * y })), borderColor: '#0891b2', backgroundColor: '#0891b2', borderWidth: 2.5, pointRadius: 3, tension: 0.25 },
        { label: 'Target, 10%', data: [{ x: 0, y: 10 }, { x: 2000, y: 10 }], borderColor: '#16a34a', borderDash: [6, 5], borderWidth: 2, pointRadius: 0 },
      ] },
      options: { maintainAspectRatio: false, animation: { duration: 1200 },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 14, font: { size: 11 } } } },
        scales: { x: { type: 'linear', min: 0, max: 2000, title: { display: true, text: 'Training step' } },
                  y: { min: -10, max: 60, ticks: { callback: v => `${v}%` }, title: { display: true, text: 'Share of the help lost' } } } },
    });
  };
  document.addEventListener('step', e => { if (e.detail === 2) make(); });
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
