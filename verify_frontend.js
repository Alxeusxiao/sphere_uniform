/* 前端一致性 + 页面行为检查（合并自 verify_js.js / verify_s.js /
   check_page_cover.js / check_page_kcenter.js），共用一套 DOM/canvas 桩。

   跑法： node verify_frontend.js [math|page|all]

   1. math —— 前端数学 vs Python 内核：能量/夹角/覆盖半径/梯度（含各 s）
   2. page —— 页面行为：初始状态、拖动重绘、三种迭代算法、自定义点数、
              保存/加载/收藏夹、覆盖半径专项、k-center 单调性与收敛
*/
const fs = require('fs');
const vm = require('vm');

const FAILS = [];
const chk = (name, cond, extra) => {
  console.log((cond ? 'OK   ' : 'FAIL ') + name + (extra ? '   ' + extra : ''));
  if (!cond) FAILS.push(name);
};

/* ======================================================================
   0. 共用工具：抽取页面里的数学段 / 搭一个 DOM+canvas 桩
   ====================================================================== */
function mathApi() {
  const html = fs.readFileSync('sphere_uniform.html', 'utf8');
  const code = html.match(/<script>\s*"use strict";([\s\S]*?)<\/script>/)[1];
  const stubEl = { getContext: () => CTX, addEventListener() {}, classList: { add() {}, remove() {} },
                   style: {}, value: '', textContent: '',
                   getBoundingClientRect: () => ({ width: 900, height: 600, left: 0, top: 0 }),
                   appendChild() {}, click() {} };
  stubEl.parentElement = stubEl;
  const ctx = {
    document: { getElementById: () => stubEl, querySelector: () => stubEl,
                querySelectorAll: () => [], createElement: () => stubEl, addEventListener() {},
                body: stubEl },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    performance: { now: () => Date.now() },
    requestAnimationFrame: () => 1, setTimeout: () => 1,
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    fetch: async () => { throw new Error('no network'); },
    console,
  };
  vm.createContext(ctx);
  vm.runInContext(code +
    ';this.API={fibonacciPoints,minAngle,rieszEnergy,logEnergy,energyGrad,dminSubgrad,' +
    'voronoiCells,cellArea,cellAreasOf,areaCvOf,coverageRadius,dot,cross,norm3};', ctx);
  return ctx.API;
}

/* rAF 队列 + canvas 绘制计数 */
let rafQueue = [];
let rafRuns = 0;
const draws = { clearRect: 0, arc: 0, fillRect: 0 };
function flushRaf(n = 1) {
  for (let k = 0; k < n; k++) {
    const q = rafQueue; rafQueue = [];
    q.forEach(f => { rafRuns++; try { f(Date.now()); } catch (e) { FAILS.push('raf:' + e.message); } });
  }
}
const totalDraws = () => draws.clearRect + draws.arc + draws.fillRect;

function makeEl(id) {
  const el = {
    id, tagName: 'DIV', value: '', textContent: '', innerHTML: '', checked: false,
    clientWidth: 900, clientHeight: 600, width: 900, height: 600,
    style: {}, dataset: {}, files: [], type: '',
    classList: { add() {}, remove() {}, toggle() {} },
    getContext: () => CTX,
    addEventListener(t, f) { (el._l = el._l || {})[t] = f; },
    removeEventListener() {}, dispatchEvent(ev) { const m = 'on' + ev.type; if (typeof el[m] === 'function') el[m](ev); return true; },
    setPointerCapture() {}, getBoundingClientRect: () => ({ width: 900, height: 600, left: 0, top: 0 }),
    appendChild(c) { (el._kids = el._kids || []).push(c); if (c) c.parentElement = el; return c; },
    removeChild(c) { el._kids = (el._kids || []).filter(x => x !== c); },
    remove() { const p = el.parentElement; if (p && p._kids) p._kids = p._kids.filter(x => x !== el); },
    click() { if (typeof el.onclick === 'function') return el.onclick({ type: 'click' }); },
    querySelectorAll: () => [],
  };
  el.parentElement = { getBoundingClientRect: () => ({ width: 900, height: 600, left: 0, top: 0 }) };
  return el;
}
const CTX = new Proxy({}, {
  get(_, k) {
    if (k === 'createRadialGradient') return () => ({ addColorStop() {} });
    if (k === 'measureText') return () => ({ width: 10 });
    if (k in draws) return () => { draws[k]++; };
    return () => {};
  }, set: () => true,
});

const IDS = ['globe', 'chart', 'nRange', 'nOut', 'objSel', 'srcSel', 'algoSel', 'spdRange', 'runBtn',
  'resetBtn', 'gridSel', 'colorSel', 'capBtn', 'dataNote', 'iterNote', 'mE', 'mD', 'mR', 'cmpNote',
  'nCustom', 'nApply', 'nameIn', 'saveFileBtn', 'loadFileBtn', 'favSaveBtn', 'favClearBtn', 'fileIn',
  'ioNote', 'favList', 'aside', 'sInput', 'sPreset', 'stepBtn', 'stepNBtn', 'stepCount',
  'mCv'];

/** 起一个可运行的页面沙箱（注入 sphere_data.js 的数据） */
function pageSandbox() {
  const els = {};
  for (const id of IDS) els[id] = makeEl(id);
  els.nRange.value = '20'; els.objSel.value = 'energy'; els.srcSel.value = 'pre';
  els.algoSel.value = 'auto'; els.spdRange.value = '12'; els.gridSel.value = 'fib';
  els.colorSel.value = 'nbr'; els.sInput.value = '1'; els.sPreset.value = '1'; els.stepCount.value = '1';

  const store = {}, winListeners = {}, downloads = [];
  const sandbox = {
    console, performance: { now: () => Date.now() },
    requestAnimationFrame: (f) => { rafQueue.push(f); return rafQueue.length; },
    setTimeout: (f) => { try { f(); } catch (e) { FAILS.push('timeout:' + e.message); } return 1; },
    clearTimeout() {},
    fetch: async () => { throw new Error('no network'); },
    localStorage: { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: k => { delete store[k]; } },
    confirm: () => true,
    Blob: class { constructor(p) { this.parts = p; } },
    URL: { createObjectURL: (b) => { downloads.push(b); return 'blob:x'; }, revokeObjectURL() {} },
    FileReader: class { readAsText() {} },
    document: {
      body: makeEl('body'),
      getElementById: id => els[id] || makeEl(id),
      querySelector: sel => (sel === 'aside' ? els.aside : makeEl(sel)),
      querySelectorAll: () => [],
      createElement: t => makeEl('c-' + t),
      addEventListener() {},
    },
    window: { addEventListener(t, f) { winListeners[t] = f; }, devicePixelRatio: 1, SPHERE_DATA: null, confirm: () => true },
    Math, Date, JSON, Number, Array, Object, String, Boolean, Error, isNaN, parseFloat, parseInt,
  };
  sandbox.window.document = sandbox.document;
  sandbox.globalThis = sandbox;
  // 用生成好的数据文件当注入源（等价于 file:// 打开时的回退）
  const raw = fs.readFileSync('sphere_data.js', 'utf8');
  sandbox.window.SPHERE_DATA = JSON.parse(raw.slice(raw.indexOf('=') + 1, raw.lastIndexOf(';')));

  const html = fs.readFileSync('sphere_uniform.html', 'utf8');
  const code = html.match(/<script>\s*"use strict";([\s\S]*?)<\/script>/)[1] + `
;globalThis.__X = {S, project, draw, drawChart, computeMetrics, stepAlgo, stepN, stepOnce,
  fibonacciPoints, randomPoints, voronoiCells, coverageRadius, gridCoverRadius, kcenterScanFast,
  kcenterEscapeCheap, effectiveAlgo, minAngle, rieszEnergy, logEnergy, energyGrad, dminSubgrad,
  areaCvOf, cellAreasOf, cellArea, areaLloydStep, areaImprove, cellCentroid,
  updateMetrics, gensphere, frame, applyS, applyCustomN, syncNControls, setIoNote,
  exportConfig, importConfig, loadConfigText, sanitizePoints, parseCSV,
  saveToGallery, loadFromGallery, deleteFromGallery, clearGallery, readFavs,
  bootDone: boot().then(()=>{ globalThis.__BOOTED = true; })};`;
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: 'page.js' });
  return { sandbox, els, downloads, winListeners };
}

/* ======================================================================
   1. 数学一致性：前端 vs Python 内核
   ====================================================================== */
function checkMath() {
  const A = mathApi();
  const refPath = 'tests/reference.json';
  if (!fs.existsSync(refPath)) {
    console.log('（缺少参考值，调用 tools.py reference 生成…）');
    require('child_process').execFileSync(process.env.DSH_PYTHON || 'python', ['tools.py', 'reference'],
      { stdio: 'inherit', cwd: process.cwd() });
  }
  const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));
  const py = ref.configs;

  console.log('\n--- 正多面体（前端独立算） ---');
  const phi = (1 + Math.sqrt(5)) / 2;
  const ico = [];
  for (const s1 of [1, -1]) for (const s2 of [1, -1])
    ico.push([0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]);
  const nz = P => P.map(p => { const n = Math.hypot(...p); return p.map(x => x / n); });
  const tetra = nz([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]);
  const octa = nz([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]);
  const icoN = nz(ico);
  chk('四面体 d_min=109.4712°', Math.abs(A.minAngle(tetra) * 180 / Math.PI - 109.4712206) < 1e-6);
  chk('四面体 R_cov=70.5288°', Math.abs(A.coverageRadius(tetra) * 180 / Math.PI - 70.5287794) < 1e-5);
  chk('八面体 R_cov=54.7356°', Math.abs(A.coverageRadius(octa) * 180 / Math.PI - 54.7356103) < 1e-5);
  chk('二十面体 d_min=63.4349°', Math.abs(A.minAngle(icoN) * 180 / Math.PI - 63.4349488) < 1e-6);
  chk('二十面体 R_cov=37.3774°', Math.abs(A.coverageRadius(icoN) * 180 / Math.PI - 37.3773681) < 1e-5);

  console.log('\n--- 与 Python 数据逐项比对 ---');
  let worstR = 0, worstD = 0, worstE = 0;
  for (const n of [6, 12, 20, 32, 60, 100]) {
    const c = py[String(n)];
    for (const obj of ['energy', 'tammes', 'cover']) {
      const P = c[obj].pts;
      const R = A.coverageRadius(P) * 180 / Math.PI;
      const d = A.minAngle(P) * 180 / Math.PI;
      const E = A.rieszEnergy(P, 1);
      const pyR = c[obj].radiusDeg, pyE = c[obj].energy, pyD = c[obj].minDeg;
      worstR = Math.max(worstR, Math.abs(R - pyR));
      worstD = Math.max(worstD, Math.abs(d - pyD));
      worstE = Math.max(worstE, Math.abs(E - pyE) / Math.max(1, Math.abs(pyE)));
    }
  }
  chk('覆盖半径一致', worstR < 1e-4, `最大差 ${worstR.toExponential(2)}°`);
  chk('最小夹角一致', worstD < 1e-4, `最大差 ${worstD.toExponential(2)}°`);
  chk('能量一致', worstE < 1e-6, `最大相对差 ${worstE.toExponential(2)}`);

  console.log('\n--- 各 s 的 Riesz 能量/梯度（含 s=0 对数能量）---');
  for (const key of Object.keys(ref.riesz_cases)) {
    const c = ref.riesz_cases[key];
    const s = c.s;
    if (!c) continue;
    const P2 = c.points;
    const eOk = Math.abs(A.rieszEnergy(P2, s) - c.energy) < 1e-9 * Math.max(1, Math.abs(c.energy));
    const G = A.energyGrad(P2, s);
    let mx = 0;
    for (let i = 0; i < P2.length; i++) for (let k = 0; k < 3; k++)
      mx = Math.max(mx, Math.abs(G[i * 3 + k] - c.grad[i][k]) / Math.max(1e-12, Math.abs(c.grad[i][k])));
    chk(`s=${s} 能量与梯度`, eOk && mx < 1e-9, `梯度最大相对差 ${mx.toExponential(2)}`);
  }

  console.log('\n--- Voronoi 胞腔面积（前端 vs Python 参考值）---');
  {
    let worstA = 0, worstSum = 0, worstCv = 0;
    for (const n of [6, 12, 20, 50, 100]) {
      const P = py[String(n)].cover.pts;
      const cells = A.voronoiCells(P);
      const areas = cells.map(V => A.cellArea(V));
      const sum = areas.reduce((x, y) => x + y, 0);
      worstSum = Math.max(worstSum, Math.abs(sum - 4 * Math.PI));
      const refA = ref.cell_area_ref[String(n)];
      if (refA) for (let i = 0; i < n; i++) worstA = Math.max(worstA, Math.abs(areas[i] - refA[i]));
      const cv = A.areaCvOf(P);
      if (ref.cell_cv_ref && ref.cell_cv_ref[String(n)] !== undefined)
        worstCv = Math.max(worstCv, Math.abs(cv - ref.cell_cv_ref[String(n)]));
    }
    chk('各胞腔面积与 Python 一致', worstA < 1e-9, `最大差 ${worstA.toExponential(2)}`);
    chk('胞腔面积总和 == 4π', worstSum < 1e-9, `最大偏差 ${worstSum.toExponential(2)}`);
    chk('面积离散度 cv 与 Python 一致', worstCv < 1e-9, `最大差 ${worstCv.toExponential(2)}`);
  }

  console.log('\n--- Voronoi 胞腔顶点数 ---');
  for (const n of [6, 12, 20, 32, 60, 100]) {
    const cells = A.voronoiCells(py[String(n)].cover.pts);
    const bad = cells.filter(c => c.length < 3).length;
    chk(`n=${n} 胞腔数=${n} 且非退化`, cells.length === n && bad === 0,
      `平均顶点数 ${(cells.reduce((s, c) => s + c.length, 0) / n).toFixed(2)}`);
  }
}

/* ======================================================================
   2. 页面行为
   ====================================================================== */
async function checkPage() {
  const { sandbox, els, downloads, winListeners } = pageSandbox();
  const X = sandbox.__X;
  await X.bootDone;
  flushRaf(3);
  const S = X.S, deg = r => r * 180 / Math.PI;

  console.log('\n--- 初始状态 ---');
  chk('boot 完成且点集长度=N', S.P.length === S.n, `N=${S.n}`);
  chk('指标已算出', Number.isFinite(S.metrics.e) && S.metrics.d > 0);
  const pr = X.project(S.P);
  chk('投影无 NaN 且尺度合理', pr.every(q => q.every(Number.isFinite) &&
    Math.abs(q[0] - 450) < 400 && Math.abs(q[1] - 300) < 400));

  console.log('\n--- 拖动旋转 / 滚轮缩放实时重绘 ---');
  flushRaf(3);
  let before = totalDraws();
  const dn = els.globe._l.pointerdown, mv = els.globe._l.pointermove, up = els.globe._l.pointerup;
  chk('已注册 pointer 事件', !!dn && !!mv && !!up);
  const p0 = X.project([[1, 0, 0]])[0];
  dn({ clientX: 400, clientY: 300, pointerId: 1 });
  for (let k = 0; k < 3; k++) mv({ clientX: 400 + 40 * (k + 1), clientY: 300 + 12 * (k + 1) });
  chk('拖动排入重绘请求', rafQueue.length > 0);
  flushRaf(3);
  chk('拖动后确实重绘', totalDraws() > before);
  chk('拖动改变了视角', Math.abs(X.project([[1, 0, 0]])[0][0] - p0[0]) > 1);
  up({ clientX: 520, clientY: 336 });
  before = totalDraws();
  els.globe._l.wheel({ deltaY: -240, preventDefault() {} });
  flushRaf(3);
  chk('滚轮缩放后重绘', totalDraws() > before);
  before = totalDraws();
  if (winListeners.resize) { sandbox.window.devicePixelRatio = 2; winListeners.resize(); }
  flushRaf(3);
  chk('窗口缩放后重绘', totalDraws() > before);
  flushRaf(3);
  const r0 = rafRuns;
  X.frame(); X.frame(); X.frame();
  const queued = rafQueue.length;
  flushRaf(3);
  chk('同帧多次请求合并成一帧', queued === 1 && rafRuns - r0 === 1);

  console.log('\n--- 三种迭代算法（小 N，精确 R_cov）---');
  const FIB = n => X.fibonacciPoints(n);
  function runAlgo(algo, n, steps) {
    S.algo = algo; S.n = n; S.s = 1; X.applyS('1');
    S.P = FIB(n);
    const m0 = X.computeMetrics(S.P);
    for (let k = 0; k < steps; k++) X.stepAlgo();
    const m1 = X.computeMetrics(S.P);
    const unit = S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-9);
    console.log(`      ${algo} N=${n}: E ${m0.e.toFixed(3)}->${m1.e.toFixed(3)} | ` +
      `dmin ${deg(m0.d).toFixed(3)}°->${deg(m1.d).toFixed(3)}° | R ${deg(m0.r).toFixed(3)}°->${deg(m1.r).toFixed(3)}°`);
    return { m0, m1, unit };
  }
  const rf = runAlgo('fw', 20, 6);
  chk('Frank–Wolfe 降低能量', rf.m1.e < rf.m0.e - 1e-9);
  chk('Frank–Wolfe 保持单位长度', rf.unit);
  const rt = runAlgo('tammes', 20, 200);
  chk('Tammes 不降低最小间距', rt.m1.d >= rt.m0.d - 1e-12);
  chk('Tammes 保持单位长度', rt.unit);
  const rk = runAlgo('kcenter', 20, 120);
  chk('k-center 降低覆盖半径', rk.m1.r < rk.m0.r - 1e-9);
  chk('k-center 保持单位长度', rk.unit);
  const rl = runAlgo('lloyd', 20, 12);
  chk('Lloyd 保持单位长度', rl.unit);

  console.log('\n--- k-center 单调性（逐步检查，不得上升）---');
  for (const n of [6, 12, 20]) {
    S.algo = 'kcenter'; S.n = n; S.P = FIB(n); S.kBest = null;
    let prev = X.coverageRadius(S.P), bad = 0;
    for (let k = 0; k < 120; k++) {
      X.stepAlgo();
      const now = X.coverageRadius(S.P);
      if (now > prev + 1e-9) bad++;
      prev = now;
    }
    chk(`N=${n} k-center 单调且改善`, bad === 0 && prev < X.coverageRadius(FIB(n)) - 1e-9,
      `${deg(X.coverageRadius(FIB(n))).toFixed(4)}° -> ${deg(prev).toFixed(4)}° 上升${bad}次`);
  }

  console.log('\n--- 覆盖半径专项：切目标/面板显示 ---');
  for (const n of [8, 12, 20, 32]) {
    for (const obj of ['energy', 'cover']) {
      S.n = n; S.obj = obj; S.src = 'pre'; S.s = 1; X.applyS('1');
      X.gensphere();
      const real = X.coverageRadius(S.P) * 180 / Math.PI;
      const panel = parseFloat(String(els.mR.textContent));
      chk(`N=${n} 选 ${obj}：面板与实际一致`, Math.abs(panel - real) < 1e-3,
        `面板=${panel} 实际=${real.toFixed(4)}`);
    }
  }
  chk('目标=覆盖 时自动选 k-center', (() => { S.algo = 'auto'; S.obj = 'cover'; return X.effectiveAlgo() === 'kcenter'; })());
  chk('目标=间距 时自动选 tammes', (() => { S.obj = 'tammes'; return X.effectiveAlgo() === 'tammes'; })());
  chk('目标=能量 时自动选 fw', (() => { S.obj = 'energy'; return X.effectiveAlgo() === 'fw'; })());

  console.log('\n--- 面积均匀目标（第四个目标）---');
  {
    chk('目标下拉含"面积最均匀"', /value="area"/.test(htmlSource()));
    S.algo = 'auto'; S.obj = 'area';
    chk('目标=面积 时自动选 arealloyd', X.effectiveAlgo() === 'arealloyd', X.effectiveAlgo());

    // cv 单调性：逐步检查
    for (const n of [12, 20, 50]) {
      S.algo = 'arealloyd'; S.n = n; S.P = X.fibonacciPoints(n);
      const cv0 = X.areaCvOf(S.P);
      let prev = cv0, up = 0;
      const steps = n <= 20 ? 120 : 40;
      for (let k = 0; k < steps; k++) {
        X.stepAlgo();
        const now = X.areaCvOf(S.P);
        if (now > prev + 1e-12) up++;
        prev = now;
      }
      const unit = S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-9);
      const gain = 100 * (cv0 - prev) / cv0;
      console.log(`      N=${n} cv ${cv0.toFixed(5)} -> ${prev.toFixed(5)} （上升 ${up} 次，改善 ${gain.toFixed(2)}%）`);
      chk(`N=${n} 面积迭代单调不增且保持单位长度`, up === 0 && unit, `cv=${prev.toFixed(5)}`);
      // 斐波那契格子本身可能就在局部极小附近（n=12 即如此），所以允许"已很均匀"这一支
      chk(`N=${n} 面积迭代有改善或起点已很均匀`, gain > 0.5 || cv0 < 0.02,
        `改善 ${gain.toFixed(2)}%，起 cv=${cv0.toFixed(5)}`);
      chk(`N=${n} 面积迭代不比预计算解差太多`, (() => {
        const d = dataCv(n);
        // 预计算的 cv 可能极小（如 n=12 的正二十面体约 1.7e-8，打印成 0.00000），
        // 所以除了相对容差还要给一个绝对容差，避免把浮点噪声判成失败
        return d === null || prev <= Math.max(d * 2, 1e-4);   // cv<1e-4 已可视为等面积
      })(), `页面=${prev.toFixed(5)} 预计算=${(dataCv(n) || 0).toFixed(5)}`);
    }

    // 面板 cv 与实际一致
    for (const n of [12, 32]) {
      S.n = n; S.obj = 'area'; S.src = 'pre'; S.s = 1;
      if (dataHas(n, 'area')) X.gensphere();
      const real = X.areaCvOf(S.P);
      const panel = parseFloat(String(els.mCv.textContent));
      chk(`N=${n} 面板 cv 与实际一致`, Math.abs(panel - real) < 1e-4,
        `面板=${panel} 实际=${real.toFixed(5)}`);
    }
  }

  console.log('\n--- 自定义点数 ---');
  els.nCustom.value = '137'; X.applyCustomN();
  chk('自定义 N=137 生效', S.P.length === 137 && String(els.nOut.textContent) === '137');
  els.nCustom.value = '900'; X.applyCustomN();
  chk('自定义 N=900 生效', S.P.length === 900);
  for (const algo of ['fw', 'tammes', 'kcenter']) {
    S.algo = algo;
    const t0 = Date.now(); X.stepAlgo();
    chk(`N=900 时 ${algo} 可运行且保持单位长度`,
      S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-9), `${Date.now() - t0} ms`);
  }
  els.nCustom.value = '5000'; X.applyCustomN();
  chk('N=5000 可生成', S.P.length === 5000);
  els.nCustom.value = 'abc'; X.applyCustomN();
  chk('非法输入被拒绝且不破坏状态', S.P.length === 5000);
  els.nCustom.value = '256'; X.applyCustomN();

  console.log('\n--- 保存 / 加载 / 收藏夹 ---');
  S.n = 256; S.s = 1; X.applyS('1'); S.P = FIB(256); S.iters = 0;
  els.nameIn.value = '我的测试球';
  const exp = X.exportConfig();
  chk('导出对象完整', exp.n === 256 && exp.points.length === 256 &&
    Math.abs(Math.hypot(...exp.points[0]) - 1) < 1e-6 &&
    Number.isFinite(exp.metrics.energy) && Number(exp.riesz) === 1);
  const nDl = downloads.length;
  els.saveFileBtn.click();
  chk('保存到文件产生下载', downloads.length === nDl + 1);
  const text = String(downloads[downloads.length - 1].parts[0]);
  const r1 = X.loadConfigText(text, 'roundtrip.json');
  chk('导出文件能原样加载回来', r1.ok && r1.n === 256 && els.nameIn.value === '我的测试球');
  chk('CSV 也能加载', X.loadConfigText('1,0,0\n0,1,0\n0,0,1\n').ok);
  chk('裸 {points} 会被归一化', X.importConfig({ name: 'raw', points: [[2, 0, 0], [0, 3, 0], [0, 0, -4]] }).ok &&
    S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-12));
  chk('非法点集被拒绝', !X.importConfig({ name: 'bad', points: [[1, 2], ['a', 'b', 'c']] }).ok);
  chk('sanitizePoints 过滤坏点', (X.sanitizePoints([[1, 0, 0], [0, 0, 0], ['x', 0, 0], [0, 1, 0]]) || []).length === 2);
  X.applyS('3');
  const exp3 = X.exportConfig();
  X.loadConfigText(JSON.stringify(exp3), 's3.json');
  chk('加载后 s 被恢复', Math.abs(S.s - 3) < 1e-12 && String(els.sInput.value) === '3');
  X.applyS('1');

  X.importConfig({ name: 'A 球', points: FIB(12) }); X.saveToGallery('A 球');
  X.importConfig({ name: 'B 球', points: FIB(20) }); X.saveToGallery('B 球');
  chk('收藏夹存入两条', X.readFavs().length === 2);
  X.loadFromGallery(0);
  chk('从收藏夹载入', S.P.length === 12 && els.nameIn.value === 'A 球');
  X.saveToGallery('A 球');
  chk('同名保存是覆盖', X.readFavs().length === 2);
  X.deleteFromGallery(1);
  chk('删除一条', X.readFavs().length === 1);
  X.clearGallery();
  chk('清空收藏夹', X.readFavs().length === 0);

  console.log('\n--- 单步 / 推进 n 步 ---');
  S.algo = 'fw'; S.n = 40; S.P = FIB(40); S.iters = 0;
  els.stepBtn.click();
  chk('"单步"只推进一轮', S.iters === 1);
  els.stepCount.value = '5';
  const it2 = S.iters;
  els.stepNBtn.click();
  chk('"推进 n 步"按步数推进', S.iters === it2 + 5);
  els.stepCount.value = '999999'; els.stepNBtn.click();
  chk('步数夹到上限 10000', Number(els.stepCount.value) === 10000);
  els.runBtn.click();
  chk('开始迭代后 running=true', S.running === true);
  els.stepCount.value = '1'; els.stepNBtn.click();
  chk('手动推进会先停止自动循环', S.running === false);
  flushRaf(2);

  console.log('\n--- 渲染分支 ---');
  try {
    S.n = 256; X.gensphere();
    X.draw(); X.drawChart();
    S.caps = true; S.colorMode = 'cell'; S.grid = 'lat'; X.draw();
    S.colorMode = 'nbr'; S.grid = 'none'; X.draw(); S.caps = false; X.draw();
    els.nCustom.value = '1500'; X.applyCustomN();
    S.colorMode = 'cell'; X.draw(); S.colorMode = 'nbr'; X.draw();
    chk('全部渲染分支正常（含 N=1500）', true);
  } catch (e) {
    chk('全部渲染分支正常', false, e.message);
  }
}

function htmlSource() { return fs.readFileSync('sphere_uniform.html', 'utf8'); }
function dataCv(n) {
  const raw = fs.readFileSync('sphere_data.js', 'utf8');
  const d = JSON.parse(raw.slice(raw.indexOf('=') + 1, raw.lastIndexOf(';')));
  const a = d[String(n)] && d[String(n)].configs.area;
  return a ? a.cv : null;
}

function dataHas(n, key) {
  const raw = fs.readFileSync('sphere_data.js', 'utf8');
  const d = JSON.parse(raw.slice(raw.indexOf('=') + 1, raw.lastIndexOf(';')));
  return !!(d[String(n)] && d[String(n)].configs[key]);
}

/* ======================================================================
   入口
   ====================================================================== */
(async () => {
  const mode = (process.argv[2] || 'all').toLowerCase();
  if (mode === 'math' || mode === 'all') checkMath();
  if (mode === 'page' || mode === 'all') await checkPage();
  console.log('\n' + (FAILS.length ? `有 ${FAILS.length} 项失败 ✗` : '前端检查全部通过 ✓'));
  process.exit(FAILS.length ? 1 : 0);
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
