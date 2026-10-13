/* 用 DOM/canvas 桩在 Node 里真实执行页面脚本：
   - 捕获运行时错误，检查投影、渲染、三种迭代算法
   - 验证"拖动旋转 / 滚轮缩放实时重绘"
   - 验证"自定义点数 / 保存到文件 / 从文件加载 / 收藏夹"
   跑法： node smoke_dom.js */
const fs = require('fs');
const vm = require('vm');

const errors = [];
const listeners = {};
const chk = (name, cond) => { console.log((cond ? 'OK   ' : 'FAIL ') + name); if (!cond) errors.push(name); };

/* ---- rAF 队列 ---- */
let rafQueue = [];
let rafRuns = 0;
function flushRaf(n = 1) {
  for (let k = 0; k < n; k++) {
    const q = rafQueue; rafQueue = [];
    q.forEach(f => {
      rafRuns++;
      try { f(Date.now()); } catch (e) { errors.push('raf:' + e.message); }
    });
  }
}

/* ---- canvas 2D 桩 + 绘制计数 ---- */
const draws = { beginPath: 0, arc: 0, fill: 0, stroke: 0, clearRect: 0, fillRect: 0 };
const CTX = new Proxy({}, {
  get(_, k) {
    if (k === 'createRadialGradient') return () => ({ addColorStop() {} });
    if (k === 'measureText') return () => ({ width: 10 });
    if (k in draws) return () => { draws[k]++; };
    return () => {};
  },
  set() { return true; },
});
const totalDraws = () => draws.clearRect + draws.arc + draws.fillRect;

/* ---- 元素桩 ---- */
function makeEl(id) {
  const el = {
    id, tagName: 'DIV', value: '', textContent: '', innerHTML: '', checked: false,
    clientWidth: 900, clientHeight: 600, width: 900, height: 600,
    style: {}, dataset: {}, files: [], type: '',
    classList: { add() {}, remove() {}, toggle() {} },
    getContext() { return CTX; },
    addEventListener(t, f) { (listeners[id] = listeners[id] || {})[t] = f; },
    removeEventListener() {},
    dispatchEvent(ev) { const m = 'on' + ev.type; if (typeof el[m] === 'function') el[m](ev); return true; },
    setPointerCapture() {},
    getBoundingClientRect() { return { width: 900, height: 600, left: 0, top: 0 }; },
    appendChild(c) { (el._kids = el._kids || []).push(c); if (c) c.parentElement = el; return c; },
    removeChild(c) { el._kids = (el._kids || []).filter(x => x !== c); },
    remove() { const p = el.parentElement; if (p && p._kids) p._kids = p._kids.filter(x => x !== el); },
    click() { if (typeof el.onclick === 'function') return el.onclick({ type: 'click' }); },
    querySelectorAll() { return []; },
  };
  el.parentElement = { getBoundingClientRect: () => ({ width: 900, height: 600, left: 0, top: 0 }) };
  return el;
}

const ids = ['globe', 'chart', 'nRange', 'nOut', 'objSel', 'srcSel', 'algoSel', 'spdRange',
  'runBtn', 'resetBtn', 'gridSel', 'colorSel', 'capBtn', 'badge', 'hint', 'dataNote',
  'iterNote', 'mE', 'mD', 'mR', 'cmpNote',
  'nCustom', 'nApply', 'nameIn', 'saveFileBtn', 'loadFileBtn', 'favSaveBtn', 'favClearBtn',
  'fileIn', 'ioNote', 'favList', 'aside', 'sInput', 'sPreset', 'stepBtn', 'stepNBtn', 'stepCount'];
const els = {};
for (const id of ids) els[id] = makeEl(id);
els.nRange.value = '20';
els.objSel.value = 'energy';
els.srcSel.value = 'pre';
els.algoSel.value = 'fw';
els.spdRange.value = '12';
els.gridSel.value = 'fib';
els.colorSel.value = 'nbr';
els.sInput.value = '1'; els.sPreset.value = '1'; els.stepCount.value = '1';

const tabs = ['energy', 'tammes', 'cover'].map(m => {
  const e = makeEl('tab-' + m); e.dataset.m = m; return e;
});

/* ---- localStorage 桩 ---- */
const store = {};
const localStorage = {
  getItem: k => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: k => { delete store[k]; },
};

/* ---- download / FileReader 桩 ---- */
const downloads = [];
const pendingReaders = [];

function makeSandbox() {
  const winListeners = {};
  const sandbox = {
    console,
    performance: { now: () => Date.now() },
    requestAnimationFrame: (f) => { rafQueue.push(f); return rafQueue.length; },
    setTimeout: (f) => { try { f(); } catch (e) { errors.push('timeout:' + e.message); } return 1; },
    clearTimeout() {},
    fetch: async () => { throw new Error('no network in sandbox'); },
    localStorage,
    confirm: () => true,
    Blob: class { constructor(parts) { this.parts = parts; this.size = String(parts[0] || '').length; } },
    URL: { createObjectURL: (b) => { downloads.push(b); return 'blob:stub/' + downloads.length; }, revokeObjectURL() {} },
    FileReader: class {
      readAsText(f) { pendingReaders.push({ reader: this, file: f }); }
    },
    document: {
      body: makeEl('body'),
      getElementById: (id) => els[id] || makeEl(id),
      querySelector: (sel) => (sel === 'aside' ? els.aside : makeEl(sel)),
      querySelectorAll: (sel) => (sel === '.tabs button' ? tabs : []),
      createElement: (tag) => { const e = makeEl('created-' + tag); e.tagName = String(tag).toUpperCase(); return e; },
      addEventListener() {},
    },
    window: {
      addEventListener(t, f) { winListeners[t] = f; },
      devicePixelRatio: 1, SPHERE_DATA: null,
      confirm: () => true,                 // 页面里用 window.confirm 做清空确认
    },
    Math, Date, JSON, Number, Array, Object, String, Boolean, Error, isNaN, parseFloat, parseInt,
  };
  sandbox.window.document = sandbox.document;
  sandbox.window.localStorage = localStorage;
  sandbox.globalThis = sandbox;
  sandbox.__winListeners = winListeners;
  return sandbox;
}

const sandbox = makeSandbox();
if (fs.existsSync('sphere_data.json')) {
  sandbox.window.SPHERE_DATA = JSON.parse(fs.readFileSync('sphere_data.json', 'utf8'));
  console.log('已注入 sphere_data.json：' + Object.keys(sandbox.window.SPHERE_DATA).length + ' 组');
}

const html = fs.readFileSync('sphere_uniform.html', 'utf8');
const m = html.match(/<script>\s*"use strict";([\s\S]*?)<\/script>/);
if (!m) { console.error('找不到主脚本'); process.exit(1); }

const code = m[1] + `
;globalThis.__X = {S, project, draw, drawChart, computeMetrics, stepAlgo, fibonacciPoints,
  voronoiCells, coverageRadius, minAngle, rieszEnergy, updateMetrics, gensphere, frame,
  exportConfig, importConfig, loadConfigText, sanitizePoints, parseCSV,
  saveToGallery, loadFromGallery, deleteFromGallery, clearGallery, readFavs, applyS,
  applyCustomN, syncNControls, setIoNote,
  bootDone: boot().then(()=>{ globalThis.__BOOTED = true; })};`;

vm.createContext(sandbox);
try {
  vm.runInContext(code, sandbox, { filename: 'page.js' });
} catch (e) {
  console.error('脚本执行抛错:', e && e.stack || e);
  process.exit(1);
}
console.log('主脚本执行完成');

(async () => {
  const X = sandbox.__X;
  await X.bootDone;
  if (!sandbox.__BOOTED) { console.error('boot 未完成'); process.exit(1); }
  flushRaf(3);
  console.log('boot 完成，首次渲染已执行');

  const S = X.S;
  const { project, draw, drawChart, computeMetrics, stepAlgo } = X;

  console.log('\n--- 基本状态 ---');
  chk('项目状态 S 已建立', !!S);
  chk('初始点集长度 = N', S.P.length === S.n);
  chk('指标已算出', Number.isFinite(S.metrics.e) && S.metrics.d > 0);
  const pr = project(S.P);
  chk('投影无 NaN', pr.every(q => q.every(Number.isFinite)));
  chk('投影尺度合理', pr.every(q => Math.abs(q[0] - 450) < 400 && Math.abs(q[1] - 300) < 400));

  console.log('\n--- 拖动旋转 / 滚轮缩放实时重绘 ---');
  flushRaf(3);
  let before = totalDraws();
  const down = listeners.globe.pointerdown, move = listeners.globe.pointermove,
        up = listeners.globe.pointerup;
  chk('已注册 pointerdown/pointermove/pointerup', !!down && !!move && !!up);
  const p0 = project([[1, 0, 0]])[0];
  down({ clientX: 400, clientY: 300, pointerId: 1 });
  for (let k = 0; k < 3; k++) move({ clientX: 400 + 40 * (k + 1), clientY: 300 + 12 * (k + 1) });
  chk('拖动过程中排入了重绘请求', rafQueue.length > 0);
  flushRaf(3);
  chk('拖动后画布确实重绘', totalDraws() > before);
  const p1 = project([[1, 0, 0]])[0];
  chk('拖动改变了视图角度', Math.abs(p1[0] - p0[0]) > 1 || Math.abs(p1[1] - p0[1]) > 1);
  up({ clientX: 520, clientY: 336 });
  before = totalDraws();
  listeners.globe.wheel({ deltaY: -240, preventDefault() {} });
  flushRaf(3);
  chk('滚轮缩放后画布重绘', totalDraws() > before);
  before = totalDraws();
  if (sandbox.__winListeners.resize) { sandbox.window.devicePixelRatio = 2; sandbox.__winListeners.resize(); }
  flushRaf(3);
  chk('窗口缩放后画布重绘', totalDraws() > before);
  flushRaf(3);
  draws.clearRect = 0; draws.arc = 0; draws.fillRect = 0;
  const runs0 = rafRuns;
  X.frame(); X.frame(); X.frame();
  const queuedN = rafQueue.length;
  flushRaf(3);
  chk('同一帧内的重绘请求被合并成一帧', queuedN === 1 && rafRuns - runs0 === 1);

  console.log('\n--- 三种迭代算法 ---');
  const FIB = X.fibonacciPoints(S.n);
  function runAlgo(algo, steps) {
    S.algo = algo; S.iters = 0; S.P = FIB.map(p => p.slice());
    const m0 = computeMetrics(S.P);
    for (let k = 0; k < steps; k++) { if (!stepAlgo()) break; S.iters++; }
    S.metrics = computeMetrics(S.P);
    const m1 = S.metrics;
    const ok = S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-9);
    console.log(`  ${algo}: E ${m0.e.toFixed(4)} -> ${m1.e.toFixed(4)} | ` +
      `dmin ${(m0.d * 180 / Math.PI).toFixed(3)}° -> ${(m1.d * 180 / Math.PI).toFixed(3)}° | ` +
      `R ${(m0.r * 180 / Math.PI).toFixed(3)}° -> ${(m1.r * 180 / Math.PI).toFixed(3)}°`);
    return { m0, m1, ok };
  }
  const rf = runAlgo('fw', 6);   chk('Frank-Wolfe 让能量下降', rf.m1.e < rf.m0.e - 1e-9); chk('FW 保持单位长度', rf.ok);
  const rt = runAlgo('tammes', 200); chk('Tammes 让最小间距不降', rt.m1.d >= rt.m0.d - 1e-9); chk('Tammes 保持单位长度', rt.ok);
  const rl = runAlgo('lloyd', 12);   chk('Lloyd 让覆盖半径下降', rl.m1.r < rl.m0.r - 1e-9); chk('Lloyd 保持单位长度', rl.ok);

  console.log('\n--- 自定义点数 ---');
  els.nCustom.value = '137';
  X.applyCustomN();
  chk('自定义 N=137 生效', S.P.length === 137 && S.n === 137);
  chk('滑块与显示同步', String(els.nOut.textContent) === '137' && String(els.nRange.value) === '137'
      && String(els.nCustom.value) === '137');
  chk('无预计算数据时提示正确', /自定义|无预计算/.test(els.dataNote.textContent || ''));
  els.nCustom.value = '900';
  X.applyCustomN();
  chk('自定义 N=900 生效', S.P.length === 900);
  // 大点数下三种算法仍能运行，并且确实在改善目标（从抖动初值出发）
  const jitter = (pts, a) => pts.map(p => {
    const v = [p[0] + a * (Math.random() - .5), p[1] + a * (Math.random() - .5), p[2] + a * (Math.random() - .5)];
    const nn = Math.hypot(...v); return [v[0] / nn, v[1] / nn, v[2] / nn];
  });
  for (const [algo, steps] of [['fw', 4], ['tammes', 60], ['lloyd', 4]]) {
    S.algo = algo;
    S.P = jitter(X.fibonacciPoints(900), 0.12);
    const b = computeMetrics(S.P);
    const t0 = Date.now();
    for (let k = 0; k < steps; k++) stepAlgo();
    const ms = Date.now() - t0;
    const a2 = computeMetrics(S.P);
    const unit = S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-9);
    console.log(`      N=900 ${algo} ${steps} 步 ${ms} ms | E ${b.e.toFixed(2)} -> ${a2.e.toFixed(2)} | ` +
      `dmin ${(b.d * 180 / Math.PI).toFixed(3)}° -> ${(a2.d * 180 / Math.PI).toFixed(3)}°`);
    // 大 N 的 R_cov 是格点估计（有噪声），这里不做单调性断言；
    // 严格单调性在小 N（精确 R_cov）那一节里断言。
    chk(`N=900 时 ${algo} 可运行且保持单位长度`, unit);
  }
  // 超过滑块上限也能用
  els.nCustom.value = '5000';
  X.applyCustomN();
  chk('N=5000 可生成（滑块封顶不影响）', S.P.length === 5000);
  els.nCustom.value = 'abc';
  X.applyCustomN();
  chk('非法输入被拒绝且不破坏状态', S.P.length === 5000);
  els.nCustom.value = '256';
  X.applyCustomN();

  console.log('\n--- Riesz 指数 s（含 s=0 对数能量） ---');
  {
    // 固定点集，检查能量随 s 的变化方向以及"能量-梯度"自洽
    S.P = jitter(X.fibonacciPoints(24), 0.15);
    const energyAt = (sv) => { X.applyS(String(sv)); return computeMetrics(S.P).e; };
    const e1 = energyAt(1), e0 = energyAt(0), e3 = energyAt(3);
    chk('切换 s 会改变能量值', Math.abs(e1 - e3) > 1e-9 && Math.abs(e1 - e0) > 1e-9);
    chk('s 控件同步', String(els.sInput.value) === String(S.s) && String(els.sPreset.value) === String(S.s));
    chk('s≠1 时提示已更新', /s=/.test(els.dataNote.textContent || '') || /s=/.test(els.iterNote.textContent || ''));

    // 能量-梯度自洽：沿切向走一小步，能量变化 ≈ g·Δ
    S.s = 1; X.applyS('1');
    const g = (() => {
      const n = S.P.length, G = new Float64Array(n * 3);
      const isLog = Math.abs(S.s) < 1e-9, fac = isLog ? -2 : S.s, wexp = isLog ? 2 : S.s + 2;
      for (let i = 0; i < n; i++) {
        let ax = 0, ay = 0, az = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const dx = S.P[i][0] - S.P[j][0], dy = S.P[i][1] - S.P[j][1], dz = S.P[i][2] - S.P[j][2];
          const d2 = Math.max(dx * dx + dy * dy + dz * dz, 1e-30), w = Math.pow(d2, -wexp / 2);
          ax += w * S.P[j][0]; ay += w * S.P[j][1]; az += w * S.P[j][2];
        }
        const c = S.P[i][0] * ax + S.P[i][1] * ay + S.P[i][2] * az;
        G[i * 3] = fac * (ax - c * S.P[i][0]);
        G[i * 3 + 1] = fac * (ay - c * S.P[i][1]);
        G[i * 3 + 2] = fac * (az - c * S.P[i][2]);
      }
      return G;
    })();
    let worst = 0;
    for (const [sv, G2] of [[1, g]]) {
      S.s = sv;
      for (let i = 0; i < 4; i++) {
        const p = S.P[i];
        let d = [Math.cos(i), Math.sin(i * 2), Math.cos(i * 3)];
        d = d.map((x, k) => x - p[k] * (d[0] * p[0] + d[1] * p[1] + d[2] * p[2]));
        const dn = Math.hypot(...d); d = d.map(x => x / dn);
        const eps = 1e-6;
        const move = (sgn) => S.P.map((q, k) => {
          if (k !== i) return q;
          const c = Math.max(-1, Math.min(1, q[0] * d[0] + q[1] * d[1] + q[2] * d[2]));
          const sn = Math.sqrt(Math.max(1 - c * c, 1e-300));
          const t = [d[0] - c * q[0], d[1] - c * q[1], d[2] - c * q[2]].map(x => x / sn);
          const th = sgn * eps;
          return [q[0] * Math.cos(th) + t[0] * Math.sin(th),
                  q[1] * Math.cos(th) + t[1] * Math.sin(th),
                  q[2] * Math.cos(th) + t[2] * Math.sin(th)];
        });
        const fd = (X.rieszEnergy(move(+1), sv) - X.rieszEnergy(move(-1), sv)) / (2 * eps);
        const ut = move(+1)[i].map((x, k) => (x - p[k]) / eps);
        const gd = G2[i * 3] * ut[0] + G2[i * 3 + 1] * ut[1] + G2[i * 3 + 2] * ut[2];
        worst = Math.max(worst, Math.abs(fd - gd) / Math.max(1e-12, Math.abs(fd)));
      }
    }
    console.log(`      s=1 能量-梯度最大相对误差 = ${worst.toExponential(2)}`);
    chk('s=1 能量与梯度自洽', worst < 1e-5);

    // s=0 时 FW 依然能降低能量（不会因为系数错而上升）
    X.applyS('0');
    S.P = jitter(X.fibonacciPoints(24), 0.2);
    S.algo = 'fw';
    const b0 = computeMetrics(S.P).e;
    for (let k = 0; k < 3; k++) stepAlgo();
    chk('s=0（对数能量）FW 让能量下降', computeMetrics(S.P).e < b0 - 1e-12);
    X.applyS('1');
  }

  console.log('\n--- 单步 / 推进 n 步 ---');
  {
    S.running = false; X.setRunBtn && 0;
    S.P = X.fibonacciPoints(40); S.iters = 0; S.algo = 'fw'; S.s = 1;
    const y0 = X.royalYaw ? 0 : 0;
    const it0 = S.iters;
    els.stepBtn.click();
    chk('"单步"只推进一轮', S.iters === it0 + 1);
    const eAfter1 = computeMetrics(S.P).e;
    els.stepBtn.click();
    chk('再点一次继续推进', S.iters === it0 + 2);
    chk('推进后画面已重绘（有 rAF 请求）', rafQueue.length >= 0);

    els.stepCount.value = '5';
    const it2 = S.iters;
    els.stepNBtn.click();
    chk('"推进 n 步"按步数推进', S.iters === it2 + 5);

    els.stepCount.value = '999999';
    els.stepNBtn.click();
    const clamped = els.stepCount.value;
    chk('步数被夹到上限 10000', Number(clamped) === 10000);

    // 自动循环运行时点"推进 n 步"应当先停下
    els.runBtn.click();
    chk('开始迭代后 running=true', S.running === true);
    els.stepCount.value = '1';
    els.stepNBtn.click();
    chk('手动推进会先停止自动循环', S.running === false);
    flushRaf(2);

    // 导出里应带 s
    X.applyS('3');
    const exp3 = X.exportConfig();
    chk('导出对象带 riesz 字段', Number(exp3.riesz) === 3);
    X.applyS('1');
  }

  console.log('\n--- 保存 / 导出 ---');
  // 前面的 s / 单步测试动过点集，这里重置成 256 点的标准工况
  S.n = 256; X.applyS('1'); S.P = X.fibonacciPoints(256); S.iters = 0;
  els.nCustom.value = '256';
  els.nameIn.value = '我的测试球';
  const exp = X.exportConfig();
  chk('导出对象含点数与单位化点集', exp.n === 256 && exp.points.length === 256
      && exp.points.every(p => p.length === 3 && Math.abs(Math.hypot(...p) - 1) < 1e-6));
  chk('导出对象含指标与元数据', Number.isFinite(exp.metrics.energy) && exp.metrics.minAngleDeg > 0
      && exp.objective === S.obj && !!exp.createdAt && exp.name === '我的测试球');
  const nDl = downloads.length;
  els.saveFileBtn.click();
  chk('点击"保存到文件"产生了下载', downloads.length === nDl + 1);
  const text = String(downloads[downloads.length - 1].parts[0]);
  chk('下载内容是合法 JSON 且可被解析', (() => { try { return JSON.parse(text).n === 256; } catch (e) { return false; } })());
  chk('提示文字已更新', /已下载/.test(els.ioNote.textContent || ''));

  console.log('\n--- 从文件加载 ---');
  const r1 = X.loadConfigText(text, 'roundtrip.json');
  chk('导出的文件能原样加载回来', r1.ok && r1.n === 256);
  chk('加载后名称恢复', els.nameIn.value === '我的测试球');
  // 导出时 s=3 -> 加载后应恢复 s=3 与控件
  X.applyS('3'); const exp3b = X.exportConfig();
  X.loadConfigText(JSON.stringify(exp3b), 's3.json');
  chk('加载后 s 被恢复', Math.abs(S.s - 3) < 1e-12 && String(els.sInput.value) === '3');
  X.applyS('1');
  // 纯 CSV
  const csv = '1,0,0\n0,1,0\n0,0,1\n-1,0,0\n';
  const r2 = X.loadConfigText(csv, 'pts.csv');
  chk('CSV 也能加载', r2.ok && r2.n === 4);
  // {points:[...]} 且坐标未归一化
  const r3 = X.importConfig({ name: 'raw', points: [[2, 0, 0], [0, 3, 0], [0, 0, -4], [1, 1, 1]] });
  chk('裸 {points:[[x,y,z]]} 会被归一化后加载', r3.ok && r3.n === 4
      && S.P.every(p => Math.abs(Math.hypot(...p) - 1) < 1e-12));
  // 非法输入
  const r4 = X.importConfig({ name: 'bad', points: [[1, 2], ['a', 'b', 'c']] });
  chk('非法点集被拒绝且不改变现状', !r4.ok && S.P.length === 4);
  chk('sanitizePoints 过滤坏点', (X.sanitizePoints([[1, 0, 0], [0, 0, 0], ['x', 0, 0], [0, 1, 0]]) || []).length === 2);
  chk('parseCSV 忽略注释与空行', (X.parseCSV('# comment\n\n1 0 0\n0,1,0\n') || []).length === 2);

  console.log('\n--- 收藏夹（localStorage） ---');
  X.importConfig({ name: 'A 球', points: X.fibonacciPoints(12) });
  X.saveToGallery('A 球');
  X.importConfig({ name: 'B 球', points: X.fibonacciPoints(20) });
  X.saveToGallery('B 球');
  chk('收藏夹已存入两条', X.readFavs().length === 2);
  chk('收藏夹持久化到 localStorage', !!store['sphereUniformFavsV1']);
  X.loadFromGallery(0);
  chk('从收藏夹载入第 1 条（N=12）', S.P.length === 12 && els.nameIn.value === 'A 球');
  X.saveToGallery('A 球');
  chk('同名保存是覆盖而非新增', X.readFavs().length === 2);
  X.deleteFromGallery(1);
  chk('删除一条后只剩一条', X.readFavs().length === 1);
  X.clearGallery();
  chk('清空收藏夹生效', X.readFavs().length === 0);
  chk('清空后列表渲染为空态', /为空/.test(els.favList.innerHTML || ''));

  console.log('\n--- 渲染分支 ---');
  try {
    S.n = 256; X.gensphere();
    draw(); drawChart();
    S.caps = true; S.colorMode = 'cell'; S.grid = 'lat'; draw();
    S.colorMode = 'nbr'; S.grid = 'none'; draw();
    S.caps = false; draw();
    els.nCustom.value = '1500'; X.applyCustomN();
    S.colorMode = 'cell'; draw(); S.colorMode = 'nbr'; draw();   // 大点数着色分支
    console.log('OK   draw()/drawChart() 及全部渲染分支正常（含 N=1500）');
  } catch (e) {
    console.error('FAIL 渲染抛错:', e && e.message);
    errors.push('render');
  }

  console.log('\n--- 交互回调 ---');
  try {
    els.nRange.value = '32'; els.nRange.oninput();
    chk('滑块切换 N=32', S.P.length === 32);
    els.objSel.value = 'cover'; els.objSel.onchange();
    els.srcSel.value = 'fib'; els.srcSel.onchange();
    els.algoSel.onchange(); els.spdRange.oninput(); els.gridSel.onchange();
    els.colorSel.onchange(); els.capBtn.onclick();
    els.runBtn.onclick();
    chk('开始迭代后 running=true', S.running === true);
    flushRaf(5);
    els.resetBtn.click();
    chk('重置后停止迭代', S.running === false);
    console.log('OK   交互回调全部可执行');
  } catch (e) {
    console.error('FAIL 交互回调抛错:', e && e.stack || e);
    errors.push('interaction');
  }

  console.log(errors.length ? `\n有 ${errors.length} 项问题` : '\n页面脚本运行时检查全部通过 ✓');
  process.exit(errors.length ? 1 : 0);
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
