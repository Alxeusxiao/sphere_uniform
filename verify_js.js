/* 从 sphere_uniform.html 抽出前端数学函数，与 Python 内核做交叉验证。
   跑法： node verify_js.js  （需要先有 sphere_data.json） */
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('sphere_uniform.html', 'utf8');
const start = html.indexOf('const clamp=');
const end = html.lastIndexOf('/* ====', html.indexOf('1. 数据'));
if (start < 0 || end < 0) { console.error('抽不出函数段', start, end); process.exit(1); }
const src = html.slice(start, end);

const ctx = {};
vm.createContext(ctx);
vm.runInContext(src + '\n;this.API={fibonacciPoints,minAngle,rieszEnergy,energyGrad,dminSubgrad,voronoiCells,cellArea,coverageRadius,gram,dot,cross,norm3};', ctx);
const A = ctx.API;

let fails = 0;
const chk = (name, a, b, tol) => {
  const ok = Math.abs(a - b) <= tol;
  if (!ok) fails++;
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${name}: js=${a.toFixed(8)} py=${b.toFixed(8)} diff=${Math.abs(a-b).toExponential(2)}`);
};

/* 1. 正多面体：JS 的 minAngle / coverageRadius */
const phi = (1 + Math.sqrt(5)) / 2;
const ico = [];
for (const s1 of [1, -1]) for (const s2 of [1, -1])
  ico.push([0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]);
const tetra = A.norm3 ? [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]].map(p=>{const n=Math.hypot(...p);return p.map(x=>x/n);}) : null;
const octa = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const icoN = ico.map(p => { const n = Math.hypot(...p); return p.map(x => x / n); });

chk('tetra minAngle(deg)', A.minAngle(tetra) * 180 / Math.PI, 109.47122063449069, 1e-9);
chk('tetra coverageR(deg)', A.coverageRadius(tetra) * 180 / Math.PI, 70.52877936550931, 1e-6);
chk('octa minAngle(deg)', A.minAngle(octa) * 180 / Math.PI, 90.0, 1e-9);
chk('octa coverageR(deg)', A.coverageRadius(octa) * 180 / Math.PI, 54.73561031724535, 1e-6);
chk('ico minAngle(deg)', A.minAngle(icoN) * 180 / Math.PI, 63.43494882292201, 1e-9);
chk('ico coverageR(deg)', A.coverageRadius(icoN) * 180 / Math.PI, 37.37736814064969, 1e-6);

/* 2. 与 Python 在真实 N 上对比能量 / dmin / 覆盖半径 */
const data = JSON.parse(fs.readFileSync('sphere_data.json', 'utf8'));
for (const n of Object.keys(data).map(Number).sort((a, b) => a - b).slice(0, 12)) {
  const rec = data[String(n)];
  for (const key of ['energy', 'tammes', 'cover']) {
    const P = rec.configs[key].pts;
    const E = A.rieszEnergy(P, 1);
    const d = A.minAngle(P) * 180 / Math.PI;
    const R = A.coverageRadius(P) * 180 / Math.PI;
    const pyD = key === 'tammes' ? rec.configs.tammes.minDeg
      : (key === 'cover' ? rec.configs.cover.minDeg : rec.configs.energy.minDeg);
    const pyE = rec.configs[key].energy !== undefined ? rec.configs[key].energy
      : rec.configs.energy.energy;
    const pyR = rec.configs.cover.radiusDeg;
    if (key === 'cover') chk(`n=${n} ${key} R_cov`, R, pyR, 1e-4);
    chk(`n=${n} ${key} minAngle`, d, pyD, 1e-4);
    chk(`n=${n} ${key} energy`, E, pyE, 1e-3);
  }
}

/* 3. 梯度一致性：解析梯度 vs 中心差分 */
const rng = (() => { let s = 12345; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
const P = [];
for (let i = 0; i < 6; i++) P.push([rng() - .5, rng() - .5, rng() - .5]);
const Pn = P.map(p => { const n = Math.hypot(...p); return p.map(x => x / n); });
const G = A.energyGrad(Pn, 1);
let maxrel = 0;
for (let i = 0; i < 6; i++) {
  let q = [rng() - .5, rng() - .5, rng() - .5];
  const qn = Math.hypot(...q); q = q.map(x => x / qn);
  const eps = 1e-6;
  const mk = (t) => Pn.map((p, k) => k !== i ? p : (() => {
    const c = Math.max(-1, Math.min(1, p[0]*q[0]+p[1]*q[1]+p[2]*q[2]));
    const sn = Math.sqrt(Math.max(1 - c*c, 0));
    const ut = [(q[0]-c*p[0])/sn, (q[1]-c*p[1])/sn, (q[2]-c*p[2])/sn];
    return [p[0]*Math.cos(t)+ut[0]*Math.sin(t), p[1]*Math.cos(t)+ut[1]*Math.sin(t), p[2]*Math.cos(t)+ut[2]*Math.sin(t)];
  })());
  const fd = (A.rieszEnergy(mk(eps), 1) - A.rieszEnergy(mk(-eps), 1)) / (2 * eps);
  const ut = (mk(eps)[i][0]-Pn[i][0])/eps*0 + 0;
  const dir = [(mk(eps)[i][0]-Pn[i][0])/eps, (mk(eps)[i][1]-Pn[i][1])/eps, (mk(eps)[i][2]-Pn[i][2])/eps];
  const gd = G[i*3]*dir[0]+G[i*3+1]*dir[1]+G[i*3+2]*dir[2];
  maxrel = Math.max(maxrel, Math.abs(fd - gd) / Math.max(1e-12, Math.abs(fd)));
}
console.log(`${maxrel < 1e-5 ? 'OK  ' : 'FAIL'} 能量梯度 vs 中心差分: 最大相对误差 ${maxrel.toExponential(2)}`);
if (maxrel >= 1e-5) fails++;

/* 4. Voronoi 胞腔数应与 n 相同且非退化 */
for (const n of [6, 12, 20, 32, 60, 100]) {
  const rec = data[String(n)];
  if (!rec) continue;
  const cells = A.voronoiCells(rec.configs.cover.pts);
  const bad = cells.filter(c => c.length < 3).length;
  const ok = cells.length === n && bad === 0;
  if (!ok) fails++;
  console.log(`${ok ? 'OK  ' : 'FAIL'} n=${n} Voronoi 胞腔数=${cells.length} 退化=${bad} 平均顶点数=${(cells.reduce((s,c)=>s+c.length,0)/n).toFixed(2)}`);
}

console.log(fails === 0 ? '\n前端 JS 与 Python 内核一致 ✓' : `\n有 ${fails} 项不一致 ✗`);
process.exit(fails === 0 ? 0 : 1);
