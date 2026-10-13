/* s 的交叉验证（修正版）：前端通用 Riesz 能量/梯度 vs Python 内核。 */
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('sphere_uniform.html', 'utf8');
const s0 = html.indexOf('const clamp=');
const cut = html.indexOf('/* 切平面正交基');       // 数学函数段的结尾
if (cut < 0) { console.error('找不到数学段结尾'); process.exit(1); }
const ctx = {};
vm.createContext(ctx);
vm.runInContext(html.slice(s0, cut) + ';this.API={rieszEnergy,energyGrad};', ctx);
const A = ctx.API;

const ref = JSON.parse(fs.readFileSync('tmp_cmp_s.json', 'utf8'));
const P = ref.P;
let fails = 0;
console.log('点数 =', P.length, ' 对比的 s =', ref.cases.map(c => c.s).join(', '));
for (const c of ref.cases) {
  const eJs = A.rieszEnergy(P, c.s);
  const eOk = Math.abs(eJs - c.energy) <= 1e-9 * Math.max(1, Math.abs(c.energy));
  if (!eOk) fails++;
  console.log(`${eOk ? 'OK  ' : 'FAIL'} s=${String(c.s).padEnd(5)} 能量 js=${eJs.toFixed(9)} py=${c.energy.toFixed(9)}`);

  const G = A.energyGrad(P, c.s);
  let maxrel = 0;
  for (let i = 0; i < P.length; i++) for (let k = 0; k < 3; k++) {
    const py = c.grad[i][k];
    maxrel = Math.max(maxrel, Math.abs(G[i * 3 + k] - py) / Math.max(1e-12, Math.abs(py)));
  }
  const gOk = maxrel < 1e-9;
  if (!gOk) fails++;
  console.log(`${gOk ? 'OK  ' : 'FAIL'} s=${String(c.s).padEnd(5)} 梯度最大相对误差 = ${maxrel.toExponential(2)}`);
}
console.log(fails ? `\n有 ${fails} 项不一致 ✗` : '\n各 s 下前端 Riesz 能量/梯度与 Python 内核一致 ✓');
process.exit(fails ? 1 : 0);
