// Systematic autocorrect prober for krillion.io (paste into the page console during a dive).
// kp2(seeds, bank, {stopAt}) probes controlled garbles of each seed answer until `stopAt`
// seconds remain on the clock, then banks `bank`. Each (seed, op) series stops after two
// rejections in a row, so probes go where the acceptance boundary is.
window.kpLog = window.kpLog || [];
window.kp2 = async function (seeds, bank, opts = {}) {
  const stopAt = opts.stopAt ?? 4;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const txt = s => [...document.querySelectorAll(s)].map(e => e.innerText.trim()).join(' | ');
  const timeLeft = () => +(document.querySelector('.ds-time-digits')?.innerText || 0);
  const prompt = (document.body.innerText.match(/PROMPT \d OF 7\n(.+)/) || [])[1];
  const setv = (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  async function send(v) {
    const el = document.querySelector('.ds-input');
    if (!el) return { res: 'no-input' };
    setv(el, v); await sleep(20);
    el.closest('form').requestSubmit();
    for (let i = 0; i < 80; i++) {
      await sleep(40);
      if (!document.querySelector('.ds-input')) return { res: 'scored' };
      if (txt('.ds-correction')) return { res: 'corr', to: document.querySelector('.ds-input').value };
      if (txt('.ds-reject')) return { res: 'rej' };
    }
    return { res: 'timeout' };
  }

  // Deterministic garble series. Each yields {op, k, raw}; k = size of the edit.
  const spaced = (L, k) => { const p = new Set(); for (let i = 0; i < k; i++) p.add(Math.min(L - 1, Math.round((i + 1) * L / (k + 1)))); return p; };
  function series(seed) {
    const s = seed.toLowerCase(), L = s.length, maxK = Math.min(8, L - 2), out = {};
    out.del = []; out.sub = []; out.trunc = []; out.suffix = []; out.swap = [];
    for (let k = 1; k <= maxK; k++) {
      const p = spaced(L, k);
      out.del.push({ op: 'del', k, raw: [...s].filter((c, i) => !p.has(i)).join('') });
      out.sub.push({ op: 'sub', k, raw: [...s].map((c, i) => p.has(i) ? (c === 'x' ? 'q' : 'x') : c).join('') });
      out.swap.push({ op: 'swap', k, raw: (() => { const a = [...s]; for (const i of p) if (i + 1 < L) [a[i], a[i + 1]] = [a[i + 1], a[i]]; return a.join(''); })() });
    }
    for (let k = 1; L - k >= 2; k++) out.trunc.push({ op: 'trunc', k, raw: s.slice(0, L - k) });
    for (let k = 1; k <= L - 2; k++) out.suffix.push({ op: 'suffix', k, raw: 'z'.repeat(k) + s.slice(k) });
    out.novowel = [{ op: 'novowel', k: (s.slice(1).match(/[aeiou]/g) || []).length, raw: s[0] + s.slice(1).replace(/[aeiou]/g, '') }];
    return Object.values(out).map(q => ({ seed, q, rejStreak: 0 }));
  }

  const queues = seeds.flatMap(series), tried = new Set(seeds.map(s => s.toLowerCase())), out = [];
  let qi = 0;
  while (timeLeft() > stopAt && queues.some(q => q.q.length && q.rejStreak < 2)) {
    const Q = queues[qi++ % queues.length];
    if (!Q.q.length || Q.rejStreak >= 2) continue;
    const g = Q.q.shift();
    if (tried.has(g.raw)) continue; tried.add(g.raw);
    const r = await send(g.raw);
    const rec = { prompt, seed: Q.seed, ...g, ...r, t: timeLeft() };
    out.push(rec); kpLog.push(rec);
    Q.rejStreak = r.res === 'rej' ? Q.rejStreak + 1 : 0;
    if (r.res === 'scored' || r.res === 'no-input') return { out, ended: r.res };
  }
  if (bank) {
    let r = await send(bank);
    if (r.res === 'corr') r = { ...(await send(r.to)), via: r.to };
    const rec = { prompt, bank: true, raw: bank, ...r }; out.push(rec); kpLog.push(rec);
  }
  await sleep(1500);
  const tier = (document.body.innerText.match(/(PLANKTON|TOO CLEVER|SCHOOLER|RARE|DEEP CUT|ONE IN A KRILLION)\n“(.+)”\n\+(\d+)/) || []).slice(1);
  [...document.querySelectorAll('button')].find(b => b.innerText.includes('DESCEND'))?.click();
  await sleep(900);
  const next = (document.body.innerText.match(/PROMPT \d OF 7\n(.+)/) || [])[1];
  const summary = {};
  for (const r of out) if (!r.bank) (summary[r.seed + ':' + r.op] ||= []).push(r.k + (r.res === 'corr' ? '→' + r.to : r.res === 'rej' ? '✗' : '·' + r.res));
  return { n: out.length, summary, tier, next };
};
