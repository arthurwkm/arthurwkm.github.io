// Shared pixel-art canvas for the background games (ball.js, bowman.js).
// The canvas is drawn at 1/P resolution and scaled up with crisp pixels; all drawing coordinates are art pixels.

export const WHITE = '#f8fafc';
export const BG = '#0b1220';
export const alpha = a => `rgba(248, 250, 252, ${a})`;

// 3×5 font: only the characters the games print
const FONT = {
  A: '010101111101101', B: '110101110101110', C: '011100100100011', D: '110101101101110', E: '111100110100111',
  G: '011100101101011', H: '101101111101101', I: '111010010010111', K: '101101110101101', L: '100100100100111',
  M: '101111111101101', N: '110101101101101', O: '010101101101010', P: '110101110100100', R: '110101110101101',
  S: '011100010001110', T: '111010010010010', U: '101101101101111', W: '101101111111101', Y: '101101010010010',
  0: '111101101101111', 1: '010110010010111', 2: '110001010100111', 3: '110001010001110', 4: '101101111001001',
  5: '111100110001110', 6: '011100110101010', 7: '111001010010010', 8: '010101010101010', 9: '010101011001110',
  '+': '000010111010000', '-': '000000111000000', '%': '101001010100101', '°': '010101010000000', '!': '010010010000010',
  ' ': '000000000000000',
};

export function pixelCanvas(P) {
  const canvas = document.createElement('canvas');
  canvas.className = 'court';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);
  const ctx = canvas.getContext('2d');
  const g = { canvas, ctx, P, W: 0, H: 0, cols: 0, rows: 0 };

  g.resize = () => {
    g.W = innerWidth; g.H = innerHeight;
    g.cols = canvas.width = Math.ceil(g.W / P);
    g.rows = canvas.height = Math.ceil(g.H / P);
    canvas.style.width = g.cols * P + 'px'; canvas.style.height = g.rows * P + 'px';
  };
  g.clear = () => ctx.clearRect(0, 0, g.cols, g.rows);
  g.px = (x, y, c) => { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), 1, 1); };
  g.rect = (x, y, w, h, c) => { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h)); };

  // Bresenham line; `dash` skips pixels for dotted lines
  g.line = (x0, y0, x1, y1, c, dash = 0) => {
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let err = dx + dy, i = 0;
    ctx.fillStyle = c;
    for (;;) {
      if (!dash || i++ % (dash * 2) < dash) ctx.fillRect(x0, y0, 1, 1);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  };

  // 1-pixel circle outline (optionally filled with `fill`)
  g.circle = (cx, cy, r, c, fill) => {
    cx = Math.round(cx); cy = Math.round(cy);
    const inside = (x, y) => x * x + y * y <= r * r + r * 0.6;
    for (let y = -r - 1; y <= r + 1; y++) for (let x = -r - 1; x <= r + 1; x++) {
      if (!inside(x, y)) continue;
      const edge = !inside(x + 1, y) || !inside(x - 1, y) || !inside(x, y + 1) || !inside(x, y - 1);
      if (edge) g.px(cx + x, cy + y, c); else if (fill) g.px(cx + x, cy + y, fill);
    }
  };

  // text in the 3×5 font, scale = size of each font pixel in art pixels
  g.text = (str, x, y, c, s = 1, align = 'left') => {
    str = String(str).toUpperCase();
    const w = g.textWidth(str, s);
    if (align === 'center') x -= w / 2; else if (align === 'right') x -= w;
    x = Math.round(x); y = Math.round(y);
    ctx.fillStyle = c;
    [...str].forEach((ch, i) => {
      const bits = FONT[ch] || FONT[' '];
      for (let k = 0; k < 15; k++) if (bits[k] === '1') ctx.fillRect(x + (i * 4 + k % 3) * s, y + Math.floor(k / 3) * s, s, s);
    });
  };
  g.textWidth = (str, s = 1) => (String(str).length * 4 - 1) * s;

  return g;
}

// Pointer helpers shared by both games: the canvas ignores the pointer, so hits are tested by distance
// and touch scrolling is blocked only when the finger starts on something interactive.
export function onPointer({ hit, down, move, up, hover }) {
  let active = null;
  addEventListener('pointerdown', e => {
    if (e.button !== 0 || !hit(e.clientX, e.clientY)) return;
    e.preventDefault();
    active = e.pointerId;
    document.documentElement.classList.add('dragging-ball');
    down(e.clientX, e.clientY);
  });
  addEventListener('pointermove', e => {
    if (e.pointerId === active) move(e.clientX, e.clientY);
    else if (e.pointerType === 'mouse') {
      const h = hit(e.clientX, e.clientY);
      document.documentElement.classList.toggle(h === 'button' ? 'over-button' : 'over-ball', !!h);
      if (!h) document.documentElement.classList.remove('over-button', 'over-ball');
      hover?.(e.clientX, e.clientY);
    }
  }, { passive: true });
  const finish = e => {
    if (e.pointerId !== active) return;
    active = null;
    document.documentElement.classList.remove('dragging-ball');
    up(e.clientX, e.clientY);
  };
  addEventListener('pointerup', finish);
  addEventListener('pointercancel', finish);
  addEventListener('touchstart', e => {
    const t = e.touches[0];
    if (t && hit(t.clientX, t.clientY)) e.preventDefault();
  }, { passive: false });
  addEventListener('touchmove', e => { if (active !== null) e.preventDefault(); }, { passive: false });
}
