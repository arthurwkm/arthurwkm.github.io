// Draggable pixel-art basketball with a hoop on the right edge. Decorative: the page works the same without it.
// Physics is scaled from a real ball (24 cm across), with gravity slowed down a little so throws feel floatier.
(() => {
  const canvas = document.createElement('canvas');
  canvas.className = 'court';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);
  const ctx = canvas.getContext('2d');

  const P = 3;                        // one art pixel = 3×3 screen pixels
  const R = 18;                       // ball radius in screen px (6 art pixels)
  const PX_PER_M = (2 * R) / 0.24;
  const G = 0.6 * 9.81 * PX_PER_M;    // 60% of real gravity: easier to follow and to aim
  const AIR = 0.04;                   // linear air drag per second
  const FLOOR_E = 0.82;               // restitution of a basketball on a hard floor
  const RIM_E = 0.55, BOARD_E = 0.62, WALL_E = 0.6;
  const ROLL_DECEL = 0.3 * PX_PER_M;  // rolling resistance on the ground
  const RIM_R = 2;                    // rim tube radius
  const MAX_THROW = 11 * PX_PER_M;
  const GRAB = R + 26;                // generous grab ring: easy to catch a moving ball
  const HOOP_SCALE = 1.4;             // hoop drawn bigger than regulation relative to the ball

  const WHITE = '#f8fafc', DIM = 'rgba(248, 250, 252, 0.45)', FAINT = 'rgba(248, 250, 252, 0.28)', BG = '#0b1220';

  let W, H, hoop;
  const ball = { x: 0, y: 0, vx: 0, vy: 0, a: 0, w: 0 };
  let drag = null, samples = [], running = false, last = 0;
  const pops = [];                    // "+1" labels

  function layout() {
    W = innerWidth; H = innerHeight;
    canvas.width = Math.ceil(W / P); canvas.height = Math.ceil(H / P);
    canvas.style.width = canvas.width * P + 'px'; canvas.style.height = canvas.height * P + 'px';
    const m = PX_PER_M * HOOP_SCALE;
    const rimY = H - Math.min(3.05 * PX_PER_M, H * 0.55);
    const boardX = W - 6 * P;
    const rimW = 0.45 * m;
    hoop = {
      boardX, boardW: 2 * P, boardTop: rimY - 0.9 * m, boardBottom: rimY + 0.15 * m,
      rimY, front: boardX - 0.15 * m - rimW, back: boardX - 0.15 * m, depth: 0.45 * m,
    };
    // starts in the empty gap between the two columns on desktop, mid-screen on phones
    ball.x = Math.min(Math.max(ball.x || W * (W >= 1024 ? 0.42 : 0.5), R), W - R);
    ball.y = Math.min(ball.y || H - R, H - R);
    wake();
  }

  // ---------- collisions ----------
  function bounceOff(nx, ny, depth, e) {
    ball.x += nx * depth; ball.y += ny * depth;
    const vn = ball.vx * nx + ball.vy * ny;
    if (vn < 0) {
      ball.vx -= (1 + e) * vn * nx; ball.vy -= (1 + e) * vn * ny;
      const tx = -ny, ty = nx;                      // friction along the contact turns into spin
      const vt = ball.vx * tx + ball.vy * ty;
      ball.vx -= 0.08 * vt * tx; ball.vy -= 0.08 * vt * ty;
      ball.w = vt / R;
    }
  }

  function hitPoint(px, py, pr, e) {
    const dx = ball.x - px, dy = ball.y - py, d = Math.hypot(dx, dy);
    if (d > 0 && d < R + pr) bounceOff(dx / d, dy / d, R + pr - d, e);
  }

  function hitBoard() {
    const { boardX, boardW, boardTop, boardBottom } = hoop;
    const cx = Math.max(boardX, Math.min(ball.x, boardX + boardW));
    const cy = Math.max(boardTop, Math.min(ball.y, boardBottom));
    const dx = ball.x - cx, dy = ball.y - cy, d = Math.hypot(dx, dy);
    if (d > 0 && d < R) bounceOff(dx / d, dy / d, R - d, BOARD_E);
  }

  function step(dt) {
    const prevY = ball.y;
    ball.vy += G * dt;
    ball.vx *= 1 - AIR * dt; ball.vy *= 1 - AIR * dt;
    ball.x += ball.vx * dt; ball.y += ball.vy * dt;

    hitPoint(hoop.front, hoop.rimY, RIM_R, RIM_E);
    hitPoint(hoop.back, hoop.rimY, RIM_R, RIM_E);
    hitBoard();

    // score: centre passes down through the rim opening
    if (prevY < hoop.rimY && ball.y >= hoop.rimY && ball.vy > 0 && ball.x > hoop.front && ball.x < hoop.back) {
      pops.push({ t: 0, x: hoop.front - 12 * P, y: hoop.rimY - 6 * P });
    }

    if (ball.x < R) { ball.x = R; ball.vx = Math.abs(ball.vx) * WALL_E; }
    if (ball.x > W - R) { ball.x = W - R; ball.vx = -Math.abs(ball.vx) * WALL_E; }

    const floor = H - R;
    if (ball.y >= floor) {
      ball.y = floor;
      if (ball.vy > 0) ball.vy = ball.vy * FLOOR_E > 40 ? -ball.vy * FLOOR_E : 0;  // tiny bounces die out
      if (ball.vy === 0) {
        const s = Math.sign(ball.vx), dv = ROLL_DECEL * dt;
        ball.vx = Math.abs(ball.vx) <= dv ? 0 : ball.vx - s * dv;
        ball.w = ball.vx / R;                         // rolling without slipping
      }
    }
    ball.a += ball.w * dt;
  }

  // ---------- pixel drawing (all coordinates in art pixels) ----------
  const px = (x, y, c) => { ctx.fillStyle = c; ctx.fillRect(x, y, 1, 1); };
  function line(x0, y0, x1, y1, c) {                 // Bresenham: hard pixel steps, no anti-aliasing
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    ctx.fillStyle = c;
    for (;;) {
      ctx.fillRect(x0, y0, 1, 1);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  // 3×5 glyphs for "+1"
  const GLYPHS = { '+': ['000', '010', '111', '010', '000'], '1': ['010', '110', '010', '010', '111'] };
  function text(str, x, y, c) {
    ctx.fillStyle = c;
    [...str].forEach((ch, i) => GLYPHS[ch].forEach((row, r) =>
      [...row].forEach((b, k) => b === '1' && ctx.fillRect(x + i * 4 + k, y + r, 1, 1))));
  }

  function drawBall() {
    const r = R / P, cx = Math.round(ball.x / P), cy = Math.min(Math.round(ball.y / P), Math.floor((H - 1) / P) - r);
    const cos = Math.cos(ball.a), sin = Math.sin(ball.a);
    for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) {
      const d = Math.hypot(x, y);
      if (d > r + 0.3) continue;
      const u = x * cos + y * sin, v = -x * sin + y * cos;        // rotate with the ball's spin
      const seam = d > r - 0.8 || Math.abs(u) < 0.5 || Math.abs(v) < 0.5
        || Math.abs(Math.hypot(u - 1.55 * r, v) - 1.05 * r) < 0.5 || Math.abs(Math.hypot(u + 1.55 * r, v) - 1.05 * r) < 0.5;
      px(cx + x, cy + y, seam ? WHITE : BG);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const q = v => Math.round(v / P);
    const bx = q(hoop.boardX), bw = q(hoop.boardW), bt = q(hoop.boardTop), bb = q(hoop.boardBottom);
    const ry = q(hoop.rimY), f = q(hoop.front), b = q(hoop.back), dep = q(hoop.depth);
    const right = canvas.width - 1, floor = Math.floor((H - 1) / P);

    // pole and arm
    line(right - 1, floor, right - 1, bb + 4, DIM);
    line(right - 1, bb + 4, bx + bw, (bt + bb) >> 1, DIM);
    // backboard, seen edge-on
    ctx.fillStyle = WHITE; ctx.fillRect(bx, bt, bw, bb - bt);
    // net: strings narrowing downwards, crossed diagonally
    const n = 5, inset = (b - f) * 0.22;
    const top = i => f + (b - f) * i / n, bot = i => f + inset + (b - f - 2 * inset) * i / n;
    for (let i = 0; i <= n; i++) {
      line(top(i), ry + 1, bot(i), ry + dep, FAINT);
      if (i < n) line(top(i), ry + 1, bot(i + 1), ry + dep, FAINT);
    }
    line(bot(0), ry + dep, bot(n), ry + dep, DIM);
    // rim and its bracket to the board
    ctx.fillStyle = WHITE; ctx.fillRect(f, ry, bx - f, 1);
    px(f, ry + 1, WHITE);

    drawBall();

    for (const p of pops) {
      const alpha = Math.max(0, 1 - p.t / 1.1);
      text('+1', q(p.x), q(p.y) - Math.round(p.t * 8), `rgba(248, 250, 252, ${alpha.toFixed(2)})`);
    }
  }

  // ---------- loop: sleeps once the ball is still ----------
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    if (!drag) {
      const n = 4;
      for (let i = 0; i < n; i++) step(dt / n);
    }
    for (const p of pops) p.t += dt;
    while (pops.length && pops[0].t > 1.1) pops.shift();
    draw();
    const resting = !drag && ball.vx === 0 && ball.vy === 0 && ball.y >= H - R && !pops.length;
    if (resting) { running = false; return; }
    requestAnimationFrame(frame);
  }

  function wake() {
    if (running) return;
    running = true; last = performance.now();
    requestAnimationFrame(frame);
  }

  // ---------- dragging (the canvas ignores the pointer, so the page stays clickable) ----------
  const onBall = (x, y) => Math.hypot(x - ball.x, y - ball.y) <= GRAB;

  function start(x, y, id) {
    drag = { dx: ball.x - x, dy: ball.y - y, id };
    samples = [{ x, y, t: performance.now() }];
    ball.vx = ball.vy = 0;
    document.documentElement.classList.add('dragging-ball');
    wake();
  }

  function move(x, y) {
    // ease the grab offset away so the ball slides under the cursor instead of hanging off to the side
    drag.dx *= 0.8; drag.dy *= 0.8;
    ball.x = Math.min(Math.max(x + drag.dx, R), W - R);
    ball.y = Math.min(y + drag.dy, H - R);
    const t = performance.now();
    samples.push({ x, y, t });
    while (samples.length > 2 && t - samples[0].t > 90) samples.shift();
  }

  function end() {
    const a = samples[0], b = samples[samples.length - 1], s = (b.t - a.t) / 1000;
    if (s > 0.008) {
      const k = Math.min(1, MAX_THROW / (Math.hypot(b.x - a.x, b.y - a.y) / s || 1));
      ball.vx = (b.x - a.x) / s * k; ball.vy = (b.y - a.y) / s * k;
      ball.w = ball.vx / R;
    }
    drag = null;
    document.documentElement.classList.remove('dragging-ball');
    wake();
  }

  addEventListener('pointerdown', e => {
    if (e.button !== 0 || !onBall(e.clientX, e.clientY)) return;
    e.preventDefault();
    start(e.clientX, e.clientY, e.pointerId);
  });
  addEventListener('pointermove', e => {
    if (drag && e.pointerId === drag.id) move(e.clientX, e.clientY);
    else if (e.pointerType === 'mouse') document.documentElement.classList.toggle('over-ball', onBall(e.clientX, e.clientY));
  }, { passive: true });
  addEventListener('pointerup', e => { if (drag && e.pointerId === drag.id) end(); });
  addEventListener('pointercancel', e => { if (drag && e.pointerId === drag.id) end(); });
  // on touch screens, stop the page from scrolling while the finger is on the ball
  addEventListener('touchstart', e => {
    const t = e.touches[0];
    if (t && onBall(t.clientX, t.clientY)) e.preventDefault();
  }, { passive: false });
  addEventListener('touchmove', e => { if (drag) e.preventDefault(); }, { passive: false });

  addEventListener('resize', layout);
  layout();
})();
