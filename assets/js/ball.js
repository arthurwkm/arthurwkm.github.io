// Draggable basketball with a hoop on the right edge. Decorative: the page works the same without it.
// Physics is scaled from a real ball (24 cm across): gravity, bounce and hoop size are all in real proportions.
(() => {
  const canvas = document.createElement('canvas');
  canvas.className = 'court';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);
  const ctx = canvas.getContext('2d');

  const R = 14;                       // ball radius in px (24 cm ball → 1 m ≈ 117 px)
  const PX_PER_M = (2 * R) / 0.24;
  const G = 9.81 * PX_PER_M;          // gravity
  const AIR = 0.04;                   // linear air drag per second (light; a ball barely slows in flight)
  const FLOOR_E = 0.82;               // coefficient of restitution of a basketball on a hard floor
  const RIM_E = 0.55, BOARD_E = 0.62, WALL_E = 0.6;
  const ROLL_DECEL = 0.35 * PX_PER_M; // rolling resistance once it is on the ground
  const RIM_R = 2;                    // rim tube radius
  const MAX_THROW = 14 * PX_PER_M;   // ~14 m/s, a hard throw

  let W, H, dpr, hoop;
  const ball = { x: 0, y: 0, vx: 0, vy: 0, a: 0, w: 0 };
  let drag = null, samples = [], running = false, last = 0;
  const pops = [];                    // "+1" labels

  function layout() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // regulation rim is 3.05 m up and 45 cm wide; capped so it stays on short screens
    const rimY = H - Math.min(3.05 * PX_PER_M, H * 0.55);
    const boardX = W - 14;
    const rimW = 0.45 * PX_PER_M;
    hoop = {
      boardX, boardTop: rimY - 0.9 * PX_PER_M, boardBottom: rimY + 0.15 * PX_PER_M, boardW: 6,  // 1.05 m board, rim 15 cm above its bottom
      rimY, front: boardX - 0.15 * PX_PER_M - rimW, back: boardX - 0.15 * PX_PER_M,
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
      pops.push({ t: 0, x: hoop.front - 26, y: hoop.rimY - 18 });
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

  // ---------- drawing ----------
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const { boardX, boardW, boardTop, boardBottom, rimY, front, back } = hoop;

    // backboard (seen edge-on from the side) and its pole
    ctx.fillStyle = 'rgba(226, 232, 240, 0.55)';
    ctx.fillRect(boardX, boardTop, boardW, boardBottom - boardTop);
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.3)';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(boardX + boardW, (boardTop + boardBottom) / 2); ctx.lineTo(W - 4, boardBottom + 12);
    ctx.lineTo(W - 4, H); ctx.stroke();

    // net: strings narrowing towards the bottom, crossed diagonally
    const n = 6, depth = 0.45 * PX_PER_M, inset = (back - front) * 0.2;
    const top = t => front + (back - front) * t, bot = t => front + inset + (back - front - 2 * inset) * t;
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      ctx.moveTo(top(i / n), rimY); ctx.lineTo(bot(i / n), rimY + depth);
      if (i < n) { ctx.moveTo(top(i / n), rimY); ctx.lineTo(bot((i + 1) / n), rimY + depth); }
      if (i > 0) { ctx.moveTo(top(i / n), rimY); ctx.lineTo(bot((i - 1) / n), rimY + depth); }
    }
    ctx.moveTo(bot(0), rimY + depth); ctx.lineTo(bot(1), rimY + depth);
    ctx.stroke();

    // rim and the arm holding it to the board
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(front, rimY); ctx.lineTo(boardX, rimY); ctx.stroke();

    // ball
    ctx.save();
    ctx.translate(ball.x, ball.y); ctx.rotate(ball.a);
    const grad = ctx.createRadialGradient(-R * 0.35, -R * 0.35, R * 0.2, 0, 0, R);
    grad.addColorStop(0, '#fb923c'); grad.addColorStop(1, '#c2410c');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(0, 0, R, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(30, 20, 15, 0.85)';
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(-R, 0); ctx.lineTo(R, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, -R); ctx.lineTo(0, R); ctx.stroke();
    ctx.beginPath(); ctx.arc(-R * 1.25, 0, R * 0.95, -0.8, 0.8); ctx.stroke();
    ctx.beginPath(); ctx.arc(R * 1.25, 0, R * 0.95, Math.PI - 0.8, Math.PI + 0.8); ctx.stroke();
    ctx.restore();

    // "+1"
    ctx.font = '700 15px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (const p of pops) {
      ctx.globalAlpha = Math.max(0, 1 - p.t / 1.1);
      ctx.fillStyle = '#7cc4ff';
      ctx.fillText('+1', p.x, p.y - p.t * 22);
    }
    ctx.globalAlpha = 1;
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
  const onBall = (x, y) => Math.hypot(x - ball.x, y - ball.y) <= R + 6;

  function start(x, y, id) {
    drag = { dx: ball.x - x, dy: ball.y - y, id };
    samples = [{ x, y, t: performance.now() }];
    ball.vx = ball.vy = 0;
    document.documentElement.classList.add('dragging-ball');
    wake();
  }

  function move(x, y) {
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
