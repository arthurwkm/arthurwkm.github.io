// Draggable pixel-art basketball with a hoop on the right edge. Decorative: the page works the same without it.
// Physics is scaled from a real ball (24 cm across), with gravity slowed down a little so throws feel floatier.
import { pixelCanvas, onPointer, WHITE, BG, alpha } from './pixel.js';

const g = pixelCanvas(2);             // one art pixel = 2×2 screen pixels
const P = g.P;
const R = 22;                         // ball radius in screen px (11 art pixels)
const PX_PER_M = (2 * R) / 0.24;
const G = 0.6 * 9.81 * PX_PER_M;      // 60% of real gravity: easier to follow and to aim
const AIR = 0.04;                     // linear air drag per second
const FLOOR_E = 0.82;                 // restitution of a basketball on a hard floor
const RIM_E = 0.55, BOARD_E = 0.62, WALL_E = 0.6;
const ROLL_DECEL = 0.3 * PX_PER_M;    // rolling resistance on the ground
const RIM_R = 3;                      // rim tube radius
const MAX_THROW = 11 * PX_PER_M;
const GRAB = R + 26;                  // generous grab ring: easy to catch a moving ball
const HOOP_SCALE = 1.4;               // hoop drawn bigger than regulation relative to the ball

let W, H, hoop;
const ball = { x: 0, y: 0, vx: 0, vy: 0, a: 0, w: 0 };
let drag = null, samples = [], running = false, last = 0;
const pops = [];                      // "+1" labels

function layout() {
  g.resize();
  W = g.W; H = g.H;
  const m = PX_PER_M * HOOP_SCALE;
  const rimY = H - Math.min(3.05 * PX_PER_M, H * 0.55);
  const boardX = W - 8 * P;
  const rimW = 0.45 * m;
  hoop = {
    boardX, boardW: 3 * P, boardTop: rimY - 0.9 * m, boardBottom: rimY + 0.15 * m,
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
    pops.push({ t: 0, x: hoop.front - 10 * P, y: hoop.rimY - 12 * P });
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

// ---------- drawing (art pixels) ----------
function drawBall() {
  const r = R / P, cx = Math.round(ball.x / P), cy = Math.min(Math.round(ball.y / P), Math.floor((H - 1) / P) - r);
  const cos = Math.cos(ball.a), sin = Math.sin(ball.a);
  const inside = (x, y) => x * x + y * y <= r * r + r * 0.6;
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) {
    if (!inside(x, y)) continue;
    const u = x * cos + y * sin, v = -x * sin + y * cos;          // seams rotate with the ball's spin
    const edge = !inside(x + 1, y) || !inside(x - 1, y) || !inside(x, y + 1) || !inside(x, y - 1);
    const seam = edge || Math.abs(u) < 0.6 || Math.abs(v) < 0.6
      || Math.abs(Math.hypot(u - 1.55 * r, v) - 1.05 * r) < 0.6 || Math.abs(Math.hypot(u + 1.55 * r, v) - 1.05 * r) < 0.6;
    g.px(cx + x, cy + y, seam ? WHITE : BG);
  }
}

function draw() {
  g.clear();
  const q = v => Math.round(v / P);
  const bx = q(hoop.boardX), bw = q(hoop.boardW), bt = q(hoop.boardTop), bb = q(hoop.boardBottom);
  const ry = q(hoop.rimY), f = q(hoop.front), b = q(hoop.back), dep = q(hoop.depth);
  const right = g.cols - 1, floor = Math.floor((H - 1) / P);

  // pole and arm
  g.rect(right - 2, bb + 6, 2, floor - bb - 5, alpha(0.45));
  g.line(right - 2, bb + 6, bx + bw, (bt + bb) >> 1, alpha(0.45));
  g.line(right - 1, bb + 6, bx + bw, ((bt + bb) >> 1) + 1, alpha(0.45));
  // backboard, seen edge-on
  g.rect(bx, bt, bw, bb - bt, WHITE);
  // net: strings narrowing downwards, crossed diagonally
  const n = 6, inset = (b - f) * 0.22;
  const top = i => f + (b - f) * i / n, bot = i => f + inset + (b - f - 2 * inset) * i / n;
  for (let i = 0; i <= n; i++) {
    g.line(top(i), ry + 2, bot(i), ry + dep, alpha(0.3));
    if (i < n) g.line(top(i), ry + 2, bot(i + 1), ry + dep, alpha(0.3));
  }
  g.line(bot(0), ry + dep, bot(n), ry + dep, alpha(0.5));
  // rim (2 px thick) and its bracket to the board
  g.rect(f, ry, bx - f, 2, WHITE);

  drawBall();

  for (const p of pops) {
    const a = Math.max(0, 1 - p.t / 1.1);
    g.text('+1', q(p.x), q(p.y) - Math.round(p.t * 10), alpha(a.toFixed(2)), 2);
  }
}

// ---------- loop: sleeps once the ball is still ----------
function frame(now) {
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  if (!drag) for (let i = 0; i < 4; i++) step(dt / 4);
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

// ---------- dragging ----------
onPointer({
  hit: (x, y) => Math.hypot(x - ball.x, y - ball.y) <= GRAB,
  down(x, y) {
    drag = { dx: ball.x - x, dy: ball.y - y };
    samples = [{ x, y, t: performance.now() }];
    ball.vx = ball.vy = 0;
    wake();
  },
  move(x, y) {
    // ease the grab offset away so the ball slides under the cursor instead of hanging off to the side
    drag.dx *= 0.8; drag.dy *= 0.8;
    ball.x = Math.min(Math.max(x + drag.dx, R), W - R);
    ball.y = Math.min(y + drag.dy, H - R);
    const t = performance.now();
    samples.push({ x, y, t });
    while (samples.length > 2 && t - samples[0].t > 90) samples.shift();
  },
  up() {
    const a = samples[0], b = samples[samples.length - 1], s = (b.t - a.t) / 1000;
    if (s > 0.008) {
      const k = Math.min(1, MAX_THROW / (Math.hypot(b.x - a.x, b.y - a.y) / s || 1));
      ball.vx = (b.x - a.x) / s * k; ball.vy = (b.y - a.y) / s * k;
      ball.w = ball.vx / R;
    }
    drag = null;
    wake();
  },
});

addEventListener('resize', layout);
layout();
