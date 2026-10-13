// Draggable pixel-art basketball with a hoop on the right edge. Decorative: the page works the same without it.
// Physics is scaled from a real ball (24 cm across), with gravity slowed down a little so throws feel floatier.
import { pixelCanvas, onPointer, WHITE, BG, alpha } from './pixel.js';

const g = pixelCanvas(2);             // one art pixel = 2×2 screen pixels
const P = g.P;
const R = 22;                         // ball radius in screen px (11 art pixels)
const PX_PER_M = (2 * R) / 0.24;
const G = 9.81 * PX_PER_M;            // real gravity
const AIR = 0.04;                     // linear air drag per second
const FLOOR_E = 0.82;                 // restitution of a basketball on a hard floor
const RIM_E = 0.55, BOARD_E = 0.62, WALL_E = 0.6;
const ROLL_DECEL = 0.3 * PX_PER_M;    // rolling resistance on the ground
const RIM_R = 3;                      // rim tube radius
const MAX_THROW = 11 * PX_PER_M;
const THROW_GAIN = 0.5;               // the ball leaves at half the mouse speed: flicks stay gentle
const GRAB = R + 26;                  // generous grab ring: easy to catch a moving ball
const HOOP_SCALE = 1.4;               // hoop drawn bigger than regulation relative to the ball

let W, H, hoop;
const ball = { x: 0, y: 0, vx: 0, vy: 0, a: 0, w: 0 };
let drag = null, samples = [], running = false, last = 0;
const pops = [];                      // "+1" labels

// net: a 3D tube of verlet knots hanging from the rim under gravity, drawn flat (orthographic, front strands
// brighter). Diagonal strands make the diamond mesh and only pull; soft rings keep the tube tapered but open,
// so the bottom ends in loose loops. The ball is a sphere in the hoop's plane and pushes knots out of its way.
const NET_ROWS = 4, NET_AROUND = 12;
const NET_DAMP = 0.992;               // per substep: swings die out over a couple of seconds
const NET_ITER = 6;                   // constraint passes per substep: higher = less stretchy strands
const NET_RING = 0.08;                // ring stiffness: how firmly the tube keeps its taper
const NET_DRAG = 5;                   // ball slow-down per px of net it pushes aside
let net = [], links = [], rings = [], netAwake = false;

function buildNet() {
  const rx = (hoop.back - hoop.front) / 2, cx = hoop.front + rx;
  net = [];
  for (let r = 0; r <= NET_ROWS; r++) for (let i = 0; i < NET_AROUND; i++) {
    const t = 2 * Math.PI * (i + (r % 2) / 2) / NET_AROUND, rr = rx * (1 - 0.72 * r / NET_ROWS);
    const x = cx + rr * Math.cos(t), y = hoop.rimY + 2 * P + (hoop.depth - 2 * P) * r / NET_ROWS, z = rr * Math.sin(t);
    net.push({ x, y, z, px: x, py: y, pz: z, r });
  }
  const at = (r, i) => net[r * NET_AROUND + (i + NET_AROUND) % NET_AROUND];
  const len = (a, c) => Math.hypot(a.x - c.x, a.y - c.y, a.z - c.z);
  links = []; rings = [];
  for (let r = 0; r <= NET_ROWS; r++) for (let i = 0; i < NET_AROUND; i++) {
    const a = at(r, i), c = at(r, i + 1);
    if (r) rings.push([a, c, len(a, c)]);
    if (r === NET_ROWS) continue;
    for (const j of r % 2 ? [i, i + 1] : [i - 1, i]) { const d = at(r + 1, j); links.push([a, d, len(a, d)]); }
  }
  for (let k = 0; k < 600; k++) stepNet(1 / 240, false);       // settle into the hanging shape before the first frame
}

function stepNet(dt, touch = true) {
  let moving = false;
  for (const n of net) {
    if (!n.r) continue;                                         // top row is tied to the rim
    const vx = (n.x - n.px) * NET_DAMP, vy = (n.y - n.py) * NET_DAMP, vz = (n.z - n.pz) * NET_DAMP;
    n.px = n.x; n.py = n.y; n.pz = n.z;
    n.x += vx; n.y += vy + G * dt * dt; n.z += vz;
    if (Math.abs(vx) + Math.abs(vy) + Math.abs(vz) > 0.01) moving = true;
  }
  const pull = (a, c, rest, k, slack) => {
    const dx = c.x - a.x, dy = c.y - a.y, dz = c.z - a.z, d = Math.hypot(dx, dy, dz) || 1;
    if (slack && d <= rest) return;                             // strands go slack, never push
    const f = (d - rest) / d * k, wa = a.r ? 0.5 : 0, wc = a.r ? 0.5 : 1;
    a.x += dx * f * wa; a.y += dy * f * wa; a.z += dz * f * wa;
    c.x -= dx * f * wc; c.y -= dy * f * wc; c.z -= dz * f * wc;
  };
  for (let k = 0; k < NET_ITER; k++) {
    for (const [a, c, rest] of links) pull(a, c, rest, 1, true);
    for (const [a, c, rest] of rings) pull(a, c, rest, NET_RING, false);
    if (!touch) continue;
    for (const n of net) {
      if (!n.r) continue;
      const dx = n.x - ball.x, dy = n.y - ball.y, d = Math.hypot(dx, dy, n.z);
      if (d >= R || d === 0) continue;
      const push = R - d, nx = dx / d, ny = dy / d;
      n.x += nx * push; n.y += ny * push; n.z += n.z / d * push;
      ball.vx -= nx * push * NET_DRAG; ball.vy -= ny * push * NET_DRAG;
    }
  }
  netAwake = moving;
}

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
  buildNet();
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
  stepNet(dt);
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
    // fixed lighting (it does not spin with the ball): dithered shadow bottom-right, glint top-left
    const shade = !seam && x + y > r * 0.25 && ((x + y) & 1) === 0;
    const glint = !seam && Math.hypot(x + r * 0.45, y + r * 0.45) < 1.3;
    g.px(cx + x, cy + y, seam ? WHITE : glint ? alpha(0.7) : shade ? '#2b3a50' : BG);
  }
}

function draw() {
  g.clear();
  const q = v => Math.round(v / P);
  const bx = q(hoop.boardX), bw = q(hoop.boardW), bt = q(hoop.boardTop), bb = q(hoop.boardBottom);
  const ry = q(hoop.rimY), f = q(hoop.front), b = q(hoop.back);
  const right = g.cols - 1, floor = Math.floor((H - 1) / P);

  // pole: 3 px wide with a lit edge, a base plate on the floor and two struts holding the board
  const mid = (bt + bb) >> 1;
  g.rect(right - 3, bb + 4, 3, floor - bb - 3, alpha(0.3));
  g.line(right - 3, bb + 4, right - 3, floor, alpha(0.6));
  g.rect(right - 9, floor - 1, 10, 2, alpha(0.6));
  g.line(right - 3, bb + 4, bx + bw, mid, alpha(0.5));
  g.line(right - 3, bb + 16, bx + bw, bb - 3, alpha(0.5));
  // backboard, seen edge-on: white slab with a darker core
  g.rect(bx, bt, bw, bb - bt, WHITE);
  g.rect(bx + 1, bt + 2, Math.max(bw - 2, 1), bb - bt - 4, alpha(0.4));

  // net: diagonals between the simulated nodes form the diamond mesh
  for (const [a, c] of links) g.line(a.x / P, a.y / P, c.x / P, c.y / P, alpha(a.z + c.z < 0 ? 0.45 : 0.2));

  // rim: a thin ellipse in perspective, the far half dim and the near half bright and 2 px thick
  const rx = (b - f) / 2, cx = f + rx;
  for (let k = 0; k < 24; k++) {
    const t0 = Math.PI * k / 24, t1 = Math.PI * (k + 1) / 24;
    const x0 = cx + rx * Math.cos(t0), x1 = cx + rx * Math.cos(t1), s0 = 2 * Math.sin(t0), s1 = 2 * Math.sin(t1);
    g.line(x0, ry - s0, x1, ry - s1, alpha(0.55));
    g.line(x0, ry + s0, x1, ry + s1, WHITE); g.line(x0, ry + 1 + s0, x1, ry + 1 + s1, WHITE);
  }
  g.rect(b, ry, bx - b, 2, WHITE);                                         // bracket to the board

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
  else for (let i = 0; i < 4; i++) stepNet(dt / 4);                          // a held ball can still push the net
  for (const p of pops) p.t += dt;
  while (pops.length && pops[0].t > 1.1) pops.shift();
  draw();
  const resting = !drag && ball.vx === 0 && ball.vy === 0 && ball.y >= H - R && !pops.length && !netAwake;
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
      const k = Math.min(THROW_GAIN, MAX_THROW / (Math.hypot(b.x - a.x, b.y - a.y) / s || 1));
      ball.vx = (b.x - a.x) / s * k; ball.vy = (b.y - a.y) / s * k;
      ball.w = ball.vx / R;
    }
    drag = null;
    wake();
  },
});

addEventListener('resize', layout);
layout();
