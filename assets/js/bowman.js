// Bowman: a pixel-art archery duel in the page background, after the classic Flash game.
// You are the archer on the left: press on him, drag back to draw the bow, release to shoot. The AI answers.
import { pixelCanvas, onPointer, WHITE, alpha } from './pixel.js';

const g = pixelCanvas(2);
const P = g.P;
const RED = '#dc2626';
const G = 900;                        // gravity, px/s²
const MAX_PULL = 150;                 // drag distance (px) for a full-power shot
const HEALTH = 100;
const DAMAGE = { head: 50, body: 25 };// at a typical strong shot; scaled by arrow speed
const AI_SPEED_ERR = 0.075, AI_ANGLE_ERR = 3 * Math.PI / 180;   // the AI's aim is deliberately imperfect
const GRAB = 48;

let W, H, VMAX;
const me = { side: 1 }, ai = { side: -1 };
let state, timer, arrow, aim, msg, shooter, round = 0;
const stuck = [], blood = [], popups = [];
let running = false, last = 0;

// ---------- layout ----------
const rand = (a, b) => a + Math.random() * (b - a);
function place(c, x0, x1) { c.fx = rand(x0, x1); c.fy = rand(0.38, 0.88); }
function position(c) {
  c.x = Math.round(c.fx * W / P) * P;
  c.y = Math.max(Math.round(c.fy * H / P) * P, 62 * P);   // feet; keeps the health bar and labels on screen
}

function newRound(fullReset) {
  place(me, 0.06, 0.28); place(ai, 0.72, 0.94);
  for (const c of [me, ai]) { position(c); c.hp = HEALTH; c.dead = 0; c.fall = 0; c.aim = Math.PI / 6; c.pull = 0; }
  if (fullReset) round = 0;
  stuck.length = 0; blood.length = 0; popups.length = 0;
  arrow = null; aim = null; msg = null;
  state = 'player'; timer = 0;
  wake();
}

function layout() {
  g.resize();
  W = g.W; H = g.H;
  VMAX = Math.sqrt(1.4 * W * G);      // full power reaches ~1.4 screen widths at 45°
  if (state) for (const c of [me, ai]) position(c);
  wake();
}

// ---------- geometry (screen px) ----------
const shoulder = c => ({ x: c.x, y: c.y - 27 * P });
const dir = c => ({ x: Math.cos(c.aim) * c.side, y: -Math.sin(c.aim) });
const launchPoint = c => { const s = shoulder(c), d = dir(c); return { x: s.x + d.x * 14 * P, y: s.y + d.y * 14 * P }; };
const head = c => ({ x: c.x, y: c.y - 37 * P, r: 5 * P });
const body = c => ({ x0: c.x - 4 * P, x1: c.x + 4 * P, y0: c.y - 32 * P, y1: c.y });

// ---------- shooting ----------
function fire(c, speed) {
  const p = launchPoint(c), d = dir(c);
  arrow = { x: p.x, y: p.y, vx: d.x * speed, vy: d.y * speed, owner: c, target: c === me ? ai : me };
  c.pull = 0; shooter = c;
  state = 'flying'; timer = 0;
}

function aiAim() {
  const from = launchPoint({ ...ai, aim: Math.PI / 4 });
  const dx = Math.abs(me.x - from.x), up = from.y - (me.y - 18 * P);
  let theta = rand(25, 60) * Math.PI / 180, v2;
  for (let i = 0; i < 20; i++) {                   // find an angle with a reachable solution
    const den = 2 * Math.cos(theta) ** 2 * (dx * Math.tan(theta) - up);
    v2 = den > 0 ? G * dx * dx / den : Infinity;
    if (v2 < (VMAX * 1.05) ** 2) break;
    theta = Math.min(theta + 4 * Math.PI / 180, 1.4);
  }
  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 0.5;   // ~N(0,1)
  ai.aim = theta + gauss() * AI_ANGLE_ERR;
  return Math.min(Math.sqrt(v2) * (1 + gauss() * AI_SPEED_ERR), VMAX * 1.05);
}

function hit(c, part, speed, x, y) {
  const mult = Math.min(Math.max(speed / (0.75 * VMAX), 0.3), 1.4);
  const dmg = Math.round(DAMAGE[part] * mult);
  c.hp = Math.max(0, c.hp - dmg);
  popups.push({ text: part === 'head' ? `HEADSHOT -${dmg}` : `-${dmg}`, x, y: y - 16 * P, t: 0, color: part === 'head' ? RED : WHITE });
  const a = Math.atan2(arrow.vy, arrow.vx);
  for (let i = 0; i < 26; i++) {                   // spurt of blood, mostly along the arrow's direction
    const s = rand(60, 260), da = rand(-0.9, 0.9) + (Math.random() < 0.35 ? Math.PI : 0);
    blood.push({ x, y, vx: Math.cos(a + da) * s, vy: Math.sin(a + da) * s - rand(40, 140), t: 0,
      size: Math.random() < 0.3 ? 2 : 1, color: Math.random() < 0.4 ? '#991b1b' : RED });
  }
  stuck.push({ c, ox: x - c.x, oy: y - c.y, a });
  if (c.hp === 0) c.dead = 1;
}

function stepArrow(dt) {
  const a = arrow;
  a.vy += G * dt;
  a.x += a.vx * dt; a.y += a.vy * dt;
  const t = a.target, hd = head(t), b = body(t), speed = Math.hypot(a.vx, a.vy);
  if (!t.dead && Math.hypot(a.x - hd.x, a.y - hd.y) < hd.r) return hit(t, 'head', speed, a.x, a.y), true;
  if (!t.dead && a.x > b.x0 && a.x < b.x1 && a.y > b.y0 && a.y < b.y1) return hit(t, 'body', speed, a.x, a.y), true;
  if (a.y >= H - P) { stuck.push({ c: null, ox: a.x, oy: H - P, a: Math.atan2(a.vy, a.vx) }); return true; }
  return a.x < -200 || a.x > W + 200;               // flew off the side
}

// ---------- turn flow ----------
function update(dt) {
  timer += dt;
  if (state === 'flying') {
    for (let i = 0; i < 8 && arrow; i++) if (stepArrow(dt / 8)) { arrow = null; state = 'resolve'; timer = 0; }
  } else if (state === 'resolve' && timer > 0.8) {
    if (ai.dead) { state = 'won'; timer = 0; round++; msg = 'YOU WIN!'; }
    else if (me.dead) { state = 'lost'; msg = 'YOU LOSE'; }
    else if (shooter === ai) state = 'player';
    else { state = 'ai'; timer = 0; aim = { speed: aiAim(), target: ai.aim }; ai.aim = Math.PI / 6; }
  } else if (state === 'ai') {
    // draw the bow over ~1 s, turning towards the chosen angle, then release
    const k = Math.min(timer / 1, 1);
    ai.aim = Math.PI / 6 + (aim.target - Math.PI / 6) * k;
    ai.pull = k * Math.min(aim.speed / VMAX, 1);
    if (timer > 1.15) { ai.aim = aim.target; fire(ai, aim.speed); }
  } else if (state === 'won' && timer > 1.6) {
    newRound(false);
  }
  for (const c of [me, ai]) if (c.dead) c.fall = Math.min(c.fall + dt * 3, 1);
  for (const b of blood) { b.t += dt; b.vy += G * dt; b.x += b.vx * dt; b.y = Math.min(b.y + b.vy * dt, H - P); }
  while (blood.length && blood[0].t > 1.4) blood.shift();
  for (const p of popups) p.t += dt;
  while (popups.length && popups[0].t > 1.3) popups.shift();
}

// ---------- drawing (art pixels) ----------
const q = v => Math.round(v / P);

function drawArrow(x, y, a, len = 18, c = WHITE) {
  const dx = Math.cos(a), dy = Math.sin(a), nx = -dy, ny = dx;
  const tx = x - dx * len, ty = y - dy * len;
  g.line(x - dx * 2, y - dy * 2, tx, ty, c);                                // shaft
  g.line(x, y, x - dx * 4 + nx * 2, y - dy * 4 + ny * 2, c);                // arrowhead
  g.line(x, y, x - dx * 4 - nx * 2, y - dy * 4 - ny * 2, c);
  g.line(x - dx * 4 + nx * 2, y - dy * 4 + ny * 2, x - dx * 4 - nx * 2, y - dy * 4 - ny * 2, c);
  for (const k of [0, 2]) {                                                 // two feathers each side
    const fx = tx + dx * k, fy = ty + dy * k;
    g.line(fx, fy, fx - dx * 3 + nx * 2, fy - dy * 3 + ny * 2, alpha(0.6));
    g.line(fx, fy, fx - dx * 3 - nx * 2, fy - dy * 3 - ny * 2, alpha(0.6));
  }
}

function drawArcher(c, holdingArrow) {
  const fx = q(c.x), fy = q(c.y), s = c.side;
  // dead archers tip over backwards around their feet
  const rot = c.fall * (Math.PI / 2) * -s, cr = Math.cos(rot), sr = Math.sin(rot);
  const T = (x, y) => [fx + x * cr - y * sr, fy + x * sr + y * cr];
  const L = (a, b, col = WHITE) => g.line(...T(...a), ...T(...b), col);
  const X = x => x * s;                                                     // mirror for the archer on the right

  // ledge: a short stone slab with a lighter top edge
  g.line(fx - 13, fy + 1, fx + 13, fy + 1, alpha(0.6));
  g.line(fx - 12, fy + 2, fx + 12, fy + 2, alpha(0.25), 1);

  // legs with feet, slightly apart in a firing stance
  L([0, -14], [X(-5), -1]); L([X(-5), -1], [X(-7), 0]);
  L([0, -14], [X(4), -1]); L([X(4), -1], [X(7), 0]);
  // torso, 2 px wide, and a belt
  L([0, -32], [0, -14]); L([X(-1), -29], [X(-1), -14]);
  L([X(-3), -15], [X(2), -15], alpha(0.6));
  // quiver on the back with three fletchings sticking out
  L([X(-3), -28], [X(-6), -18], alpha(0.7)); L([X(-4), -28], [X(-7), -18], alpha(0.7));
  for (const k of [0, 1, 2]) L([X(-3 + k), -29 - k], [X(-2 + k), -32 - k], alpha(0.6));
  // head: outlined circle, an eye looking at the enemy, and a headband with a tail fluttering behind
  const [hx, hy] = T(0, -37);
  g.circle(hx, hy, 4, WHITE, '#0b1220');
  g.px(...T(X(2), -38), WHITE);
  L([X(-4), -39], [X(4), -39]);
  L([X(-4), -39], [X(-8), -37], alpha(0.7)); L([X(-4), -38], [X(-7), -35], alpha(0.5));

  // bow: a curved limb through the bow hand, string to the nock point
  const d = [Math.cos(c.aim) * s, -Math.sin(c.aim)], n = [-d[1], d[0]];
  const S = [0, -27], Hd = [S[0] + d[0] * 12, S[1] + d[1] * 12];
  const bowPt = t => [Hd[0] + n[0] * 10 * t - d[0] * 4 * t * t, Hd[1] + n[1] * 10 * t - d[1] * 4 * t * t];
  const N = [Hd[0] - d[0] * (4 + c.pull * 12), Hd[1] - d[1] * (4 + c.pull * 12)];
  L(S, [S[0] + d[0] * 5 + n[0], S[1] + d[1] * 5 + n[1]]); L([S[0] + d[0] * 5 + n[0], S[1] + d[1] * 5 + n[1]], Hd);   // bow arm
  for (let i = -6; i < 6; i++) L(bowPt(i / 6), bowPt((i + 1) / 6));
  L(bowPt(-1), [bowPt(-1)[0] - d[0], bowPt(-1)[1] - d[1]]); L(bowPt(1), [bowPt(1)[0] - d[0], bowPt(1)[1] - d[1]]);   // tips
  if (c.dead) return;
  L(bowPt(-1), N, alpha(0.6)); L(N, bowPt(1), alpha(0.6));                  // string
  const elbow = [S[0] - d[0] * 4 - n[0] * 3, S[1] - d[1] * 4 - n[1] * 3];
  L(S, elbow); L(elbow, N);                                                 // drawing arm
  if (holdingArrow) {
    const [nx, ny] = T(...N), [tx, ty] = T(N[0] + d[0] * 19, N[1] + d[1] * 19);
    drawArrow(tx, ty, Math.atan2(ty - ny, tx - nx), 19);
  }

  // health bar: framed, with the fill in white (red when low)
  const bx = fx - 14, by = fy - 50;
  g.line(bx, by, bx + 28, by, alpha(0.5)); g.line(bx, by + 4, bx + 28, by + 4, alpha(0.5));
  g.line(bx, by, bx, by + 4, alpha(0.5)); g.line(bx + 28, by, bx + 28, by + 4, alpha(0.5));
  g.rect(bx + 1, by + 1, Math.round(27 * c.hp / HEALTH), 3, c.hp > 30 ? WHITE : RED);
}

function drawGuide() {
  const p = launchPoint(me), d = dir(me);
  const len = 10 + me.pull * 70;
  g.line(q(p.x), q(p.y), q(p.x) + d.x * len, q(p.y) + d.y * len, alpha(0.8), 2);
  // small arc from the horizontal to the aim line, like a protractor
  const cx = q(p.x), cy = q(p.y);
  for (let k = 0; k <= 16; k++) {
    const t = me.aim * k / 16;
    if (k % 2 === 0) g.px(cx + Math.cos(t) * 9, cy - Math.sin(t) * 9, alpha(0.5));
  }
  const deg = (me.aim * 180 / Math.PI).toFixed(2);
  g.text(`${deg}° ${Math.round(me.pull * 100)}%`, q(me.x), q(me.y) - 62, WHITE, 2, 'center');
}

function button() {
  const w = 360, h = 78;
  return { x: W / 2 - w / 2, y: H / 2 - h / 2 + 50, w, h };
}

function draw() {
  g.clear();
  for (const s of stuck) {
    if (s.c) {
      const c = s.c, rot = c.fall * (Math.PI / 2) * -c.side, cr = Math.cos(rot), sr = Math.sin(rot);
      const x = c.x + s.ox * cr - s.oy * sr, y = c.y + s.ox * sr + s.oy * cr;
      drawArrow(q(x), q(y), s.a + rot, 15);
    } else drawArrow(q(s.ox), q(s.oy), s.a, 15, alpha(0.7));
  }
  drawArcher(me, state === 'player' || state === 'aiming');
  drawArcher(ai, state === 'ai');
  if (arrow) drawArrow(q(arrow.x), q(arrow.y), Math.atan2(arrow.vy, arrow.vx));
  if (state === 'aiming') drawGuide();
  if (state === 'player') g.text('YOUR TURN', q(me.x), q(me.y) - 62, alpha(0.75), 2, 'center');

  for (const b of blood) g.rect(q(b.x), q(b.y), b.size, b.size, b.color);
  for (const p of popups) {
    const a = Math.max(0, 1 - p.t / 1.3);
    g.text(p.text, q(p.x), q(p.y) - Math.round(p.t * 12), p.color === RED ? `rgba(220, 38, 38, ${a.toFixed(2)})` : alpha(a.toFixed(2)), 2, 'center');
  }
  if (msg && (state === 'won' || state === 'lost')) g.text(msg, g.cols / 2, g.rows / 2 - 24, WHITE, 4, 'center');
  if (state === 'lost') {
    const b = button();
    g.rect(q(b.x), q(b.y), q(b.w), q(b.h), '#0b1220');
    g.line(q(b.x), q(b.y), q(b.x + b.w), q(b.y), WHITE); g.line(q(b.x), q(b.y + b.h), q(b.x + b.w), q(b.y + b.h), WHITE);
    g.line(q(b.x), q(b.y), q(b.x), q(b.y + b.h), WHITE); g.line(q(b.x + b.w), q(b.y), q(b.x + b.w), q(b.y + b.h), WHITE);
    g.text('PLAY AGAIN', q(b.x + b.w / 2), q(b.y + b.h / 2) - 7, WHITE, 3, 'center');
  }
}

// ---------- loop: sleeps while waiting for the player ----------
function frame(now) {
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  update(dt);
  draw();
  const idle = (state === 'player' || state === 'lost') && !blood.length && !popups.length && (!me.dead || me.fall === 1);
  if (idle) { running = false; return; }
  requestAnimationFrame(frame);
}

function wake() {
  if (running) return;
  running = true; last = performance.now();
  requestAnimationFrame(frame);
}

// ---------- input ----------
const onButton = (x, y) => { const b = button(); return x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h; };
function setAim(x, y) {
  const s = shoulder(me), vx = s.x - x, vy = s.y - y, d = Math.hypot(vx, vy);
  me.pull = Math.min(d / MAX_PULL, 1);
  if (d > 4) {
    // shoot forward only: aim between straight down and straight up, towards the enemy
    me.aim = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, Math.atan2(-vy, Math.max(vx, 0.001))));
  }
}

onPointer({
  hit: (x, y) => state === 'lost' ? (onButton(x, y) ? 'button' : false)
    : state === 'player' && Math.hypot(x - me.x, y - (me.y - 19 * P)) <= GRAB,
  down(x, y) {
    if (state === 'lost') return newRound(true);
    state = 'aiming'; aim = null;
    setAim(x, y); wake();
  },
  move(x, y) { if (state === 'aiming') { setAim(x, y); wake(); } },
  up() {
    if (state !== 'aiming') return;
    if (me.pull < 0.1) { state = 'player'; me.pull = 0; wake(); return; }   // barely pulled: cancel
    fire(me, me.pull * VMAX);
    wake();
  },
});

addEventListener('resize', layout);
layout();
newRound(true);
