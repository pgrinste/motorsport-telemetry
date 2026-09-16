/* SUZUKA // Telemetry & Predictive Line — CesiumJS frontend.
   Replay mode: CZML (driver + ideal ghost) + static bundle.json.
   Live mode: /ws/live streams lap 2 at native rate over WebSocket. */
"use strict";

const EPOCH = "2026-09-12T00:00:00Z";
let BUNDLE, viewer, driverEnt, ghostEnt, liveEnt = null;
let deltaPrim = null, cornerLabels = null;
let selectedLap = "1", tracking = false, ws = null, liveSpeed = 0;

const $ = (id) => document.getElementById(id);

/* ---------- helpers ---------- */
function diverging(t) { // t in [-1,1]: - ahead(green), + behind(red)
  const base = [38, 48, 62];
  const tgt = t >= 0 ? [231, 76, 60] : [46, 204, 113];
  const k = Math.min(1, Math.abs(t));
  return base.map((b, i) => Math.round(b + (tgt[i] - b) * k));
}
function interp(arr, x, xs) { // linear interp of arr over uniform-ish xs
  if (x <= xs[0]) return arr[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return arr[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const f = (x - xs[lo]) / Math.max(1e-9, xs[hi] - xs[lo]);
  return arr[lo] + f * (arr[hi] - arr[lo]);
}

/* ---------- delta heatmap ribbon (custom geometry) ---------- */
function buildDeltaRibbon(lapKey) {
  const tr = BUNDLE.track;
  const deltas = BUNDLE.delta.laps[lapKey].cum_delta_ms;
  const n = tr.lat.length, m = deltas.length;
  let maxAbs = 1;
  for (const v of deltas) maxAbs = Math.max(maxAbs, Math.abs(v));

  const pos = new Float64Array(n * 2 * 3);
  const col = new Uint8Array(n * 2 * 4);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const la = tr.lat[i], lo = tr.lon[i], el = tr.elev[i] + 8;
    const i1 = (i + 1) % n, i0 = (i - 1 + n) % n;
    let dx = (tr.lon[i1] - tr.lon[i0]) * Math.cos(la * Math.PI / 180) * 111320;
    let dy = (tr.lat[i1] - tr.lat[i0]) * 110540;
    const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
    const nx = -dy, ny = dx; // left normal
    for (let side = 0; side < 2; side++) {
      const sgn = side === 0 ? 1 : -1, w = 2.6 * sgn;
      const vlat = la + (ny * w) / 110540;
      const vlon = lo + (nx * w) / (111320 * Math.cos(la * Math.PI / 180));
      const p = Cesium.Cartesian3.fromDegrees(vlon, vlat, el);
      pts.push(p);
      const o = (i * 2 + side) * 3;
      pos[o] = p.x; pos[o + 1] = p.y; pos[o + 2] = p.z;
      const di = Math.round(i * (m - 1) / (n - 1));
      const t = Math.max(-1, Math.min(1, deltas[di] / maxAbs));
      const [r, g, b] = diverging(t);
      const co = (i * 2 + side) * 4;
      col[co] = r; col[co + 1] = g; col[co + 2] = b; col[co + 3] = 205;
    }
  }
  const idx = [];
  for (let i = 0; i < n; i++) {
    const a = i * 2, b = a + 1, c = ((i + 1) % n) * 2, d = c + 1;
    idx.push(a, c, b, b, c, d);   // both windings -> visible from any angle
  }
  const geometry = Cesium.Geometry.create({
    attributes: {
      position: new Cesium.GeometryAttribute({ componentType: Cesium.ComponentType.DOUBLE, componentsPerAttribute: 3, values: pos }),
      color: new Cesium.GeometryAttribute({ componentType: Cesium.ComponentType.UNSIGNED_BYTE, componentsPerAttribute: 4, values: col })
    },
    indices: idx,
    boundingSphere: Cesium.BoundingSphere.fromPoints(pts)
  });
  return new Cesium.Primitive({ geometry, allowPicking: false });
}

/* ---------- speed profile chart (canvas 2D) ---------- */
const chart = () => $("chart");
function drawChart(cursorDistM) {
  const cv = chart(), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const Lkm = 7.3, vmax = 340;
  const X = (s) => 46 + (s / 1000) / Lkm * (W - 58);
  const Y = (v) => H - 26 - (v / vmax) * (H - 40);

  ctx.strokeStyle = "#22304f"; ctx.lineWidth = 1;
  for (let v = 0; v <= vmax; v += 85) {
    ctx.beginPath(); ctx.moveTo(40, Y(v)); ctx.lineTo(W - 8, Y(v)); ctx.stroke();
    ctx.fillStyle = "#7c8db0"; ctx.font = "16px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(String(Math.round(v)), 36, Y(v) + 5);
  }
  for (let s = 0; s <= Lkm * 1000; s += 1000) {
    ctx.beginPath(); ctx.moveTo(X(s), 8); ctx.lineTo(X(s), H - 24); ctx.stroke();
    ctx.fillStyle = "#7c8db0"; ctx.textAlign = "center";
    ctx.fillText((s / 1000).toFixed(0) + " km", X(s), H - 8);
  }

  const mc = BUNDLE.mc;
  // MC uncertainty band
  ctx.beginPath();
  for (let i = 0; i < mc.s.length; i++) { const x = X(mc.s[i]), y = Y(mc.v_p90[i] * 3.6); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  for (let i = mc.s.length - 1; i >= 0; i--) ctx.lineTo(X(mc.s[i]), Y(mc.v_p10[i] * 3.6));
  ctx.closePath(); ctx.fillStyle = "rgba(77,163,255,.18)"; ctx.fill();
  const line = (xs, ys, color, width, dash) => {
    ctx.beginPath(); ctx.setLineDash(dash || []);
    for (let i = 0; i < xs.length; i++) { const x = X(xs[i]), y = Y(ys[i] * 3.6); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.setLineDash([]);
  };
  line(mc.s, mc.v_p50, "rgba(77,163,255,.8)", 2.5);          // MC median
  line(BUNDLE.ideal.s, BUNDLE.ideal.v_ms, "#ffffff", 1.5, [6, 5]); // deterministic ideal
  const dv = BUNDLE.driver_v_s[selectedLap];                  // driver (selected lap)
  if (dv && dv.length === BUNDLE.delta.s.length) line(BUNDLE.delta.s, dv, "#ffd23c", 2.5);

  if (cursorDistM != null) {
    const x = X(cursorDistM % Lkm * 1000);
    ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, H - 24);
    ctx.strokeStyle = "rgba(255,255,255,.65)"; ctx.lineWidth = 2; ctx.stroke();
  }

  // legend
  ctx.font = "15px sans-serif"; ctx.textAlign = "left";
  const leg = [["#ffd23c", "driver (lap " + selectedLap + ")"], ["#ffffff", "ideal line"], ["rgba(77,163,255,.9)", "MC p50 ± band"]];
  let lx = 54;
  for (const [c, t] of leg) { ctx.fillStyle = c; ctx.fillRect(lx, 12, 18, 4); ctx.fillStyle = "#dbe4f5"; ctx.fillText(t, lx + 24, 19); lx += 24 + ctx.measureText(t).width + 26; }
}

/* ---------- HUD ---------- */
function lapBounds() {
  const lts = [BUNDLE.delta.laps["1"].lap_time_s, BUNDLE.delta.laps["2"].lap_time_s, BUNDLE.delta.laps["3"].lap_time_s];
  return [0, lts[0], lts[0] + lts[1]];
}
function updateHud() {
  if (!BUNDLE) return;
  const t = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, Cesium.JulianDate.fromIso8601(EPOCH));
  const b = lapBounds();
  let lap = 3, tLocal = Math.max(0, t - b[2]);
  if (t < b[1]) { lap = 1; tLocal = t; } else if (t < b[2]) { lap = 2; tLocal = t - b[1]; }
  const gs = BUNDLE.gap_series[String(lap)];
  let speed, gap, dist;
  if (ws && ws.readyState === 1) { // live mode
    speed = liveSpeed; gap = null; dist = null;
  } else {
    speed = interp(gs.speed_kmh, tLocal, gs.t);
    gap = interp(gs.gap_ms, tLocal, gs.t);
    dist = interp(gs.dist_m, tLocal, gs.t);
  }
  $("hSpeed").textContent = speed != null ? Math.round(speed) : "—";
  $("hLap").textContent = ws && ws.readyState === 1 ? "2 (live)" : lap + " / 3";
  const gEl = $("hGap");
  if (gap == null) { gEl.textContent = "streaming…"; gEl.className = ""; }
  else {
    gEl.textContent = (gap >= 0 ? "+" : "") + Math.round(gap) + " ms";
    gEl.id = gap >= 0 ? "hGap" : "hGap"; // keep id; color via style
    gEl.style.color = gap >= 0 ? "#e74c3c" : "#2ecc71";
  }
  const mm = Math.floor(t / 60), ss = (t % 60).toFixed(1).padStart(4, "0");
  $("hTime").textContent = mm + ":" + ss;
  drawChart(dist);
}

/* ---------- init ---------- */
async function init() {
  BUNDLE = await (await fetch("bundle.json")).json();

  viewer = new Cesium.Viewer("cesiumContainer", {
    baseLayer: new Cesium.ImageryLayer(new Cesium.OpenStreetMapImageryProvider({ url: "https://tile.openstreetmap.org/" })),
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    timeline: true, animation: true, homeButton: false, sceneModePicker: false,
    navigationHelpButton: false, geocoder: false, infoBox: false,
    selectionIndicator: false, baseLayerPicker: false, fullscreenButton: false
  });
  viewer.scene.skyBox.show = false;
  viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#0b1220");
  viewer.clock.multiplier = 4; // match the slider default
  const target = Cesium.Cartesian3.fromDegrees(136.5405, 34.8425);
  viewer.camera.lookAt(target, new Cesium.HeadingPitchRange(Cesium.Math.toRadians(-90), -1.02, 2700));

  // Track centerline (real elevation).
  const cartos = BUNDLE.track.lat.map((la, i) =>
    Cesium.Cartographic.fromDegrees(BUNDLE.track.lon[i], la, BUNDLE.track.elev[i] + 2));
  viewer.entities.add({ id: "centerline", polyline: { positions: cartos.map((c) => Cesium.Cartesian3.fromCartographic(c)), width: 4, material: Cesium.Color.GRAY.withAlpha(0.8) } });

  // Corner labels.
  cornerLabels = new Cesium.LabelCollection();
  for (const c of BUNDLE.corners) {
    cornerLabels.add({ position: Cesium.Cartesian3.fromDegrees(c.lon, c.lat, c.elev + 14), text: c.name.replace(" / First Turn", ""), font: "12px 'Segoe UI', sans-serif", fillColor: Cesium.Color.WHITE, showBackground: true, backgroundColor: Cesium.Color.fromCssColorString("#0d1424").withAlpha(0.75), pixelOffset: new Cesium.Cartesian2(0, -8) });
  }
  viewer.scene.primitives.add(cornerLabels);

  // Delta ribbon (selected lap).
  deltaPrim = buildDeltaRibbon(selectedLap);
  viewer.scene.primitives.add(deltaPrim);

  // CZML: driver + ideal ghost.
  const ds = await Cesium.CzmlDataSource.load("czml/telemetry.czml");
  viewer.dataSources.add(ds);
  driverEnt = ds.entities.getById("driver");
  ghostEnt = ds.entities.getById("ideal_ghost");

  drawChart(null);
  viewer.scene.postRender.addEventListener(updateHud);
}

/* ---------- controls ---------- */
function wireControls() {
  $("lapBtns").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    selectedLap = b.dataset.lap;
    for (const x of $("lapBtns").children) x.classList.toggle("on", x === b);
    viewer.scene.primitives.remove(deltaPrim);
    deltaPrim = buildDeltaRibbon(selectedLap);
    viewer.scene.primitives.add(deltaPrim);
    drawChart(null);
  });
  $("lyrDelta").addEventListener("change", (e) => { if (deltaPrim) deltaPrim.show = e.target.checked; });
  $("lyrTrail").addEventListener("change", (e) => { if (driverEnt && driverEnt.path) driverEnt.path.show = e.target.checked; });
  $("lyrGhost").addEventListener("change", (e) => { if (ghostEnt) ghostEnt.show = e.target.checked; });
  $("lyrCorners").addEventListener("change", (e) => { cornerLabels.show = e.target.checked; });

  $("trackBtn").addEventListener("click", () => {
    tracking = !tracking;
    $("trackBtn").classList.toggle("on", tracking);
    viewer.trackedEntity = tracking ? (ws && ws.readyState === 1 ? liveEnt : driverEnt) : null;
  });
  $("speedSlider").addEventListener("input", (e) => {
    viewer.clock.multiplier = Number(e.target.value);
    $("speedVal").textContent = "×" + e.target.value;
  });

  $("liveBtn").addEventListener("click", () => {
    if (ws && ws.readyState === 1) { ws.close(); return; }
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(proto + location.host + "/ws/live");
    $("liveBtn").classList.add("on"); $("liveDot").classList.add("on");
    viewer.clock.shouldAnimate = false;
    liveEnt = viewer.entities.add({ id: "live_car", position: Cesium.Cartesian3.fromDegrees(136.5405, 34.8425, 30), point: { pixelSize: 10, color: Cesium.Color.fromCssColorString("#2ecc71"), outlineColor: Cesium.Color.WHITE, outlineWidth: 2 }, label: { text: "LIVE", font: "12px sans-serif", showBackground: true, backgroundColor: Cesium.Color.fromCssColorString("#0d3320").withAlpha(0.8), pixelOffset: new Cesium.Cartesian2(0, -18) } });
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      liveSpeed = m.speed_kmh;
      if (!liveEnt.position || !livePosSet) { livePosSet = true; }
      liveEnt.position.setValue(Cesium.Cartesian3.fromDegrees(m.lon, m.lat, m.alt + 1.5));
    };
    ws.onclose = () => {
      $("liveBtn").classList.remove("on"); $("liveDot").classList.remove("on");
      if (tracking) viewer.trackedEntity = driverEnt;
      if (liveEnt) viewer.entities.remove(liveEnt);
      liveEnt = null; ws = null;
      viewer.clock.shouldAnimate = true;
    };
  });
}

let livePosSet = false;
init().then(wireControls).catch((err) => { console.error(err); document.body.innerHTML += '<div style="position:fixed;top:40%;left:50%;transform:translateX(-50%);color:#e74c3c;font-size:18px">Failed to init: ' + err.message + "</div>"; });
