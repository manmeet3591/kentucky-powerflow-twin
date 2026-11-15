/* Kentucky Power Flow Twin: what-if N-1 explorer (static; reads docs/data/*.json) */
(async function () {
  const $ = s => document.querySelector(s);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (x, d = 0) => x == null || Number.isNaN(x) ? "–" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
  const getJSON = async u => { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); };

  const [grid, levels] = await Promise.all([getJSON("data/grid.json"), getJSON("data/levels.json")]);
  const LIM = grid.limits;
  const B = grid.buses, BR = grid.branches;
  const contSet = {};
  grid.contingencies.seen.forEach(i => contSet[i] = "seen");
  grid.contingencies.unseen.forEach(i => contSet[i] = "unseen");
  grid.contingencies.islanding.forEach(i => contSet[i] = "islanding");
  const contList = Object.keys(contSet).map(Number);
  document.querySelectorAll(".nCont").forEach(e => e.textContent = fmt(contList.length));

  // manifest.json says which surrogate outputs exist (updated by surrogate/*.py)
  const manifest = await getJSON("data/manifest.json").catch(() => ({}));
  const gnnAvail = !!manifest.gnn, evalAvail = !!manifest.eval;

  const state = { level: levels.findIndex(l => l.key === "summer_peak"), outage: null, src: "solver", kv: 0 };
  if (state.level < 0) state.level = 0;
  const cache = {};
  async function levelData(src, key) {
    const k = `${src}:${key}`;
    if (!cache[k]) cache[k] = getJSON(src === "gnn" ? `data/gnn_level_${key}.json` : `data/level_${key}.json`);
    return cache[k];
  }

  // ---------- map ----------
  const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25 }).setView([37.6, -85.6], 7.25);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const esri = s => `https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${s}/MapServer/tile/{z}/{y}/{x}`;
  L.tileLayer(esri(dark ? "World_Dark_Gray_Base" : "World_Light_Gray_Base"), {
    attribution: "Basemap &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxZoom: 16,
  }).addTo(map);
  const renderer = L.canvas({ padding: 0.3, tolerance: 6 });
  const lineLayer = L.layerGroup().addTo(map);
  const busLayer = L.layerGroup().addTo(map);

  const RAMP = () => [
    { max: 50, c: css("--muted"), label: "< 50%" },
    { max: 80, c: css("--r1"), label: "50–80" },
    { max: 90, c: css("--r2"), label: "80–90" },
    { max: 100, c: css("--r3"), label: "90–100" },
    { max: 120, c: css("--r4"), label: "100–120" },
    { max: Infinity, c: css("--r5"), label: "> 120%" },
  ];
  const colorFor = L => { if (L == null || Number.isNaN(L)) return css("--axis"); for (const r of RAMP()) if (L <= r.max) return r.c; };
  const widthFor = kv => kv >= 300 ? 3.2 : kv >= 130 ? 1.8 : 1.1;

  const polylines = new Map();
  BR.forEach(b => {
    if (b.role === "equivalent") return;
    const f = B[b.f], t = B[b.t];
    if (Math.hypot(f.lat - t.lat, f.lon - t.lon) < 0.004) return;  // same-substation transformer: not drawable
    const pl = L.polyline([[f.lat, f.lon], [t.lat, t.lon]], { renderer, weight: widthFor(b.kv), opacity: 0.9 });
    pl.on("click", () => { if (contSet[b.i] !== undefined) selectOutage(b.i); });
    pl.bindTooltip(() => tooltipFor(b.i), { className: "tt", sticky: true });
    polylines.set(b.i, pl);
  });

  let cur = { loading: [], vm: [], sum: null, data: null };

  function tooltipFor(i) {
    const b = BR[i], L = cur.loading[i];
    const out = state.outage === i;
    return `<b>${b.name}</b><span class="m">${fmt(b.kv)} kV ${b.xfmr ? "transformer" : "line"} · rating ${fmt(b.rate)} MVA${b.role === "tie" ? " · tie line" : ""}</span><br>` +
      (out ? "<b>Out of service (selected outage)</b>" : `Loading <b style="display:inline">${fmt(L, 1)}%</b>`) +
      (contSet[i] !== undefined && !out ? `<br><span class="m">Click to take this ${b.xfmr ? "transformer" : "line"} out</span>` : "");
  }

  // ---------- controls ----------
  const lv = $("#levels");
  levels.forEach((l, k) => {
    const b = document.createElement("button");
    const pct = Math.round(100 * (l.lgee_mw + l.tva_mw) / (levels.find(x => x.key === "winter_peak").lgee_mw + levels.find(x => x.key === "winter_peak").tva_mw));
    b.innerHTML = `<span>${l.label}</span><span class="pct">${pct}%</span>`;
    b.title = "Combined LGEE + TVA demand relative to the 2023–24 peak hour";
    b.onclick = () => { state.level = k; update(); };
    lv.appendChild(b);
  });
  const dl = $("#branchList");
  contList.sort((a, b) => BR[a].name.localeCompare(BR[b].name)).forEach(i => {
    const o = document.createElement("option");
    o.value = `${BR[i].name} · ${fmt(BR[i].kv)} kV · #${i}`;
    dl.appendChild(o);
  });
  $("#search").addEventListener("change", e => {
    const m = e.target.value.match(/#(\d+)$/);
    if (m && contSet[+m[1]] !== undefined) selectOutage(+m[1]);
  });
  $("#worstBtn").onclick = async () => {
    const d = await levelData(state.src, levels[state.level].key);
    const worst = Object.entries(d.outages).sort((a, b) => (b[1].nOver - a[1].nOver) || ((b[1].maxL || 0) - (a[1].maxL || 0)))[0];
    selectOutage(+worst[0], true);
  };
  document.querySelectorAll("#source button").forEach(b => b.onclick = () => {
    if (b.disabled) return;
    state.src = b.dataset.src;
    document.querySelectorAll("#source button").forEach(x => x.classList.toggle("on", x === b));
    update();
  });
  if (gnnAvail) {
    const gb = $('#source button[data-src="gnn"]');
    gb.disabled = false; gb.title = "";
    $("#sourceHint").textContent = "Switch between precomputed AC power flow and the trained GNN surrogate for the same cases.";
  }
  document.querySelectorAll("#kvf button").forEach(b => b.onclick = () => {
    state.kv = +b.dataset.kv;
    document.querySelectorAll("#kvf button").forEach(x => x.classList.toggle("on", x === b));
    draw();
  });
  $("#showV").onchange = draw;
  $("#showTies").onchange = draw;

  function selectOutage(i, zoom) {
    state.outage = i;
    $("#search").value = "";
    update().then(() => {
      if (zoom || !map.getBounds().contains([B[BR[i].f].lat, B[BR[i].f].lon])) {
        const f = B[BR[i].f], t = B[BR[i].t];
        map.fitBounds(L.latLngBounds([[f.lat, f.lon], [t.lat, t.lon]]).pad(3), { maxZoom: 10 });
      }
    });
  }

  // ---------- state -> arrays ----------
  async function update() {
    document.querySelectorAll("#levels button").forEach((b, k) => b.classList.toggle("on", k === state.level));
    const l = levels[state.level];
    const w = l.weather || {};
    const wv = k => w[k] != null ? w[k] : null;
    const t = new Date(l.time_utc + (l.time_utc.endsWith("Z") || l.time_utc.includes("+") ? "" : "Z"));
    $("#levelInfo").innerHTML =
      `<b>${t.toLocaleString("en-US", { timeZone: "America/New_York", dateStyle: "medium", timeStyle: "short" })} ET</b>` +
      (l.key.startsWith("stress") ? " (scaled; hypothetical)" : "") +
      `<br>LGEE ${fmt(l.lgee_mw)} MW (${fmt(100 * l.lgee, 0)}% of peak) · TVA ${fmt(l.tva_mw)} MW (${fmt(100 * l.tva, 0)}%)` +
      (wv("temperature_2m_louisville") != null ? `<br>ERA5 Louisville ${fmt(wv("temperature_2m_louisville"), 1)} °C, wind ${fmt(wv("wind_speed_10m_louisville"), 1)} m/s · Paducah ${fmt(wv("temperature_2m_paducah"), 1)} °C` : "");

    const d = await levelData(state.src, l.key);
    const loading = d.base.loading.map(x => x == null ? NaN : x);
    const vm = d.base.vm.slice();
    let sum = d.base, o = null;
    if (state.outage != null) {
      o = d.outages[String(state.outage)];
      o.dL.forEach(([i, v]) => loading[i] = v);
      o.dV.forEach(([i, v]) => vm[i] = v == null ? NaN : v);
      loading[state.outage] = NaN;
      sum = o;
    }
    cur = { loading, vm, sum, data: d, o };
    renderPicked();
    draw();
    renderDetail();
    renderRank();
  }

  function renderPicked() {
    const el = $("#picked");
    if (state.outage == null) { el.innerHTML = '<p class="hint" style="margin:0">No outage selected: intact network.</p>'; return; }
    const b = BR[state.outage];
    const set = contSet[state.outage];
    el.innerHTML = `<div class="picked"><span class="nm">${b.name}<br><span class="m" style="color:var(--muted)">${fmt(b.kv)} kV ${b.xfmr ? "transformer" : "line"}, ${fmt(b.rate)} MVA</span></span>` +
      (set === "unseen" ? '<span class="pill unseen" title="Never outaged in surrogate training">held out</span>' : "") +
      (set === "islanding" ? '<span class="pill island" title="This outage islands part of the network">islands</span>' : "") +
      `<button class="link" id="clearOut">Restore</button></div>`;
    $("#clearOut").onclick = () => { state.outage = null; update(); };
  }

  function draw() {
    lineLayer.clearLayers(); busLayer.clearLayers();
    const showTies = $("#showTies").checked;
    polylines.forEach((pl, i) => {
      const b = BR[i];
      if (b.kv < state.kv) return;
      if (b.role === "tie" && !showTies) return;
      const out = state.outage === i;
      pl.setStyle({
        color: out ? css("--ink") : colorFor(cur.loading[i]),
        dashArray: out ? "6 5" : (b.role === "tie" ? "2 4" : null),
        weight: out ? 4 : widthFor(b.kv), opacity: b.role === "tie" ? 0.6 : 0.95,
      });
      lineLayer.addLayer(pl);
      if (cur.loading[i] > 90 || out) pl.bringToFront();
    });
    if ($("#showV").checked) {
      B.forEach((b, i) => {
        if (b.role !== "internal" || b.kv < LIM.MON_KV || b.kv < state.kv) return;
        const v = cur.vm[i];
        if (!(v < LIM.VLO || v > LIM.VHI)) return;
        const low = v < LIM.VLO;
        L.circleMarker([b.lat, b.lon], {
          renderer, radius: 5, weight: 1.5, color: css("--surface"),
          fillColor: low ? css("--b4") : css("--p3"), fillOpacity: 1,
        }).bindTooltip(`<b>${b.name}</b><span class="m">${fmt(b.kv)} kV bus</span><br>Voltage <b style="display:inline">${fmt(v, 3)} pu</b> (${low ? "below 0.95" : "above 1.05"})`, { className: "tt" })
          .addTo(busLayer);
      });
    }
    if (state.outage != null) {
      const b = BR[state.outage], f = B[b.f], t = B[b.t];
      if (Math.hypot(f.lat - t.lat, f.lon - t.lon) < 0.004) {
        L.circleMarker([f.lat, f.lon], { renderer, radius: 9, weight: 3, color: css("--ink"), fill: false, dashArray: "4 3" })
          .bindTooltip(`<b>${b.name}</b>Out of service (transformer inside one substation)`, { className: "tt" }).addTo(busLayer);
      }
    }
    renderLegend();
  }

  function renderLegend() {
    $("#legend").innerHTML =
      `<div class="blk"><div class="t">Branch loading (% of rating)</div><div class="bar">${RAMP().map(r => `<i style="background:${r.c}"></i>`).join("")}</div><div class="ends"><span>&lt;50</span><span>100</span><span>&gt;120</span></div></div>` +
      `<div class="blk"><div class="t">Lines</div><div class="row"><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="${css("--ink")}" stroke-width="3" stroke-dasharray="6 5"/></svg> taken out</div><div class="row"><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="${css("--axis")}" stroke-width="1.5" stroke-dasharray="2 4"/></svg> tie to neighbour state</div></div>` +
      `<div class="blk"><div class="t">Bus voltage</div><div class="row"><span class="sw" style="background:${css("--b4")}"></span>below 0.95 pu</div><div class="row"><span class="sw" style="background:${css("--p3")}"></span>above 1.05 pu</div></div>`;
  }

  function renderDetail() {
    const s = cur.sum, l = levels[state.level];
    const solver = state.src === "solver";
    const banner = solver
      ? `<div class="banner solver"><b>AC power flow solver</b>: full Newton-Raphson results, precomputed (${fmt(1000 * (l.timing?.outage_mean_s || 0.08), 0)} ms per case on one CPU core). ${gnnAvail ? "" : "The GNN surrogate is not trained yet."}</div>`
      : `<div class="banner gnn"><b>GNN surrogate</b>: ${fmt(cur.data.timing?.ms_per_case, 2)} ms per case (${cur.data.timing?.device}). These are predictions, not solver results. Switch to the AC solver to compare.</div>`;
    const lost = s.lost > 0 ? `<div class="kpi bad"><b>${fmt(s.lost)} MW</b><span>load islanded</span></div>` : "";
    const kpis = `<div class="kpis">
      <div class="kpi ${s.nOver ? "bad" : (s.maxL > 90 ? "warn" : "")}"><b>${fmt(s.maxL, 1)}%</b><span>highest branch loading</span></div>
      <div class="kpi ${s.nOver ? "bad" : ""}"><b>${s.nOver}</b><span>overloaded (&gt;100%)</span></div>
      <div class="kpi"><b>${s.nNear}</b><span>at 90–100%</span></div>
      <div class="kpi ${(s.nVlow + s.nVhigh) ? "warn" : ""}"><b>${s.nVlow + s.nVhigh}</b><span>buses outside 0.95–1.05 pu</span></div>
      ${lost}</div>`;
    const title = state.outage == null ? "Intact network" : `Outage: ${BR[state.outage].name}`;
    // branches >= 90%
    const base = cur.data.base.loading;
    const hot = [];
    cur.loading.forEach((L, i) => { if (L > 90) hot.push(i); });
    hot.sort((a, b) => cur.loading[b] - cur.loading[a]);
    const hotHtml = hot.length ? `<ul class="vlist">${hot.slice(0, 40).map(i => `<li data-b="${i}"><span class="nm" title="${BR[i].name}">${BR[i].name}</span><span class="m">${fmt(BR[i].kv)} kV · ${fmt(base[i], 0)}→</span><span class="v" style="color:${cur.loading[i] > 100 ? css("--r4") : css("--ink")}">${fmt(cur.loading[i], 0)}%</span></li>`).join("")}</ul>` : '<p class="empty">None.</p>';
    // biggest increases
    let incHtml = "";
    if (state.outage != null && cur.o) {
      const inc = cur.o.dL.map(([i, v]) => [i, v - (base[i] ?? 0)]).filter(x => x[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 8);
      incHtml = `<h3>Largest loading increases</h3>` + (inc.length ? `<ul class="vlist">${inc.map(([i, d]) => `<li data-b="${i}"><span class="nm" title="${BR[i].name}">${BR[i].name}</span><span class="m">+${fmt(d, 0)} pp</span><span class="v">${fmt(cur.loading[i], 0)}%</span></li>`).join("")}</ul>` : '<p class="empty">No branch moves by more than 1.5 percentage points.</p>');
    }
    const vv = [];
    B.forEach((b, i) => { if (b.role === "internal" && b.kv >= LIM.MON_KV && (cur.vm[i] < LIM.VLO || cur.vm[i] > LIM.VHI)) vv.push(i); });
    vv.sort((a, b) => Math.abs(cur.vm[b] - 1) - Math.abs(cur.vm[a] - 1));
    const vHtml = vv.length ? `<ul class="vlist">${vv.slice(0, 30).map(i => `<li data-bus="${i}"><span class="nm">${B[i].name}</span><span class="m">${fmt(B[i].kv)} kV</span><span class="v" style="color:${cur.vm[i] < 1 ? css("--b4") : css("--p4")}">${fmt(cur.vm[i], 3)}</span></li>`).join("")}</ul>` : '<p class="empty">None.</p>';
    $("#detail").innerHTML = banner + `<h2>${title}</h2>` + kpis +
      (!s.conv ? '<p class="banner" style="border-left:4px solid var(--r4)">Power flow did not converge for this case.</p>' : "") +
      `<h3>Branches at or above 90%</h3>${hotHtml}${incHtml}<h3>Voltage violations (Kentucky, ≥69 kV)</h3>${vHtml}` +
      `<p class="hint">Voltage range at monitored buses: ${fmt(s.minV, 3)}–${fmt(s.maxV, 3)} pu. Emergency band 0.90–1.10 pu: ${s.nVlowE + s.nVhighE} buses outside.</p>`;
    $("#detail").querySelectorAll("li[data-b]").forEach(li => li.onclick = () => {
      const b = BR[+li.dataset.b], f = B[b.f], t = B[b.t];
      map.fitBounds(L.latLngBounds([[f.lat, f.lon], [t.lat, t.lon]]).pad(2), { maxZoom: 11 });
    });
    $("#detail").querySelectorAll("li[data-bus]").forEach(li => li.onclick = () => {
      const b = B[+li.dataset.bus]; map.setView([b.lat, b.lon], 10);
    });
  }

  function renderRank() {
    const d = cur.data, l = levels[state.level];
    $("#rankTitle").textContent = `Worst single outages: ${l.label} (${state.src === "gnn" ? "GNN surrogate" : "AC solver"})`;
    const rows = Object.entries(d.outages).map(([k, v]) => ({ i: +k, ...v }))
      .sort((a, b) => (b.nOver - a.nOver) || ((b.maxL || 0) - (a.maxL || 0))).slice(0, 150);
    $("#rankTable").innerHTML = `<thead><tr><th>Outage</th><th>kV</th><th>Max loading %</th><th>Overloads</th><th>90–100%</th><th>V &lt; 0.95</th><th>V &gt; 1.05</th><th>Min V</th><th>Load islanded MW</th><th>Set</th></tr></thead><tbody>` +
      rows.map(r => `<tr data-i="${r.i}"><td>${BR[r.i].name}</td><td>${fmt(BR[r.i].kv)}</td><td>${fmt(r.maxL, 1)}</td><td>${r.nOver}</td><td>${r.nNear}</td><td>${r.nVlow}</td><td>${r.nVhigh}</td><td>${fmt(r.minV, 3)}</td><td>${r.lost ? fmt(r.lost) : ""}</td><td>${contSet[r.i] === "seen" ? "" : `<span class="pill ${contSet[r.i] === "unseen" ? "unseen" : "island"}">${contSet[r.i] === "unseen" ? "held out" : "islands"}</span>`}</td></tr>`).join("") + "</tbody>";
    $("#rankTable").querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => { showView("map"); selectOutage(+tr.dataset.i, true); });
  }

  // ---------- surrogate tab ----------
  async function renderModel() {
    const el = $("#modelBody");
    if (!evalAvail) {
      el.innerHTML = `<h2>GNN surrogate: not trained yet</h2>
        <p>The surrogate code, data pipeline and TACC job scripts are in the repository. We ran only tiny CPU sanity checks here (48 samples per split, 3 epochs, a 57k-parameter model), which test that the code runs end to end. They say nothing about accuracy. Training runs on TACC:</p>
        <ol><li><code>jobs/01_generate.slurm</code>: 48k AC power flow samples on one CPU node (2023 hours for training, 2024 hours for testing).</li>
        <li><code>jobs/02_train.slurm</code>: GNN training on an A100 (Lonestar6) or GH200 (Vista).</li>
        <li><code>jobs/03_evaluate.slurm</code>: held-out evaluation against the AC solver and a DC power flow baseline, plus surrogate predictions for every case on this page.</li></ol>
        <p>Once those files are copied into <code>docs/data/</code>, this tab will show the accuracy and timing numbers below for seen and never-seen outages. The <b>GNN surrogate</b> button on the What-if tab will then switch on, so its answers can be compared case by case with the solver.</p>
        <div class="metric-grid">${["Vm error (pu)", "Loading error (pp)", "Overload recall", "Overload precision", "Voltage-violation recall", "ms per scenario"].map(t => `<div class="kpi"><b>–</b><span>${t}</span></div>`).join("")}</div>
        <p class="hint">What would count as useful: overload recall close to 1 on never-seen outages, loading error clearly below the DC power flow baseline, and at least 100× faster than the AC solve (about 100 ms per case on one CPU core).</p>`;
      return;
    }
    const ev = await getJSON("data/surrogate_eval.json");
    const row = (lab, k, d = 3, pct = false) => `<tr><td>${lab}</td>${["test_seen", "test_unseen"].map(s => {
      const g = ev[s].gnn[k], c = ev[s].dc_pf[k];
      const f = v => v == null ? "–" : pct ? fmt(100 * v, 1) + "%" : fmt(v, d);
      return `<td>${f(g)}</td><td>${f(c)}</td>`;
    }).join("")}</tr>`;
    el.innerHTML = `<h2>GNN surrogate vs AC solver (held-out 2024 hours)</h2>
      <p>Checkpoint <code>${ev.ckpt.split("/").slice(-2).join("/")}</code>, evaluated on ${ev.device}. "Seen" outages were also outaged in training (at different hours). "Never seen" outages are the 130 held-out lines, never taken out in training.</p>
      <div class="tablewrap"><table><thead><tr><th>Metric</th><th>GNN · seen</th><th>DC PF · seen</th><th>GNN · never seen</th><th>DC PF · never seen</th></tr></thead><tbody>
      ${row("Vm MAE (pu)", "vm_mae_pu", 4)}${row("Va MAE (deg)", "va_mae_deg", 2)}
      ${row("Loading MAE (pp)", "loading_mae_pp", 2)}${row("Loading 95th pct error (pp)", "loading_p95_err_pp", 1)}
      ${row("Overload recall", "overload_recall", 0, true)}${row("Overload precision", "overload_precision", 0, true)}
      ${row("Voltage-violation recall", "vviol_recall", 0, true)}${row("ms per scenario (batched)", "ms_per_scenario_batched", 2)}
      </tbody></table></div>
      <p class="hint">Samples: ${ev.test_seen.n} seen, ${ev.test_unseen.n} never seen. The DC baseline estimates loading as |P|/rating, ignoring reactive flow and voltages.</p>`;
  }

  // ---------- tabs ----------
  function showView(v) {
    document.querySelectorAll(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.view === v));
    $("#view-map").classList.toggle("hidden", v !== "map");
    $("#detail").classList.toggle("hidden", v !== "map");
    ["rank", "model", "method"].forEach(k => $(`#view-${k}`).classList.toggle("hidden", v !== k));
    if (v === "map") setTimeout(() => map.invalidateSize(), 0);
    if (v === "model") renderModel();
  }
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => showView(b.dataset.view));

  await update();
})();
