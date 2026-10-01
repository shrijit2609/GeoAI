(() => {
  const state = {
    api: "http://localhost:8000",
    map: null,
    baseLayer: null,
    showBase: true,
    layers: new Map(),
    sources: [],
    lastJob: null,
  };

  const byId = (id) => document.getElementById(id);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const apiUrl = (path) => `${state.api.replace(/\/$/, "")}${path}`;

  async function request(path, options = {}) {
    const response = await fetch(apiUrl(path), options);
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || response.statusText;
      const missing = payload?.missing?.length ? ` Missing: ${payload.missing.join(", ")}.` : "";
      throw new Error(`${response.status}: ${detail}${missing}`);
    }
    return payload;
  }

  function setMessage(id, text, kind = "") {
    const target = byId(id);
    target.textContent = text;
    target.className = `result-box${kind ? ` ${kind}` : ""}`;
  }

  function initializeMap() {
    if (!window.L) {
      byId("map-placeholder").textContent = "Map library could not load. Check network access to unpkg.com.";
      return;
    }
    state.map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([26.8, 80.9], 6);
    state.baseLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(state.map);
  }

  async function checkHealth() {
    try {
      const health = await request("/api/health");
      byId("health-text").textContent = `Backend online · ${health.device.toUpperCase()} · ${health.version}`;
      byId("side-health").textContent = "API connected";
    } catch (error) {
      byId("health-text").textContent = `Backend unavailable · ${error.message}`;
      byId("side-health").textContent = "API unavailable";
    }
  }

  async function loadDashboardSummary() {
    try {
      const summary = await request("/api/dashboard/summary");
      byId("metric-sources").textContent = String(summary.registered_sources ?? 0);
      byId("metric-features").textContent = String(summary.total_features ?? 0);
      byId("metric-models").textContent = `${summary.models_ready ?? 0}/${summary.model_health?.total ?? 0} ready`;
      return summary;
    } catch (error) {
      console.warn("Dashboard summary unavailable:", error.message);
      return null;
    }
  }

  async function loadModels() {
    const container = byId("model-list");
    container.innerHTML = '<div class="empty-state">Probing configured artifacts…</div>';
    try {
      const report = await request("/api/models/readiness");
      const models = Object.values(report.models || {});
      byId("metric-models").textContent = `${report.ready}/${report.total} ready`;
      const statusLabels = {
        ready: "Ready",
        ready_for_test: "Artifact present / probe pending",
        artifact_present_but_preprocessing_blocked: "Artifact present / preprocessing blocked",
        artifact_present_but_inference_blocked: "Artifact present / inference blocked",
        training_required: "Training required",
        missing_artifact: "Missing artifact",
        error: "Error",
        load_error: "Error",
        unsupported_runtime: "Error",
      };
      container.innerHTML = models.map((model) => `
        <article class="model-card">
          <div class="model-card-head"><strong>${escapeHtml(model.key.replaceAll("_", " "))}</strong><span class="model-status ${escapeHtml(model.status)}">${escapeHtml(statusLabels[model.status] || model.status.replaceAll("_", " "))}</span></div>
          <div class="model-error">${escapeHtml(model.readiness_reason || model.error || "Readiness reason unavailable.")}</div>
        </article>`).join("");
    } catch (error) {
      byId("metric-models").textContent = "Unavailable";
      container.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    }
  }

  async function loadLayer(source, visible = true) {
    if (!state.map || state.layers.has(source.source_id)) return;
    try {
      const collection = await request(`/api/data/layers/${encodeURIComponent(source.source_id)}`);
      const layer = L.geoJSON(collection, {
        style: () => ({ color: "#155a41", weight: 2, fillColor: "#3a9a6c", fillOpacity: 0.24 }),
        pointToLayer: (_feature, latlng) => L.circleMarker(latlng, { radius: 5, color: "#155a41", fillColor: "#3a9a6c", fillOpacity: 0.8 }),
        onEachFeature: (feature, featureLayer) => {
          const properties = feature.properties || {};
          const rows = Object.entries(properties).filter(([, value]) => value !== null && typeof value !== "object").slice(0, 24);
          featureLayer.bindPopup(`<div class="popup-title">${escapeHtml(feature.id || source.source_name || "Feature")}</div>${rows.map(([key, value]) => `<div class="popup-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}`);
        },
      });
      if (visible) layer.addTo(state.map);
      state.layers.set(source.source_id, layer);
      updateMapSummary();
      if (visible && layer.getBounds().isValid()) state.map.fitBounds(layer.getBounds(), { padding: [24, 24], maxZoom: 16 });
    } catch (error) {
      setMessage("upload-message", `Layer could not be drawn: ${error.message}`, "error");
    }
  }

  function updateMapSummary() {
    let count = 0;
    for (const layer of state.layers.values()) {
      if (state.map?.hasLayer(layer)) layer.eachLayer(() => { count += 1; });
    }
    byId("map-count").textContent = `${count} visible feature${count === 1 ? "" : "s"}`;
  }

  function renderSources() {
    const sourceList = byId("source-list");
    const choices = byId("harmonize-sources");
    byId("metric-sources").textContent = String(state.sources.length);
    byId("metric-features").textContent = String(state.sources.reduce((sum, source) => sum + (source.valid_records || 0), 0));
    if (!state.sources.length) {
      sourceList.innerHTML = '<div class="empty-state">No uploaded sources yet.</div>';
      choices.innerHTML = '<span class="muted">Upload sources to select them.</span>';
      byId("run-harmonize").disabled = true;
      return;
    }
    sourceList.innerHTML = state.sources.map((source) => `
      <div class="source-item">
        <input type="checkbox" class="layer-toggle" data-layer="${escapeHtml(source.source_id)}" checked aria-label="Toggle ${escapeHtml(source.source_name)}">
        <div><strong>${escapeHtml(source.source_name)}</strong><small>${escapeHtml(source.source_type)} · ${source.valid_records || 0} valid · ${escapeHtml(source.detected_crs || "CRS unknown")}</small><div class="source-tools"><a href="${escapeHtml(apiUrl(`/api/data/${encodeURIComponent(source.source_id)}/export?format=geojson`))}" target="_blank" rel="noreferrer">GeoJSON</a><a href="${escapeHtml(apiUrl(`/api/data/${encodeURIComponent(source.source_id)}/export?format=csv`))}" target="_blank" rel="noreferrer">CSV</a></div></div>
        <span class="source-count">${source.total_records ?? 0}</span>
      </div>`).join("");
    choices.innerHTML = state.sources.map((source) => `<label><input type="checkbox" value="${escapeHtml(source.source_id)}">${escapeHtml(source.source_name)}</label>`).join("");
    byId("run-harmonize").disabled = state.sources.length < 1;
    document.querySelectorAll(".layer-toggle").forEach((toggle) => {
      toggle.addEventListener("change", () => {
        const layer = state.layers.get(toggle.dataset.layer);
        if (!layer || !state.map) return;
        if (toggle.checked) layer.addTo(state.map); else state.map.removeLayer(layer);
        updateMapSummary();
      });
    });
  }

  async function refreshSources() {
    try {
      const result = await request("/api/data/sources");
      state.sources = result.sources || [];
      renderSources();
      for (const source of state.sources) await loadLayer(source, true);
    } catch (error) {
      setMessage("upload-message", error.message, "error");
    }
  }

  byId("api-url").addEventListener("change", () => {
    state.api = byId("api-url").value.trim().replace(/\/$/, "");
  });
  byId("connect-btn").addEventListener("click", async () => {
    state.api = byId("api-url").value.trim().replace(/\/$/, "");
    await Promise.all([checkHealth(), loadDashboardSummary(), loadModels(), refreshSources()]);
  });
  byId("refresh-sources").addEventListener("click", refreshSources);
  byId("refresh-models").addEventListener("click", loadModels);

  byId("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = byId("source-file").files[0];
    if (!file) return;
    const form = new FormData();
    form.set("file", file);
    form.set("source_name", byId("source-name").value.trim() || file.name.replace(/\.[^.]+$/, ""));
    form.set("source_type", byId("source-type").value);
    form.set("project_crs", byId("target-crs").value.trim());
    const sourceCrs = byId("source-crs").value.trim();
    if (sourceCrs) form.set("source_crs", sourceCrs);
    byId("upload-message").className = "inline-message";
    byId("upload-message").textContent = "Validating and registering source…";
    try {
      const result = await request("/api/data/upload", { method: "POST", body: form });
      byId("upload-message").textContent = `${result.valid_records} valid · ${result.invalid_records} invalid · ${result.detected_crs || "CRS unknown"}`;
      byId("upload-form").reset();
      await refreshSources();
      if (result.warnings?.length) byId("upload-message").textContent += ` · ${result.warnings[0]}`;
    } catch (error) {
      byId("upload-message").textContent = error.message;
      byId("upload-message").classList.add("error");
    }
  });

  byId("fit-map").addEventListener("click", () => {
    if (!state.map) return;
    const visible = [...state.layers.values()].filter((layer) => state.map.hasLayer(layer));
    const bounds = L.featureGroup(visible).getBounds();
    if (bounds.isValid()) state.map.fitBounds(bounds, { padding: [24, 24], maxZoom: 16 });
  });
  byId("base-toggle").addEventListener("click", () => {
    if (!state.map || !state.baseLayer) return;
    state.showBase = !state.showBase;
    if (state.showBase) state.baseLayer.addTo(state.map); else state.map.removeLayer(state.baseLayer);
    byId("base-toggle").textContent = state.showBase ? "Basemap on" : "Basemap off";
  });

  byId("ulpin-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    setMessage("ulpin-result", "Searching registered local sources…");
    try {
      const result = await request(`/api/ulpin/${encodeURIComponent(byId("ulpin-value").value.trim())}`);
      setMessage("ulpin-result", JSON.stringify(result, null, 2), result.status === "local_match" ? "success" : "");
    } catch (error) { setMessage("ulpin-result", error.message, "error"); }
  });

  byId("run-harmonize").addEventListener("click", async () => {
    const sourceIds = [...document.querySelectorAll("#harmonize-sources input:checked")].map((input) => input.value);
    if (!sourceIds.length) { setMessage("harmonize-result", "Select at least one registered source.", "error"); return; }
    setMessage("harmonize-result", "Running deterministic identifier matching…");
    try {
      const job = await request("/api/harmonize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source_ids: sourceIds }) });
      state.lastJob = job;
      setMessage("harmonize-result", `${job.feature_count} output feature(s) · ${job.conflict_count} unresolved conflict(s) · matching: ${job.matching_method}\n${job.limitations.join("\n")}`, "success");
      byId("export-harmonized").disabled = false;
      if (job.feature_collection) drawHarmonized(job.feature_collection);
    } catch (error) { setMessage("harmonize-result", error.message, "error"); }
  });

  function drawHarmonized(collection) {
    const existing = state.layers.get("__harmonized__");
    if (existing && state.map?.hasLayer(existing)) state.map.removeLayer(existing);
    if (!window.L || !state.map) return;
    const layer = L.geoJSON(collection, { style: { color: "#bc6d24", weight: 3, fillColor: "#e6a14d", fillOpacity: 0.24 } }).addTo(state.map);
    state.layers.set("__harmonized__", layer);
    if (layer.getBounds().isValid()) state.map.fitBounds(layer.getBounds(), { padding: [24, 24], maxZoom: 16 });
    updateMapSummary();
  }

  byId("export-harmonized").addEventListener("click", () => {
    if (!state.lastJob?.feature_collection) return;
    const blob = new Blob([JSON.stringify(state.lastJob.feature_collection, null, 2)], { type: "application/geo+json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${state.lastJob.job_id}-harmonized.geojson`;
    anchor.click();
    URL.revokeObjectURL(url);
  });

  byId("entity-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    let recordA; let recordB;
    try { recordA = JSON.parse(byId("record-a").value); recordB = JSON.parse(byId("record-b").value); }
    catch { setMessage("inference-result", "Both records must be valid JSON objects.", "error"); return; }
    setMessage("inference-result", "Calling the trained entity resolver…");
    try {
      const result = await request("/api/models/entity_resolver/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ record_a: recordA, record_b: recordB }) });
      setMessage("inference-result", JSON.stringify(result, null, 2), "success");
    } catch (error) { setMessage("inference-result", error.message, "error"); }
  });

  initializeMap();
  state.api = window.location.origin;
  byId("api-url").value = state.api;
  Promise.all([checkHealth(), loadDashboardSummary(), loadModels(), refreshSources()]);
})();