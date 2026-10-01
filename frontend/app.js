(() => {
  const state = {
    api: "http://localhost:8000",
    map: null,
    baseLayer: null,
    showBase: true,
    layers: new Map(),
    sources: [],
    records: [],
    pendingSource: null,
    lastJob: null,
    previewUrls: new Map(),
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
      byId("metric-harmonized").textContent = String(summary.harmonized_parcels ?? 0);
      byId("metric-conflicts").textContent = String(summary.conflicts ?? 0);
      byId("metric-anomalies").textContent = String(summary.anomalies ?? 0);
      byId("metric-confidence").textContent = summary.average_confidence ? `${(summary.average_confidence * 100).toFixed(1)}%` : "—";
      byId("metric-models").textContent = `${summary.models_ready ?? 0}/${summary.model_health?.total ?? 0} ready`;
      byId("processing-jobs").textContent = `${summary.processing_jobs ?? 0} processing jobs`;
      byId("recent-activity").textContent = summary.recent_activity?.length ? `Latest job · ${summary.recent_activity.join(", ")}` : "No harmonization jobs yet";
      byId("source-health").textContent = summary.source_health?.healthy ? `${summary.source_health.healthy} healthy source(s)` : "No sources registered";
      byId("data-quality").textContent = `${summary.source_health?.warning_count ?? 0} source warning(s)`;
      byId("model-health").textContent = `${summary.model_health?.ready ?? 0} of ${summary.model_health?.total ?? 0} models ready`;
      byId("conflict-summary").textContent = `${summary.conflicts ?? 0} conflict(s) · ${summary.anomalies ?? 0} anomaly review(s)`;
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
        invalid: "Invalid artifact",
        unsupported_runtime: "Error",
      };
      container.innerHTML = models.map((model) => `
        <article class="model-card">
          <div class="model-card-head"><strong>${escapeHtml(model.name || model.key.replaceAll("_", " "))}</strong><span class="model-status ${escapeHtml(model.status)}">${escapeHtml(statusLabels[model.status] || model.status.replaceAll("_", " "))}</span></div>
          <div class="model-purpose">${escapeHtml(model.purpose || model.task || "")}</div>
          <div class="model-meta">${escapeHtml(model.architecture_name || model.architecture || "")}${model.dataset ? ` · ${escapeHtml(model.dataset)}` : ""}</div>
          <div class="model-meta">Artifact: ${escapeHtml(model.artifact || model.artifacts?.find((item) => item.required)?.name || "not installed")}</div>
          <div class="model-meta">Live inference: ${model.live_inference_available ? "available" : "unavailable"}</div>
          <div class="model-meta">Last validation: ${escapeHtml(model.last_validation || "Not recorded")}</div>
          <div class="model-error">${escapeHtml(model.readiness_reason || model.error || "Readiness reason unavailable.")}</div>
          <div class="model-metrics">${escapeHtml(Object.entries(model.benchmark_metrics || {}).filter(([key]) => key !== "source").map(([key, value]) => `${key}: ${typeof value === "number" ? value.toFixed(3) : value}`).join(" · ") || "No benchmark metrics recorded")}</div>
          ${model.benchmark_metrics?.source ? `<div class="model-meta">${escapeHtml(model.benchmark_metrics.source)}</div>` : ""}
        </article>`).join("");
      const modelByKey = Object.fromEntries(models.map((model) => [model.key || model.model_key, model]));
      for (const [key, id] of [["building_extractor", "building-status"], ["change_detector", "change-status"]]) {
        const status = modelByKey[key]?.status || "missing_artifact";
        byId(id).className = `model-status ${escapeHtml(status)}`;
        byId(id).textContent = statusLabels[status] || status.replaceAll("_", " ");
      }
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
        style: () => ({ color: source.source_type === "building_footprint" ? "#bc6d24" : "#155a41", weight: 2, fillColor: source.source_type === "building_footprint" ? "#e6a14d" : "#3a9a6c", fillOpacity: Number(byId("layer-opacity").value) / 100 }),
        pointToLayer: (_feature, latlng) => L.circleMarker(latlng, { radius: 5, color: "#155a41", fillColor: "#3a9a6c", fillOpacity: 0.8 }),
        onEachFeature: (feature, featureLayer) => {
          const properties = feature.properties || {};
          const rows = Object.entries(properties).filter(([, value]) => value !== null && typeof value !== "object").slice(0, 24);
          featureLayer.bindPopup(`<div class="popup-title">${escapeHtml(feature.id || source.source_name || "Feature")}</div>${rows.map(([key, value]) => `<div class="popup-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}`);
          featureLayer.on("click", () => showParcelDetail({ ...properties, geometry: feature.geometry, source_id: source.source_id, source_name: source.source_name, source_type: source.source_type }));
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
        <button class="icon-button source-delete" data-source-delete="${escapeHtml(source.source_id)}" title="Delete source" aria-label="Delete ${escapeHtml(source.source_name)}">×</button>
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
    document.querySelectorAll("[data-source-delete]").forEach((button) => button.addEventListener("click", async () => {
      try { await request(`/api/sources/${encodeURIComponent(button.dataset.sourceDelete)}`, { method: "DELETE" }); await refreshSources(); await loadDashboardSummary(); }
      catch (error) { setMessage("upload-message", error.message, "error"); }
    }));
  }

  async function refreshSources() {
    try {
      const result = await request("/api/data/sources");
      state.sources = result.sources || [];
      renderSources();
      for (const source of state.sources) await loadLayer(source, true);
      const details = await Promise.all(state.sources.map((source) => request(`/api/sources/${encodeURIComponent(source.source_id)}`).catch(() => null)));
      state.records = details.flatMap((source) => (source?.records || []).map((record) => ({ ...record, source_id: source.source_id, source_name: source.source_name, source_type: source.source_type })));
      renderParcels();
      const filter = byId("parcel-source-filter");
      filter.innerHTML = '<option value="">All sources</option>' + state.sources.map((source) => `<option value="${escapeHtml(source.source_id)}">${escapeHtml(source.source_name)}</option>`).join("");
    } catch (error) {
      setMessage("upload-message", error.message, "error");
    }
  }

  function renderParcels() {
    const harmonized = state.lastJob?.feature_collection?.features || [];
    const rows = harmonized.length ? harmonized.map((feature) => ({ ...feature.properties, geometry: feature.geometry })) : state.records.map((record) => ({ ...record.raw_attributes, ...record }));
    const query = byId("parcel-search").value.trim().toLowerCase();
    const sourceId = byId("parcel-source-filter").value;
    const visible = rows.filter((row) => {
      const text = [row.ulpin, row.khasra, row.village, row.district, row.parcel_id, row.source_name].flat().join(" ").toLowerCase();
      const sourceMatches = !sourceId || (row.source_id || []).includes?.(sourceId) || row.source_id === sourceId || (row.source_records || []).some((source) => source.source_id === sourceId);
      return text.includes(query) && sourceMatches;
    });
    byId("parcel-count").textContent = `${visible.length} records`;
    byId("parcel-table").innerHTML = visible.length ? visible.map((row, index) => `<tr data-parcel-index="${index}"><td>${escapeHtml(row.harmonized_id || row.parcel_id || row.source_id || `Record ${index + 1}`)}</td><td>${escapeHtml(row.ulpin || "—")}</td><td>${escapeHtml(row.village || "—")}</td><td>${escapeHtml(row.khasra || "—")}</td><td>${escapeHtml(row.area ?? "—")}</td><td>${escapeHtml(row.source_count ?? (row.source_name ? 1 : "—"))}</td><td>${Number.isFinite(row.confidence) ? `${(row.confidence * 100).toFixed(1)}%` : "—"}</td><td>${escapeHtml(row.conflicts?.length ?? 0)}</td><td>${row.conflicts?.length ? "Requires review" : "Unresolved"}</td></tr>`).join("") : '<tr><td colspan="9" class="empty-state">No matching parcel records.</td></tr>';
    document.querySelectorAll("[data-parcel-index]").forEach((element) => element.addEventListener("click", () => showParcelDetail(visible[Number(element.dataset.parcelIndex)])));
  }

  function showParcelDetail(parcel) {
    if (!parcel) return;
    byId("parcel-detail").hidden = false;
    const entries = Object.entries(parcel).filter(([key]) => key !== "geometry");
    byId("parcel-detail-content").innerHTML = `<h3>${escapeHtml(parcel.ulpin || parcel.harmonized_id || parcel.parcel_id || "Selected record")}</h3><dl>${entries.map(([key, value]) => `<dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd>${escapeHtml(typeof value === "object" ? JSON.stringify(value) : value ?? "—")}</dd>`).join("")}</dl>`;
    if (parcel.geometry && state.map && window.L) {
      try { const selected = L.geoJSON({ type: "Feature", geometry: parcel.geometry, properties: {} }, { style: { color: "#a84138", weight: 4, fillOpacity: 0.08 } }); selected.addTo(state.map); if (selected.getBounds().isValid()) state.map.fitBounds(selected.getBounds(), { padding: [28, 28], maxZoom: 17 }); setTimeout(() => state.map.removeLayer(selected), 4500); }
      catch (error) { console.debug("Selected geometry is not drawable:", error); }
    }
  }

  function renderConflicts() {
    const conflicts = state.lastJob?.conflicts || [];
    const query = byId("conflict-search").value.trim().toLowerCase();
    const type = byId("conflict-type-filter").value;
    const rows = conflicts.filter((item) => (!type || item.conflict_type === type || item.type === type) && JSON.stringify(item).toLowerCase().includes(query));
    byId("conflict-count").textContent = `${rows.length} items`;
    byId("conflict-table").innerHTML = rows.length ? rows.map((item) => `<tr><td>${escapeHtml(item.record || item.identity?.join(" · ") || "—")}</td><td>${escapeHtml(item.conflict_type || item.type || "potential_data_conflict")}</td><td>${escapeHtml(item.severity || "medium")}</td><td>${Number.isFinite(item.confidence) ? `${(item.confidence * 100).toFixed(1)}%` : "—"}</td><td>${escapeHtml((item.source_values || item.source || []).map?.((value) => value.source_name || value.source_id || value.source || "").join(", ") || "—")}</td><td>${escapeHtml(item.explanation || item.rule || "Potential data conflict; requires review.")}</td><td>${escapeHtml(item.resolution_status || item.status || "Requires review")}</td></tr>`).join("") : '<tr><td colspan="7" class="empty-state">No items match this filter.</td></tr>';
  }

  async function downloadJobExport(format, label) {
    if (!state.lastJob?.job_id) return;
    try {
      const response = await fetch(apiUrl(`/api/exports/${encodeURIComponent(state.lastJob.job_id)}?format=${encodeURIComponent(format)}`));
      if (!response.ok) throw new Error(`Export failed (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${state.lastJob.job_id}-${label}.${format === "conflicts" || format === "provenance" ? "json" : format}`;
      anchor.click();
      URL.revokeObjectURL(url);
      setMessage("export-message", `${label} export downloaded.`, "success");
    } catch (error) { setMessage("export-message", error.message, "error"); }
  }

  function fileAsDataUrl(file) {
    return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file); });
  }

  function renderStages(stages) {
    if (!stages) return "";
    return Object.entries(stages).map(([name, value]) => `${name.replaceAll("_", " ")}: ${value.status}${value.reason ? ` (${value.reason})` : ""}`).join("\n");
  }

  function recordsBounds(records) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const collect = (coordinate) => {
      if (Array.isArray(coordinate) && coordinate.length >= 2 && typeof coordinate[0] === "number" && typeof coordinate[1] === "number") {
        minX = Math.min(minX, coordinate[0]);
        minY = Math.min(minY, coordinate[1]);
        maxX = Math.max(maxX, coordinate[0]);
        maxY = Math.max(maxY, coordinate[1]);
      }
      else if (Array.isArray(coordinate)) coordinate.forEach(collect);
    };
    (records || []).forEach((record) => collect(record.geometry?.coordinates));
    return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null;
  }

  function showFilePreview(inputId, imageId) {
    const file = byId(inputId).files[0];
    const image = byId(imageId);
    if (!file) { image.hidden = true; return; }
    const previous = state.previewUrls.get(imageId);
    if (previous) URL.revokeObjectURL(previous);
    const url = URL.createObjectURL(file);
    state.previewUrls.set(imageId, url);
    image.src = url;
    image.hidden = false;
  }

  function drawInferenceLayer(polygons, layerId) {
    if (!polygons || polygons.is_pixel_coordinates || !polygons.features?.length || !state.map || !window.L) return false;
    const oldLayer = state.layers.get(layerId);
    if (oldLayer && state.map.hasLayer(oldLayer)) state.map.removeLayer(oldLayer);
    const layer = L.geoJSON(polygons, { style: { color: "#d17b25", weight: 2, fillColor: "#e6a14d", fillOpacity: 0.35 } }).addTo(state.map);
    state.layers.set(layerId, layer);
    if (layer.getBounds().isValid()) state.map.fitBounds(layer.getBounds(), { padding: [24, 24], maxZoom: 17 });
    updateMapSummary();
    return true;
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
  byId("parcel-search").addEventListener("input", renderParcels);
  byId("parcel-source-filter").addEventListener("change", renderParcels);
  byId("conflict-search").addEventListener("input", renderConflicts);
  byId("conflict-type-filter").addEventListener("change", renderConflicts);
  byId("close-parcel-detail").addEventListener("click", () => { byId("parcel-detail").hidden = true; });
  document.querySelectorAll(".nav-item").forEach((link) => link.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  }));
  byId("layer-opacity").addEventListener("input", () => {
    const opacity = Number(byId("layer-opacity").value) / 100;
    for (const layer of state.layers.values()) layer.setStyle?.({ fillOpacity: opacity });
  });

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
    byId("upload-message").textContent = "Inspecting, validating and normalizing source…";
    byId("register-source").hidden = true;
    byId("upload-preview").hidden = true;
    try {
      const result = await request("/api/sources/upload", { method: "POST", body: form });
      state.pendingSource = result;
      byId("upload-message").textContent = `${result.valid_records ?? result.total_records ?? 0} valid record(s) · ${result.invalid_records ?? 0} invalid · ${result.detected_crs || "CRS unknown"}`;
      const bounds = result.bounds || result.bounding_box || recordsBounds(result.records);
      const requiredFields = ["ulpin", "district", "village", "tehsil", "khasra", "area"];
      const detectedFields = (result.detected_fields || []).map((field) => typeof field === "string" ? field : field.name).filter(Boolean);
      const missingFields = requiredFields.filter((field) => !detectedFields.includes(field));
      byId("upload-preview").hidden = false;
      byId("upload-preview").innerHTML = `<strong>Inspection</strong><span>Features: ${escapeHtml(result.total_records ?? result.valid_records ?? 0)} · Geometry: ${escapeHtml(result.geometry_types?.join(", ") || result.geometry_type || "unknown")}</span><span>CRS: ${escapeHtml(result.detected_crs || "unknown")}</span><span>Bounds: ${escapeHtml(bounds ? JSON.stringify(bounds) : "not available")}</span><span>Fields: ${escapeHtml(detectedFields.join(", ") || "none detected")}</span><span>Missing core fields: ${escapeHtml(missingFields.join(", ") || "none")}</span><span>Warnings: ${escapeHtml((result.warnings || []).join("; ") || "none")}</span>`;
      byId("register-source").hidden = false;
      if (result.warnings?.length) byId("upload-message").textContent += ` · ${result.warnings.length} warning(s)`;
    } catch (error) {
      byId("upload-message").textContent = error.message;
      byId("upload-message").classList.add("error");
    }
  });

  byId("register-source").addEventListener("click", async () => {
    if (!state.pendingSource) return;
    try {
      const registered = await request("/api/sources/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.pendingSource) });
      byId("upload-message").textContent = `Registered ${registered.source_name} · ${registered.total_records ?? registered.valid_records ?? 0} records`;
      byId("register-source").hidden = true;
      byId("upload-preview").hidden = true;
      byId("upload-form").reset();
      state.pendingSource = null;
      await Promise.all([refreshSources(), loadDashboardSummary()]);
    } catch (error) { setMessage("upload-message", error.message, "error"); }
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
    setMessage("harmonize-result", "Validating sources, generating spatial candidates and probing available models…");
    try {
      const job = await request("/api/harmonization/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source_ids: sourceIds }) });
      state.lastJob = job;
      setMessage("harmonize-result", `${job.feature_count} unified record(s) · ${job.conflict_count} potential conflict(s) · ${job.anomaly_count || 0} potential anomaly review(s) · ${job.candidate_count || 0} spatial candidates\n${renderStages(job.stages)}\n${(job.limitations || []).join("\n")}`, "success");
      byId("export-harmonized").disabled = false;
      for (const id of ["export-geojson", "export-csv", "export-conflicts", "export-provenance"]) byId(id).disabled = false;
      if (job.feature_collection) drawHarmonized(job.feature_collection);
      renderParcels(); renderConflicts();
      await loadDashboardSummary();
    } catch (error) { setMessage("harmonize-result", error.message, "error"); }
  });

  function drawHarmonized(collection) {
    const existing = state.layers.get("__harmonized__");
    if (existing && state.map?.hasLayer(existing)) state.map.removeLayer(existing);
    if (!window.L || !state.map) return;
    const layer = L.geoJSON(collection, { style: (feature) => feature.properties?.conflicts?.length ? { color: "#a84138", weight: 3, fillColor: "#c75b50", fillOpacity: 0.25 } : { color: "#bc6d24", weight: 3, fillColor: "#e6a14d", fillOpacity: 0.24 }, onEachFeature: (feature, featureLayer) => featureLayer.on("click", () => showParcelDetail({ ...feature.properties, geometry: feature.geometry })) }).addTo(state.map);
    state.layers.set("__harmonized__", layer);
    if (layer.getBounds().isValid()) state.map.fitBounds(layer.getBounds(), { padding: [24, 24], maxZoom: 16 });
    updateMapSummary();
  }

  byId("export-harmonized").addEventListener("click", () => downloadJobExport("geojson", "harmonized"));
  byId("export-geojson").addEventListener("click", () => downloadJobExport("geojson", "harmonized"));
  byId("export-csv").addEventListener("click", () => downloadJobExport("csv", "harmonized"));
  byId("export-conflicts").addEventListener("click", () => downloadJobExport("conflicts", "conflicts"));
  byId("export-provenance").addEventListener("click", () => downloadJobExport("provenance", "provenance"));

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

  byId("building-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = byId("building-image").files[0];
    if (!file) return;
    setMessage("building-result", "Checking Model 2 readiness…");
    try {
      const encoded = await fileAsDataUrl(file);
      const result = await request("/api/models/building_extractor/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: { base64: encoded } }) });
      byId("building-mask-preview").src = `data:image/png;base64,${result.evidence.mask_png_base64}`;
      byId("building-mask-preview").hidden = false;
      const mapped = drawInferenceLayer(result.evidence.polygons, "__model2_buildings__");
      setMessage("building-result", `${result.evidence.polygon_count} footprint(s) · ${result.evidence.building_pixel_percentage.toFixed(2)}% building pixels · ${result.evidence.polygon_space}${mapped ? " · footprints added to map" : " · pixel-space footprints are not plotted on the geographic map"}`, "success");
    } catch (error) { byId("building-mask-preview").hidden = true; setMessage("building-result", error.message, "error"); }
  });

  byId("change-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const before = byId("before-image").files[0];
    const after = byId("after-image").files[0];
    if (!before || !after) return;
    setMessage("change-result", "Checking Model 3 readiness…");
    try {
      const [beforeEncoded, afterEncoded] = await Promise.all([fileAsDataUrl(before), fileAsDataUrl(after)]);
      const result = await request("/api/models/change_detector/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ before_image: { base64: beforeEncoded }, after_image: { base64: afterEncoded } }) });
      byId("change-mask-preview").src = `data:image/png;base64,${result.evidence.mask_png_base64}`;
      byId("change-mask-preview").hidden = false;
      const mapped = drawInferenceLayer(result.evidence.change_polygons, "__model3_change__");
      setMessage("change-result", `${result.decision} · ${result.evidence.changed_pixel_percentage.toFixed(2)}% changed pixels (${result.evidence.changed_pixel_count}) · mean positive-pixel score ${Number.isFinite(result.confidence) ? `${(result.confidence * 100).toFixed(1)}% (uncalibrated)` : "not available"}${mapped ? " · change footprints added to map" : " · pixel-space footprints are not plotted on the geographic map"}`, "success");
    } catch (error) { byId("change-mask-preview").hidden = true; setMessage("change-result", error.message, "error"); }
  });

  byId("before-image").addEventListener("change", () => showFilePreview("before-image", "before-preview"));
  byId("after-image").addEventListener("change", () => showFilePreview("after-image", "after-preview"));
  byId("building-image").addEventListener("change", () => showFilePreview("building-image", "building-source-preview"));

  async function refreshSystemDetails() {
    try {
      const [health, readiness] = await Promise.all([request("/health"), request("/api/models/readiness")]);
      byId("system-details").textContent = JSON.stringify({ health, models: readiness.models }, null, 2);
    } catch (error) { byId("system-details").textContent = error.message; }
  }

  initializeMap();
  state.api = window.location.origin;
  byId("api-url").value = state.api;
  Promise.all([checkHealth(), loadDashboardSummary(), loadModels(), refreshSources(), refreshSystemDetails()]);
})();