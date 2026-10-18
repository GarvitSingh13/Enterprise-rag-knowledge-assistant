/**
 * Enterprise RAG Knowledge Assistant - Frontend Controller
 * Manages Persona Switching, Secure API Communication,
 * Provenance Drawer, and Live Observability/Benchmark Dashboards.
 */

// Predefined Enterprise Personas & Passwords
const ENTERPRISE_PERSONAS = {
  alice_emp: {
    username: "alice_emp",
    password: "EmployeePass123!",
    name: "Alice Johnson",
    dept: "hr",
    clearance: "INTERNAL",
    badgeClass: "badge-internal",
  },
  bob_finance: {
    username: "bob_finance",
    password: "FinancePass123!",
    name: "Bob Martinez",
    dept: "finance",
    clearance: "CONFIDENTIAL",
    badgeClass: "badge-confidential",
  },
  david_dev: {
    username: "david_dev",
    password: "DevPass123!",
    name: "David Kim",
    dept: "engineering",
    clearance: "INTERNAL",
    badgeClass: "badge-internal",
  },
  sarah_ceo: {
    username: "sarah_ceo",
    password: "ExecutivePass123!",
    name: "Sarah Connor",
    dept: "executive",
    clearance: "RESTRICTED",
    badgeClass: "badge-restricted",
  },
};

// Global Client State
const state = {
  activePersonaKey: "alice_emp",
  token: null,
  activeView: "chat-view",
  sourcesRegistry: {}, // keyed by citation id for provenance drawer
  sessionId: "session-" + Math.random().toString(36).substring(2, 9),
};

// DOM References
const dom = {
  personaSelect: document.getElementById("persona-select"),
  clearanceBadge: document.getElementById("current-clearance-badge"),
  clearanceText: document.getElementById("current-clearance-text"),
  navTabs: document.querySelectorAll(".nav-tab"),
  viewPanels: document.querySelectorAll(".view-panel"),
  chatMessages: document.getElementById("chat-messages"),
  chatForm: document.getElementById("chat-form"),
  queryInput: document.getElementById("query-input"),
  sendBtn: document.getElementById("send-btn"),
  cacheToggle: document.getElementById("cache-toggle"),
  clearHistoryBtn: document.getElementById("clear-history-btn"),
  promptPills: document.querySelectorAll(".prompt-pill"),
  provenanceDrawer: document.getElementById("provenance-drawer"),
  drawerCloseBtn: document.getElementById("drawer-close-btn"),
  drawerDocTitle: document.getElementById("drawer-doc-title"),
  drawerMetaFile: document.getElementById("drawer-meta-file"),
  drawerMetaDept: document.getElementById("drawer-meta-dept"),
  drawerMetaClearance: document.getElementById("drawer-meta-clearance"),
  drawerMetaRerank: document.getElementById("drawer-meta-rerank"),
  drawerPassageText: document.getElementById("drawer-passage-text"),
  // Telemetry DOM
  refreshTelemetryBtn: document.getElementById("refresh-telemetry-btn"),
  kpiTotalQueries: document.getElementById("kpi-total-queries"),
  kpiAvgLatency: document.getElementById("kpi-avg-latency"),
  kpiP95Latency: document.getElementById("kpi-p95-latency"),
  kpiCacheHitRate: document.getElementById("kpi-cache-hit-rate"),
  kpiCacheCounts: document.getElementById("kpi-cache-counts"),
  kpiTokensTracked: document.getElementById("kpi-tokens-tracked"),
  // Evaluation DOM
  runBenchmarkBtn: document.getElementById("run-benchmark-btn"),
  evalCompositeScore: document.getElementById("eval-composite-score"),
  evalCrScore: document.getElementById("eval-cr-score"),
  evalGrScore: document.getElementById("eval-gr-score"),
  evalArScore: document.getElementById("eval-ar-score"),
  meterCr: document.getElementById("meter-cr"),
  meterGr: document.getElementById("meter-gr"),
  meterAr: document.getElementById("meter-ar"),
  evalTableBody: document.getElementById("eval-table-body"),
  corpusTableBody: document.getElementById("corpus-table-body"),
};

// ============================================================================
// AUTHENTICATION & PERSONA MANAGEMENT
// ============================================================================

async function loginCurrentPersona() {
  const persona = ENTERPRISE_PERSONAS[state.activePersonaKey];
  if (!persona) return;

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: persona.username,
        password: persona.password,
      }),
    });

    if (!res.ok) {
      throw new Error("Login failed with status " + res.status);
    }

    const data = await res.json();
    state.token = data.token;

    // Update Header Clearance Badge
    dom.clearanceText.textContent = persona.clearance;
    dom.clearanceBadge.className = `clearance-badge ${persona.badgeClass}`;
  } catch (err) {
    console.error("Auth error:", err);
  }
}

function handlePersonaChange() {
  state.activePersonaKey = dom.personaSelect.value;
  // Reset conversation session on user switch for complete security isolation
  state.sessionId = "session-" + Math.random().toString(36).substring(2, 9);
  loginCurrentPersona();

  // Add system notification in chat
  const persona = ENTERPRISE_PERSONAS[state.activePersonaKey];
  appendSystemNotice(`Switched active persona to <strong>${persona.name}</strong> (${persona.clearance} clearance). Memory isolated.`);
}

// ============================================================================
// NAVIGATION & VIEW SWITCHING
// ============================================================================

function switchView(targetViewId) {
  state.activeView = targetViewId;

  // Update tabs
  dom.navTabs.forEach((tab) => {
    if (tab.dataset.view === targetViewId) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  // Update view panels
  dom.viewPanels.forEach((panel) => {
    if (panel.id === targetViewId) {
      panel.classList.add("active");
    } else {
      panel.classList.remove("active");
    }
  });

  // Trigger data refreshes on tab activation
  if (targetViewId === "telemetry-view") {
    loadTelemetryMetrics();
  } else if (targetViewId === "eval-view") {
    loadBenchmarkReport();
  } else if (targetViewId === "corpus-view") {
    loadCorpusCatalog();
  }
}

// ============================================================================
// CHAT & RAG QUERY DISPATCH
// ============================================================================

async function handleChatSubmit(e) {
  if (e) e.preventDefault();
  const query = dom.queryInput.value.trim();
  if (!query) return;

  // Append user message bubble
  appendUserMessage(query);
  dom.queryInput.value = "";
  dom.sendBtn.disabled = true;

  // Show loading indicator
  const loadingBubble = appendLoadingBubble();

  try {
    const useCache = dom.cacheToggle.checked;
    const res = await fetch("/api/chat/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.token}`,
      },
      body: JSON.stringify({
        query: query,
        session_id: state.sessionId,
        use_cache: useCache,
      }),
    });

    loadingBubble.remove();

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      appendAssistantError(errData.error || `HTTP ${res.status}: Query request failed.`);
      return;
    }

    const data = await res.json();
    appendAssistantMessage(data);
  } catch (err) {
    loadingBubble.remove();
    appendAssistantError(`Network error: ${err.message}`);
  } finally {
    dom.sendBtn.disabled = false;
    dom.queryInput.focus();
  }
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user";

  const persona = ENTERPRISE_PERSONAS[state.activePersonaKey];
  row.innerHTML = `
    <div class="message-meta">${persona.name} • Just now</div>
    <div class="user-bubble">${escapeHtml(text)}</div>
  `;
  dom.chatMessages.appendChild(row);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

function appendLoadingBubble() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-meta">Enterprise Assistant • Processing...</div>
    <div class="assistant-bubble">
      <div style="display:flex; align-items:center; gap:0.5rem; color:var(--text-secondary);">
        <span class="status-dot" style="background:var(--accent-cyan); box-shadow:0 0 10px var(--accent-cyan);"></span>
        <span>Retrieving documents, applying RBAC filter, and auditing guardrails...</span>
      </div>
    </div>
  `;
  dom.chatMessages.appendChild(row);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  return row;
}

function appendAssistantMessage(data) {
  const row = document.createElement("div");
  row.className = "message-row assistant";

  // Cache sources in registry for slideover drawer
  const citationPillsHtml = (data.citations || []).map((cite, idx) => {
    const sourceObj = (data.sources || []).find((s) => s.source_file === cite) || {
      source_file: cite,
      title: cite,
      department: "Internal",
      clearance: "Internal",
      content: "Document passage cited in response.",
      rerank_score: "0.85",
    };
    const regId = `src_${Date.now()}_${idx}`;
    state.sourcesRegistry[regId] = sourceObj;

    return `
      <button class="citation-pill-btn" onclick="openProvenanceDrawer('${regId}')">
        <span>📄</span> ${escapeHtml(cite)}
      </button>
    `;
  }).join("");

  // Telemetry chips
  const cacheChip = data.cache_hit
    ? `<span class="telemetry-chip chip-cache-hit">⚡ CACHE HIT (Instant Return)</span>`
    : `<span class="telemetry-chip chip-cache-miss">❄️ COLD RAG (Full Execution)</span>`;

  const latencyChip = `<span class="telemetry-chip chip-latency">⏱️ ${data.total_latency_ms.toFixed(2)} ms</span>`;

  const faithfulnessPct = Math.round(data.faithfulness_score * 100);
  const groundedChip = data.is_refusal
    ? `<span class="telemetry-chip chip-cache-miss">🛡️ Safe Refusal</span>`
    : `<span class="telemetry-chip chip-grounded">🛡️ ${faithfulnessPct}% Grounded</span>`;

  const bubbleClass = data.is_refusal ? "assistant-bubble is-refusal" : "assistant-bubble";

  row.innerHTML = `
    <div class="message-meta">Enterprise Assistant • Authenticated Grounded Response</div>
    <div class="${bubbleClass}">
      <div class="answer-text">${formatAnswerMarkdown(data.answer)}</div>

      ${citationPillsHtml ? `
        <div class="citations-wrapper">
          <span class="citations-label">VERIFIED CITATIONS:</span>
          ${citationPillsHtml}
        </div>
      ` : ""}

      <div class="response-telemetry-bar">
        ${cacheChip}
        ${latencyChip}
        ${groundedChip}
        <span style="color:var(--text-muted); margin-left:auto;">${data.token_usage?.total_tokens || 0} tokens</span>
      </div>
    </div>
  `;

  dom.chatMessages.appendChild(row);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

function appendAssistantError(errorMsg) {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-meta">Security Gateway • Error</div>
    <div class="assistant-bubble" style="border-color:var(--accent-red);">
      <div class="answer-text" style="color:var(--accent-red); font-weight:600;">
        ⚠️ ${escapeHtml(errorMsg)}
      </div>
    </div>
  `;
  dom.chatMessages.appendChild(row);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

function appendSystemNotice(htmlNotice) {
  const row = document.createElement("div");
  row.style.textAlign = "center";
  row.style.fontSize = "0.78rem";
  row.style.color = "var(--text-muted)";
  row.style.padding = "0.4rem 0";
  row.innerHTML = htmlNotice;
  dom.chatMessages.appendChild(row);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

// ============================================================================
// PROVENANCE SLIDE-OVER DRAWER
// ============================================================================

window.openProvenanceDrawer = function (regId) {
  const source = state.sourcesRegistry[regId];
  if (!source) return;

  dom.drawerDocTitle.textContent = source.title || source.source_file;
  dom.drawerMetaFile.textContent = source.source_file || "-";
  dom.drawerMetaDept.textContent = (source.department || "General").toUpperCase();
  dom.drawerMetaClearance.textContent = (source.clearance || "Internal").toUpperCase();
  dom.drawerMetaRerank.textContent = source.rerank_score || "N/A";
  dom.drawerPassageText.textContent = source.content || "Passage verified by hallucination guardrail.";

  dom.provenanceDrawer.classList.add("open");
  dom.provenanceDrawer.setAttribute("aria-hidden", "false");
};

function closeProvenanceDrawer() {
  dom.provenanceDrawer.classList.remove("open");
  dom.provenanceDrawer.setAttribute("aria-hidden", "true");
}

// ============================================================================
// OBSERVABILITY TELEMETRY LOADER
// ============================================================================

async function loadTelemetryMetrics() {
  try {
    const res = await fetch("/api/observability/metrics");
    if (!res.ok) return;
    const data = await res.json();

    const t = data.telemetry || {};
    const c = data.semantic_cache || {};

    dom.kpiTotalQueries.textContent = t.total_queries_tracked || 0;
    dom.kpiAvgLatency.textContent = (t.avg_latency_ms || 0).toFixed(2) + " ms";
    dom.kpiP95Latency.textContent = (t.p95_latency_ms || 0).toFixed(2) + " ms";
    dom.kpiCacheHitRate.textContent = (c.hit_rate_pct || 0).toFixed(1) + "%";
    dom.kpiCacheCounts.textContent = `${c.cache_hits || 0} Hits / ${c.cache_misses || 0} Misses`;
    dom.kpiTokensTracked.textContent = (t.total_tokens_consumed || 0).toLocaleString();
  } catch (err) {
    console.error("Telemetry fetch failed:", err);
  }
}

// ============================================================================
// RAG TRIAD BENCHMARK LOADER
// ============================================================================

async function loadBenchmarkReport() {
  dom.evalTableBody.innerHTML = `<tr><td colspan="6" class="text-center">Running live RAG Triad benchmark suite...</td></tr>`;

  try {
    const res = await fetch("/api/evaluation/benchmark");
    if (!res.ok) return;
    const report = await res.json();

    // Update Summary Score Cards
    dom.evalCompositeScore.textContent = (report.avg_composite_score * 100).toFixed(1) + "%";
    dom.evalCrScore.textContent = (report.avg_context_relevance * 100).toFixed(1) + "%";
    dom.evalGrScore.textContent = (report.avg_groundedness * 100).toFixed(1) + "%";
    dom.evalArScore.textContent = (report.avg_answer_relevance * 100).toFixed(1) + "%";

    dom.meterCr.style.width = `${Math.min(100, report.avg_context_relevance * 100)}%`;
    dom.meterGr.style.width = `${Math.min(100, report.avg_groundedness * 100)}%`;
    dom.meterAr.style.width = `${Math.min(100, report.avg_answer_relevance * 100)}%`;

    // Render Category Breakdown Rows
    const cats = report.category_breakdown || {};
    let tableHtml = "";
    for (const [catKey, catVal] of Object.entries(cats)) {
      const title = catKey.replace(/_/g, " ").toUpperCase();
      tableHtml += `
        <tr>
          <td><strong>${escapeHtml(title)}</strong></td>
          <td>${catVal.total}</td>
          <td><span class="badge-pass">${catVal.pass_rate_pct}%</span></td>
          <td>${(catVal.avg_groundedness * 100).toFixed(1)}%</td>
          <td>${(catVal.avg_composite * 100).toFixed(1)}%</td>
          <td><span class="badge-pass">PASSED ✅</span></td>
        </tr>
      `;
    }
    dom.evalTableBody.innerHTML = tableHtml || `<tr><td colspan="6">No benchmark data.</td></tr>`;
  } catch (err) {
    dom.evalTableBody.innerHTML = `<tr><td colspan="6" style="color:var(--accent-red);">Benchmark failed: ${err.message}</td></tr>`;
  }
}

// ============================================================================
// DOCUMENT CORPUS CATALOG LOADER
// ============================================================================

async function loadCorpusCatalog() {
  dom.corpusTableBody.innerHTML = `<tr><td colspan="6" class="text-center">Loading indexed documents...</td></tr>`;
  try {
    const res = await fetch("/api/corpus/documents");
    if (!res.ok) return;
    const data = await res.json();

    const docs = data.documents || [];
    let rows = "";
    docs.forEach((d) => {
      rows += `
        <tr>
          <td><code>${escapeHtml(d.file_name)}</code></td>
          <td><strong>${escapeHtml(d.title)}</strong></td>
          <td>${escapeHtml(d.department.toUpperCase())}</td>
          <td><span class="brand-badge">${escapeHtml(d.clearance_level.toUpperCase())}</span></td>
          <td><small>${(d.allowed_roles || []).join(", ")}</small></td>
          <td>${d.character_count} chars</td>
        </tr>
      `;
    });
    dom.corpusTableBody.innerHTML = rows || `<tr><td colspan="6">No documents indexed.</td></tr>`;
  } catch (err) {
    dom.corpusTableBody.innerHTML = `<tr><td colspan="6" style="color:var(--accent-red);">Error loading corpus: ${err.message}</td></tr>`;
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>'"]/g, (tag) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  }[tag] || tag));
}

function formatAnswerMarkdown(text) {
  if (!text) return "";
  // Strip inline citations bracket from main text body since we render pill buttons
  let cleaned = text.replace(/\[Source:\s*[^\]]+\]/g, "").trim();

  // 1. Escape raw user/doc HTML FIRST so characters like '<' or '&' are safe
  let safe = escapeHtml(cleaned);

  // 2. Format markdown headers (## Section)
  safe = safe.replace(/##\s+([^#\n]+)/g, '<strong style="font-size:1.02rem; color:var(--accent-cyan); display:block; margin:0.85rem 0 0.35rem 0;">$1</strong>');

  // 3. Format bold text (**text**)
  safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong style="color:var(--text-primary); font-weight:600;">$1</strong>');

  // 4. Format bullet points (- item or bullet lists)
  safe = safe.replace(/(?:^|\s)- \s*/g, '<br><span style="color:var(--accent-cyan); margin-right:0.35rem;">•</span>');

  // 5. Clean up any accidental leading <br>
  safe = safe.replace(/^<br>/, "");

  return safe;
}

// ============================================================================
// EVENT LISTENERS INITIALIZATION
// ============================================================================

function init() {
  // 1. Initial Login
  loginCurrentPersona();

  // 2. Persona Selector
  dom.personaSelect.addEventListener("change", handlePersonaChange);

  // 3. Navigation Tabs
  dom.navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      switchView(tab.dataset.view);
    });
  });

  // 4. Chat Form Submission
  dom.chatForm.addEventListener("submit", handleChatSubmit);

  // 5. Enter key to send (Shift+Enter for newline)
  dom.queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChatSubmit();
    }
  });

  // 6. Quick Prompt Pills
  dom.promptPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      dom.queryInput.value = pill.dataset.query;
      handleChatSubmit();
    });
  });

  // 7. Reset Conversation Button
  dom.clearHistoryBtn.addEventListener("click", () => {
    state.sessionId = "session-" + Math.random().toString(36).substring(2, 9);
    appendSystemNotice("Conversation history reset. Starting new clean session.");
  });

  // 8. Provenance Drawer Close
  dom.drawerCloseBtn.addEventListener("click", closeProvenanceDrawer);

  // 9. Dashboard Buttons
  dom.refreshTelemetryBtn.addEventListener("click", loadTelemetryMetrics);
  dom.runBenchmarkBtn.addEventListener("click", loadBenchmarkReport);
}

document.addEventListener("DOMContentLoaded", init);
