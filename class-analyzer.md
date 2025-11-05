---
layout: default
title: Class Analyzer
permalink: /class-analyzer/
---

# PELINN-Q Class Analyzer

Analyseer classes en genereer interactieve inzichten op basis van de broncode in de PELINN-Q repository.

<div class="app-wrapper">
  <section class="card">
    <h2>🔎 Fetch Class from Repository</h2>
    <p>Voer het pad in naar een Python-bestand binnen de <code>PELINN-Q</code> repository om de code automatisch op te halen.</p>
    <label for="class-path" class="sr-only">Pad naar class-bestand</label>
    <div class="input-row">
      <input id="class-path" type="text" value="pelinn/model.py" placeholder="bijv. pelinn/model.py" />
      <button id="fetch-button" type="button">Fetch Code</button>
    </div>
    <p class="hint">De code wordt opgehaald vanuit de <a href="https://github.com/BramDo/PELINN-Q" target="_blank" rel="noopener">hoofdtak van de repository</a>.</p>
    <p id="fetch-status" role="status" aria-live="polite" class="status"></p>
  </section>

  <section class="card">
    <h2>🧪 Analyze Class</h2>
    <p>Plak een class of gebruik de fetch-functionaliteit hierboven en klik daarna op <strong>Analyze Class</strong> voor een automatisch rapport.</p>
    <label for="code-input" class="sr-only">Class code</label>
    <textarea id="code-input" rows="16" placeholder="Class code verschijnt hier na het ophalen..."></textarea>
    <button id="analyze-button" type="button">Analyze Class</button>
  </section>

  <section class="card" id="analysis-section" hidden>
    <h2>📈 Analysis</h2>
    <div id="analysis-output"></div>
  </section>
</div>

<style>
.app-wrapper {
  display: grid;
  gap: 1.5rem;
}

.card {
  background: var(--card-bg, #fff);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}

.card h2 {
  margin-top: 0;
  margin-bottom: 0.75rem;
}

.input-row {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.input-row input {
  flex: 1 1 240px;
  padding: 0.6rem 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(0, 0, 0, 0.18);
  font-family: inherit;
}

button {
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0.6rem 1.25rem;
  cursor: pointer;
  font-weight: 600;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

button:hover,
button:focus {
  outline: none;
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
}

textarea {
  width: 100%;
  border-radius: 8px;
  border: 1px solid rgba(0, 0, 0, 0.18);
  padding: 0.75rem;
  font-family: "Fira Code", "Courier New", monospace;
  resize: vertical;
  min-height: 260px;
}

.status {
  margin-top: 0.75rem;
  font-size: 0.95rem;
  color: #334155;
}

.status.error {
  color: #b91c1c;
}

#analysis-output ul {
  margin: 0;
  padding-left: 1.25rem;
}

.analysis-grid {
  display: grid;
  gap: 1rem;
}

.analysis-grid .panel {
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 10px;
  padding: 1rem;
  background: rgba(148, 163, 184, 0.1);
}

.analysis-grid .panel h3 {
  margin-top: 0;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
</style>

<script>
(function() {
  const classPathInput = document.getElementById('class-path');
  const fetchButton = document.getElementById('fetch-button');
  const analyzeButton = document.getElementById('analyze-button');
  const codeInput = document.getElementById('code-input');
  const statusEl = document.getElementById('fetch-status');
  const analysisSection = document.getElementById('analysis-section');
  const analysisOutput = document.getElementById('analysis-output');

  const GITHUB_BASE = 'https://raw.githubusercontent.com/BramDo/PELINN-Q/main/';

  function setStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle('error', isError);
  }

  function extractClassSummaries(code) {
    const classRegex = /^class\s+(\w+)\s*(?:\(([^)]*)\))?:/gm;
    const summaries = [];
    let match;

    while ((match = classRegex.exec(code)) !== null) {
      const [fullMatch, className, baseClasses] = match;
      const startIndex = match.index + fullMatch.length;
      const classBody = code.slice(startIndex);
      const bodyEnd = classBody.search(/^\S/m);
      const scopedBody = bodyEnd === -1 ? classBody : classBody.slice(0, bodyEnd);
      const lines = scopedBody.split('\n');

      const methodRegex = /^\s+def\s+(\w+)\s*\(/gm;
      const methods = [];
      let methodMatch;
      while ((methodMatch = methodRegex.exec(scopedBody)) !== null) {
        methods.push(methodMatch[1]);
      }

      const docstringMatch = scopedBody.match(/^\s+"""([\s\S]*?)"""/);
      const docstring = docstringMatch ? docstringMatch[1].trim() : 'Geen docstring gevonden.';

      summaries.push({
        className,
        baseClasses: baseClasses ? baseClasses.split(',').map(s => s.trim()).filter(Boolean) : [],
        methods,
        docstring,
        lineCount: lines.filter(Boolean).length
      });
    }

    return summaries;
  }

  function buildAnalysis(code) {
    const importRegex = /^import\s+[^\n]+|^from\s+[^\n]+/gm;
    const imports = Array.from(code.matchAll(importRegex)).map(m => m[0]);
    const summaries = extractClassSummaries(code);

    if (!summaries.length) {
      return '<p>Geen classes gevonden in de huidige invoer.</p>';
    }

    const panels = summaries.map(summary => {
      const { className, baseClasses, methods, docstring, lineCount } = summary;
      const baseInfo = baseClasses.length ? `<p><strong>Base classes:</strong> ${baseClasses.join(', ')}</p>` : '';
      const methodsList = methods.length ? `<ul>${methods.map(m => `<li><code>${m}()</code></li>`).join('')}</ul>` : '<p>Geen methods gevonden.</p>';

      return `
        <div class="panel">
          <h3>${className}</h3>
          ${baseInfo}
          <p><strong>Regels in class:</strong> ${lineCount}</p>
          <p><strong>Docstring:</strong> ${docstring}</p>
          <div>
            <strong>Methods (${methods.length}):</strong>
            ${methodsList}
          </div>
        </div>
      `;
    }).join('');

    const importPanel = imports.length ? `
      <div class="panel">
        <h3>Imports</h3>
        <ul>${imports.map(imp => `<li><code>${imp}</code></li>`).join('')}</ul>
      </div>
    ` : '';

    return `
      <div class="analysis-grid">
        ${importPanel}
        ${panels}
      </div>
    `;
  }

  async function fetchCode() {
    const path = classPathInput.value.trim();
    if (!path) {
      setStatus('Voer een pad in naar een bestand (bijv. pelinn/model.py).', true);
      return;
    }

    setStatus('Bezig met ophalen van broncode...');
    try {
      const response = await fetch(GITHUB_BASE + path);
      if (!response.ok) {
        throw new Error(`Kon bestand niet ophalen (status ${response.status}).`);
      }
      const code = await response.text();
      codeInput.value = code;
      setStatus('Broncode succesvol opgehaald. Je kunt nu een analyse uitvoeren.');
    } catch (error) {
      console.error(error);
      setStatus(`Ophalen mislukt: ${error.message}`, true);
    }
  }

  function analyzeCode() {
    const code = codeInput.value.trim();
    if (!code) {
      analysisSection.hidden = true;
      setStatus('Voer of haal eerst code op voor analyse.', true);
      return;
    }

    const analysisHtml = buildAnalysis(code);
    analysisOutput.innerHTML = analysisHtml;
    analysisSection.hidden = false;
  }

  fetchButton.addEventListener('click', fetchCode);
  analyzeButton.addEventListener('click', analyzeCode);
})();
</script>

<p><a href="./">⬅️ Terug naar de startpagina</a></p>
