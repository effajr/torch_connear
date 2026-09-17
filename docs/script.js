const CONFIG = {
  audioBase: "samples",
  snrColumnWidth: 64,

  phrases: [
    { id: "1221-135767-0017", label: "Utterance 1" },
    { id: "4077-13751-0015", label: "Utterance 2" },
    { id: "2961-960-0018", label: "Utterance 3" },
    { id: "8455-210777-0000", label: "Utterance 4" }
  ],

  snrLevels: [-6, -3, 0, 3],

  models: [
    { id: "icconvtasnet", label: "Inter-Channel ConvTasNet" },
    { id: "fasnet", label: "FaSNet TAC" }
  ],

  losses: [
    { id: "sisdr", label: "SI-SDR" },
    { id: "cosdr", label: "Co-SDR (proposed)" }
  ]

};

const audioPlayers = new Set();

function formatSnr(snr) {
  const sign = snr > 0 ? "+" : "";
  return `${sign}${snr}dB`;
}

function getCleanPath(phraseId) {
  return `${CONFIG.audioBase}/clean/${phraseId}.wav`;
}

function getMixPath(phraseId, snr) {
  return `${CONFIG.audioBase}/mixtures/${phraseId}_ssn_${snr}.wav`;
}

function getEnhancedPath(phraseId, modelId, lossId, snr) {
  return `${CONFIG.audioBase}/${modelId}-${lossId}/${phraseId}_snr${snr}.wav`;
}

function audioCell(src, label, colName) {
  const td = document.createElement("td");

  if (colName) {
    td.setAttribute("data-col", colName);
  }

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = src;
  audio.setAttribute("aria-label", label);

  audio.addEventListener("play", () => {
    audioPlayers.forEach((player) => {
      if (player !== audio) {
        player.pause();
      }
    });

    audioPlayers.add(audio);
  });

  audio.addEventListener("ended", () => {
    audioPlayers.delete(audio);
  });

  audio.addEventListener("error", () => {
    audioPlayers.delete(audio);
    td.innerHTML = "";

    const span = document.createElement("span");
    span.className = "missing";
    span.textContent = "missing file";
    td.appendChild(span);
  });

  td.appendChild(audio);
  return td;
}

function makeHeaderCell(text, isCosdr, isSisdr) {
  const th = document.createElement("th");
  th.textContent = text;

  if (isCosdr) {
    th.classList.add("is-cosdr");
  }

  if (isSisdr) {
    th.classList.add("is-sisdr");
  }

  return th;
}

function renderSnrColumn() {
  const column = document.getElementById("snrColumn");
  document.documentElement.style.setProperty(
    "--snr-column-width",
    `${CONFIG.snrColumnWidth}px`
  );
  column.innerHTML = "";

  const heading = document.createElement("div");
  heading.className = "snr-column-heading";
  heading.textContent = "SNR";
  column.appendChild(heading);

  CONFIG.snrLevels.forEach((snr) => {
    const label = document.createElement("div");
    label.className = "snr-label";
    label.textContent = formatSnr(snr);
    column.appendChild(label);
  });
}

let currentPhraseId = CONFIG.phrases[0].id;

function renderPhraseSelector() {
  const nav = document.getElementById("phraseSelector");
  nav.innerHTML = "";

  CONFIG.phrases.forEach((phrase) => {
    const btn = document.createElement("button");

    btn.type = "button";
    btn.className = "phrase-btn";
    btn.textContent = phrase.label;
    btn.setAttribute(
      "aria-current",
      String(phrase.id === currentPhraseId)
    );

    btn.addEventListener("click", () => {
      currentPhraseId = phrase.id;
      renderAll();
    });

    nav.appendChild(btn);
  });
}

function renderCleanReference() {
  const wrap = document.getElementById("cleanRef");
  const modelPanels = document.getElementById("modelPanels");

  let section = document.getElementById("cleanReferenceSection");

  if (!section) {
    section = document.createElement("section");
    section.id = "cleanReferenceSection";
    section.className = "clean-reference-section";
    section.setAttribute(
      "aria-labelledby",
      "clean-reference-heading"
    );

    const heading = document.createElement("h2");
    heading.id = "clean-reference-heading";
    heading.textContent = "Clean reference";

    section.appendChild(heading);
    section.appendChild(wrap);

    modelPanels.parentNode.insertBefore(section, modelPanels);
  }

  wrap.innerHTML = "";

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = getCleanPath(currentPhraseId);
  audio.setAttribute("aria-label", "Clean reference");

  audio.addEventListener("error", () => {
    const missing = document.createElement("span");
    missing.className = "missing";
    missing.textContent = "missing file";
    audio.replaceWith(missing);
  });

  wrap.appendChild(audio);
}

function buildMixtureTable() {
  const table = document.createElement("table");
  table.className = "compare-table";

  const thead = document.createElement("thead");

  const titleRow = document.createElement("tr");
  const titleTh = document.createElement("th");

  titleTh.colSpan = 1;
  titleTh.className = "table-title";
  titleTh.textContent = "Input Mixture";

  titleRow.appendChild(titleTh);
  thead.appendChild(titleRow);

  const headRow = document.createElement("tr");
  headRow.className = "col-header-row";
  headRow.appendChild(makeHeaderCell("Noisy"));

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  CONFIG.snrLevels.forEach((snr) => {
    const tr = document.createElement("tr");

    tr.appendChild(
      audioCell(
        getMixPath(currentPhraseId, snr),
        `Mixture at ${formatSnr(snr)}`,
        "Mixture"
      )
    );

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  return table;
}

function buildModelTable(model) {
  const table = document.createElement("table");
  table.className = "compare-table";

  const thead = document.createElement("thead");

  const titleRow = document.createElement("tr");
  const titleTh = document.createElement("th");

  titleTh.colSpan = CONFIG.losses.length;
  titleTh.className = "table-title";
  titleTh.textContent = model.label;
  titleTh.id = `model-${model.id}-heading`;

  titleRow.appendChild(titleTh);
  thead.appendChild(titleRow);

  const headRow = document.createElement("tr");
  headRow.className = "col-header-row";

  CONFIG.losses.forEach((loss) => {
    headRow.appendChild(
      makeHeaderCell(
        loss.label,
        loss.id === "cosdr",
        loss.id === "sisdr"
      )
    );
  });

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  CONFIG.snrLevels.forEach((snr) => {
    const tr = document.createElement("tr");

    CONFIG.losses.forEach((loss) => {
      const cell = audioCell(
        getEnhancedPath(
          currentPhraseId,
          model.id,
          loss.id,
          snr
        ),
        `${model.label}, ${loss.label}, ${formatSnr(snr)}`,
        loss.label
      );

      cell.classList.add(
        loss.id === "cosdr" ? "is-cosdr" : "is-sisdr"
      );

      tr.appendChild(cell);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  return table;
}

function renderModelPanels() {
  const container = document.getElementById("modelPanels");
  container.innerHTML = "";

  const section = document.createElement("section");
  section.setAttribute("aria-labelledby", "models-heading");

  const heading = document.createElement("h2");
  heading.id = "models-heading";
  heading.textContent = "Enhancement examples";

  section.appendChild(heading);

  const comparison = document.createElement("div");
  comparison.className = "comparison";

  const snrColumn = document.createElement("div");
  snrColumn.id = "snrColumn";
  snrColumn.className = "snr-column";
  comparison.appendChild(snrColumn);

  const tablesRow = document.createElement("div");
  tablesRow.className = "tables-row";

  tablesRow.appendChild(buildMixtureTable());

  CONFIG.models.forEach((model) => {
    tablesRow.appendChild(buildModelTable(model));
  });

  comparison.appendChild(tablesRow);
  section.appendChild(comparison);
  container.appendChild(section);

  renderSnrColumn();
}

function renderAll() {
  renderPhraseSelector();
  renderCleanReference();
  renderModelPanels();
}

renderAll();