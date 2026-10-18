"use strict";

const FEATURES = [
  "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
  "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
  "lum", "dep", "com", "agg_", "int", "atm", "col", "lat", "long", "hour",
  "nb_victim", "nb_vehicules"
];
const CONTRACT = "prepared-features-v1";
const FLOAT32_MAX = 3.4028234663852886e38;
const form = document.getElementById("feature-form");
const fields = document.getElementById("feature-fields");
const message = document.getElementById("form-message");
const predictButton = document.getElementById("predict");
const exampleButton = document.getElementById("load-example");
const checkButton = document.getElementById("check-api");

for (const [index, name] of FEATURES.entries()) {
  const label = document.createElement("label");
  label.className = "feature";
  const caption = document.createElement("span");
  caption.textContent = `${String(index + 1).padStart(2, "0")}  ${name}`;
  const input = document.createElement("input");
  input.id = `feature-${name}`;
  input.name = name;
  input.type = "number";
  input.step = ["lat", "long"].includes(name) ? "any" : "1";
  input.required = true;
  input.placeholder = ["lat", "long"].includes(name) ? "Number" : "Integer";
  label.append(caption, input);
  fields.append(label);
}

function clearResult() {
  document.getElementById("prediction-result").hidden = true;
  document.getElementById("empty-result").hidden = false;
  document.getElementById("prediction-code").textContent = "";
  document.getElementById("model-hash").textContent = "";
  document.getElementById("contract-version").textContent = "";
}

function countFields() {
  const count = FEATURES.filter(name => form.elements.namedItem(name).value !== "").length;
  document.getElementById("field-count").textContent = `${count} / 28`;
}

function showMessage(text) {
  message.textContent = text;
  message.hidden = !text;
}

async function request(path, options = {}) {
  const response = await fetch(path, {...options, cache: "no-store", signal: AbortSignal.timeout(35000)});
  const payload = await response.json();
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${payload.detail || "Request failed."}`);
  return payload;
}

function hasIdentity(payload) {
  return payload.contract_version === CONTRACT && typeof payload.model_sha256 === "string" && /^[a-f0-9]{64}$/.test(payload.model_sha256);
}

async function checkApi() {
  const status = document.getElementById("api-status");
  checkButton.disabled = true;
  status.textContent = "Checking API…";
  status.className = "status";
  try {
    const ready = await request("/api/health/ready");
    if (ready.status !== "ready" || !hasIdentity(ready)) throw new Error("Unexpected readiness contract.");
    status.textContent = "API ready";
    status.className = "status ready";
  } catch {
    status.textContent = "API unavailable";
    status.className = "status unavailable";
  } finally {
    checkButton.disabled = false;
  }
}

form.addEventListener("input", () => { clearResult(); countFields(); showMessage(""); });
checkButton.addEventListener("click", checkApi);

exampleButton.addEventListener("click", async () => {
  exampleButton.disabled = true;
  clearResult();
  showMessage("");
  try {
    const example = await request("/api/example");
    if (Object.keys(example.features).length !== 28 ||
        !FEATURES.every(name => typeof example.features[name] === "number" && Number.isFinite(example.features[name]))) {
      throw new Error("The reference example does not match the feature contract.");
    }
    for (const name of FEATURES) form.elements.namedItem(name).value = example.features[name];
    countFields();
  } catch (error) {
    showMessage(error.message);
  } finally {
    exampleButton.disabled = false;
  }
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  clearResult();
  showMessage("");
  const features = {};
  for (const name of FEATURES) {
    const input = form.elements.namedItem(name);
    const value = input.valueAsNumber;
    if (input.value === "" || !Number.isFinite(value) || Math.abs(value) > FLOAT32_MAX ||
        (!["lat", "long"].includes(name) && !Number.isInteger(value))) {
      showMessage(`Enter a valid ${["lat", "long"].includes(name) ? "number" : "integer"} for ${name}.`);
      input.focus();
      return;
    }
    features[name] = value;
  }
  predictButton.disabled = true;
  exampleButton.disabled = true;
  const inputs = [...fields.querySelectorAll("input")];
  inputs.forEach(input => { input.disabled = true; });
  predictButton.textContent = "Requesting…";
  try {
    const result = await request("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({features})
    });
    if (![0, 1].includes(result.prediction_code) || !hasIdentity(result)) throw new Error("Unexpected prediction contract.");
    document.getElementById("prediction-code").textContent = String(result.prediction_code);
    document.getElementById("contract-version").textContent = result.contract_version;
    document.getElementById("model-hash").textContent = result.model_sha256;
    document.getElementById("empty-result").hidden = true;
    document.getElementById("prediction-result").hidden = false;
  } catch (error) {
    showMessage(error.message);
  } finally {
    inputs.forEach(input => { input.disabled = false; });
    predictButton.disabled = false;
    exampleButton.disabled = false;
    predictButton.textContent = "Get prediction →";
  }
});

checkApi();
