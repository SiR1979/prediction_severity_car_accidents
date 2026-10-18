"use strict";

// Isolated logic checks: this fixture does not emulate a browser or native form validation.
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");

const source = readFileSync(join(__dirname, "app.js"), "utf8");
const html = readFileSync(join(__dirname, "index.html"), "utf8");
const application = new vm.Script(source, {filename: "app.js"});
const FEATURE_NAMES = [
  "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv", "obsm", "motor",
  "catr", "circ", "surf", "situ", "vma", "jour", "mois", "lum", "dep", "com", "agg_",
  "int", "atm", "col", "lat", "long", "hour", "nb_victim", "nb_vehicules"
];
const VALUES = {
  place: 1, catu: 2, sexe: 1, secu1: 3, year_acc: 2021, victim_age: 40, catv: 2,
  obsm: 1, motor: 1, catr: 3, circ: 2, surf: 1, situ: 1, vma: 50, jour: 7,
  mois: 6, lum: 1, dep: 75, com: 75056, agg_: 2, int: 1, atm: 0, col: 1,
  lat: 48.125, long: -2.375, hour: 9, nb_victim: 2, nb_vehicules: 1
};
const HASH = "a".repeat(64);
const CONTRACT = "prepared-features-v1";
const NODE_IDS = [
  "feature-form", "feature-fields", "form-message", "predict", "load-example", "check-api",
  "prediction-result", "empty-result", "prediction-code", "model-hash", "contract-version",
  "field-count", "api-status"
];

function scenario(name, explanation, action) {
  test(name, {concurrency: false}, async t => {
    t.diagnostic(explanation);
    await action(t);
  });
}

function reply(payload, status = 200) {
  return {ok: status >= 200 && status < 300, status, async json() { return payload; }};
}

function invalidJson(message) {
  return {ok: true, status: 200, async json() { throw new SyntaxError(message); }};
}

function ready(overrides = {}) {
  return {status: "ready", model_sha256: HASH, contract_version: CONTRACT, ...overrides};
}

function prediction(code = 1, overrides = {}) {
  return {prediction_code: code, model_sha256: HASH, contract_version: CONTRACT, ...overrides};
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((accept, decline) => { resolve = accept; reject = decline; });
  return {promise, resolve, reject};
}

class Element {
  constructor(tag, document) {
    this.tagName = tag.toUpperCase();
    this.document = document;
    this.children = [];
    this.listeners = new Map();
    this.value = "";
    this.textContent = "";
    this.className = "";
    this.hidden = false;
    this.disabled = false;
  }

  // Inputs coerce assignments to strings. Numeric conversion is deliberately small;
  // HTML number sanitisation, validity flags and constraint validation are not emulated.
  set value(value) { this.storedValue = String(value); }
  get value() { return this.storedValue; }
  set textContent(value) { this.storedText = String(value); this.children = []; }
  get textContent() { return this.storedText; }
  get valueAsNumber() { return this.value.trim() === "" ? NaN : Number(this.value); }
  append(...children) { this.children.push(...children); }
  focus() { this.document.activeElement = this; }

  querySelectorAll(selector) {
    assert.equal(selector, "input", "Only the selector used by the application is supported");
    const found = [];
    for (const child of this.children) {
      if (child.tagName === "INPUT") found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  // Return the actual registered handler's result so callers can await its promise.
  // There is no synthetic bubbling, native submit validation or automatic event dispatch.
  emit(type, event = {}) {
    const listeners = this.listeners.get(type) || [];
    assert.equal(listeners.length, 1, `Expected exactly one ${type} listener`);
    return listeners[0].call(this, event);
  }
}

function harness(t, initialResponse = reply(ready())) {
  const nodes = new Map();
  const document = {
    activeElement: null,
    getElementById(id) { assert.ok(nodes.has(id), `Unplanned DOM id: ${id}`); return nodes.get(id); },
    createElement(tag) { assert.ok(["label", "span", "input"].includes(tag)); return new Element(tag, document); }
  };
  for (const id of NODE_IDS) {
    assert.ok(html.includes(`id="${id}"`), `The HTML contract must contain ${id}`);
    const tag = id === "feature-form" ? "form" : ["predict", "load-example", "check-api"].includes(id) ? "button" : "div";
    const node = new Element(tag, document);
    node.id = id;
    nodes.set(id, node);
  }
  const get = id => nodes.get(id);
  get("prediction-result").hidden = true;
  get("form-message").hidden = true;
  get("field-count").textContent = "0 / 28";
  get("api-status").textContent = "Checking API…";
  get("api-status").className = "status";
  get("predict").textContent = "Get prediction →";
  get("feature-form").elements = {
    namedItem(name) {
      const input = get("feature-fields").querySelectorAll("input").find(node => node.name === name);
      assert.ok(input, `Missing named input: ${name}`);
      return input;
    }
  };
  const pendingResponses = [{path: "/api/health/ready", response: initialResponse}];
  const calls = [], signals = [], violations = [];
  const context = vm.createContext({
    document,
    AbortSignal: {
      timeout(milliseconds) {
        assert.equal(milliseconds, 35000);
        const signal = Object.freeze({fixtureSignal: signals.length, milliseconds});
        signals.push(signal);
        return signal; // An opaque sentinel creates no real timer or abort mechanism.
      }
    },
    fetch(path, options) {
      calls.push({path, options});
      const expected = pendingResponses.shift();
      try {
        assert.ok(expected, `Unplanned request: ${path}`);
        assert.equal(path, expected.path);
        assert.equal(options.cache, "no-store");
        assert.equal(options.signal, signals[calls.length - 1]);
      } catch (error) {
        violations.push(error.message);
        throw error;
      }
      return Promise.resolve(typeof expected.response === "function" ? expected.response() : expected.response);
    }
  });
  // app.js ends with checkApi(); its evaluation result is that real startup promise.
  const initial = application.runInContext(context);
  assert.equal(typeof initial.then, "function");
  t.after(() => {
    assert.deepEqual(violations, [], "No unexpected endpoint or request options may be swallowed by application catches");
    assert.equal(pendingResponses.length, 0, "Every planned response must be consumed");
  });
  return {
    context, document, initial, calls, get,
    input(name) { return get("feature-form").elements.namedItem(name); },
    inputs() { return get("feature-fields").querySelectorAll("input"); },
    fill(values = VALUES) { for (const name of FEATURE_NAMES) this.input(name).value = values[name]; },
    expect(path, response) { pendingResponses.push({path, response}); },
    change() { return get("feature-form").emit("input"); },
    example() { return get("load-example").emit("click"); },
    check() { return get("check-api").emit("click"); },
    submit() {
      this.lastSubmit = {defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }};
      return get("feature-form").emit("submit", this.lastSubmit);
    }
  };
}

function assertEmpty(h) {
  assert.equal(h.get("prediction-result").hidden, true);
  assert.equal(h.get("empty-result").hidden, false);
  for (const id of ["prediction-code", "model-hash", "contract-version"]) assert.equal(h.get(id).textContent, "");
}

function assertControls(h, disabled) {
  assert.equal(h.get("predict").disabled, disabled);
  assert.equal(h.get("load-example").disabled, disabled);
  for (const input of h.inputs()) assert.equal(input.disabled, disabled);
  assert.equal(h.get("predict").textContent, disabled ? "Requesting…" : "Get prediction →");
}

async function showPrediction(h, code = 1, values = VALUES) {
  h.fill(values);
  h.expect("/api/predict", reply(prediction(code)));
  await h.submit();
  assert.equal(h.get("prediction-result").hidden, false);
  assert.equal(h.get("prediction-code").textContent, String(code));
}

scenario("Creates all 28 ordered required numeric fields", "Check every generated field and caption against an independent ordered feature list.", async t => {
  const h = harness(t);
  await h.initial;
  assert.equal(h.get("feature-fields").children.length, 28);
  assert.deepEqual(h.inputs().map(input => input.name), FEATURE_NAMES);
  h.get("feature-fields").children.forEach((label, index) => {
    const name = FEATURE_NAMES[index];
    assert.equal(label.className, "feature");
    assert.equal(label.children.length, 2);
    assert.equal(label.children[0].tagName, "SPAN");
    const prefix = (index < 9 ? "0" : "") + (index + 1);
    assert.equal(label.children[0].textContent, `${prefix}  ${name}`);
    const input = label.children[1];
    assert.equal(input.tagName, "INPUT");
    assert.equal(input.id, `feature-${name}`);
    assert.equal(input.type, "number");
    assert.equal(input.required, true);
    assert.equal(input.step, name === "lat" || name === "long" ? "any" : "1");
    assert.equal(input.placeholder, name === "lat" || name === "long" ? "Number" : "Integer");
    assert.equal(input.value, "");
  });
  assert.equal(h.get("field-count").textContent, "0 / 28");
});

scenario("Input events update counts and clear stale predictions and messages", "Use actual submit and input handlers to remove a previous result and validation message.", async t => {
  const h = harness(t);
  await h.initial;
  await showPrediction(h);
  h.input("place").value = "";
  h.change();
  assertEmpty(h);
  assert.equal(h.get("field-count").textContent, "27 / 28");
  await h.submit();
  assert.equal(h.get("form-message").hidden, false);
  h.input("place").value = VALUES.place;
  h.change();
  assert.equal(h.get("field-count").textContent, "28 / 28");
  assert.equal(h.get("form-message").textContent, "");
  assert.equal(h.get("form-message").hidden, true);
  assertEmpty(h);
});

scenario("Startup readiness waits for a controlled response", "Observe the real startup promise without sleeping or scheduling a timeout.", async t => {
  const gate = deferred();
  const h = harness(t, gate.promise);
  assert.equal(h.calls.length, 1);
  assert.equal(h.get("check-api").disabled, true);
  assert.equal(h.get("api-status").textContent, "Checking API…");
  gate.resolve(reply(ready()));
  await h.initial;
  assert.equal(h.get("api-status").textContent, "API ready");
  assert.equal(h.get("api-status").className, "status ready");
  assert.equal(h.get("check-api").disabled, false);
});

scenario("Requested readiness can recover from an unavailable startup", "The registered check button handler restores itself and displays a newly valid identity.", async t => {
  const h = harness(t, reply({detail: "unavailable fixture"}, 503));
  await h.initial;
  assert.equal(h.get("api-status").textContent, "API unavailable");
  const gate = deferred();
  h.expect("/api/health/ready", gate.promise);
  const checking = h.check();
  assert.equal(h.get("check-api").disabled, true);
  gate.resolve(reply(ready()));
  await checking;
  assert.equal(h.get("api-status").textContent, "API ready");
  assert.equal(h.get("check-api").disabled, false);
});

const badReadiness = [
  ["an HTTP-unavailable API", () => reply({detail: "offline"}, 503)],
  ["a network rejection", () => () => Promise.reject(new Error("readiness offline"))],
  ["malformed JSON", () => invalidJson("readiness JSON failure")],
  ["a non-ready status", () => reply(ready({status: "loading"}))],
  ["a wrong contract version", () => reply(ready({contract_version: "other"}))],
  ["a short model hash", () => reply(ready({model_sha256: "abc"}))],
  ["an uppercase model hash", () => reply(ready({model_sha256: "A".repeat(64)}))],
  ["a missing model identity", () => reply({status: "ready", contract_version: CONTRACT})],
  ["a single-item model hash array", () => reply(ready({model_sha256: [HASH]}))],
  ["a null response contract", () => reply(null)]
];
for (const [name, makeResponse] of badReadiness) {
  scenario(`Readiness rejects ${name}`, "An unavailable or malformed readiness response must not display a healthy API.", async t => {
    const h = harness(t, makeResponse());
    await h.initial;
    assert.equal(h.get("api-status").textContent, "API unavailable");
    assert.equal(h.get("api-status").className, "status unavailable");
    assert.equal(h.get("check-api").disabled, false);
  });
}

scenario("A valid example populates all fields and clears the previous result", "Await the real example handler and verify string-coerced numeric assignments and button restoration.", async t => {
  const h = harness(t);
  await h.initial;
  await showPrediction(h);
  const values = {...VALUES, lat: -12.5, long: 6.75};
  h.expect("/api/example", reply({features: values}));
  const loading = h.example();
  assert.equal(h.get("load-example").disabled, true);
  assertEmpty(h);
  await loading;
  for (const name of FEATURE_NAMES) assert.equal(h.input(name).value, String(values[name]));
  assert.equal(h.get("field-count").textContent, "28 / 28");
  assert.equal(h.get("form-message").hidden, true);
  assert.equal(h.get("load-example").disabled, false);
});

const badExamples = [
  ["a missing feature", () => { const values = {...VALUES}; delete values.place; return reply({features: values}); }],
  ["an extra feature", () => reply({features: {...VALUES, unexpected: 1}})],
  ["a replaced required feature", () => { const values = {...VALUES, unexpected: 1}; delete values.place; return reply({features: values}); }],
  ["a string value", () => reply({features: {...VALUES, place: "1"}})],
  ["a non-finite value", () => reply({features: {...VALUES, lat: Infinity}})],
  ["a NaN value", () => reply({features: {...VALUES, long: NaN}})],
  ["a null feature object", () => reply({features: null})],
  ["a missing feature object", () => reply({})],
  ["an HTTP failure", () => reply({detail: "example unavailable"}, 503)],
  ["a network failure", () => () => Promise.reject(new Error("example offline"))],
  ["a JSON failure", () => invalidJson("example JSON failure")]
];
for (const [name, makeResponse] of badExamples) {
  scenario(`Example loading rejects ${name}`, "A failed example leaves existing field values intact, hides stale output and restores its button.", async t => {
    const h = harness(t);
    await h.initial;
    await showPrediction(h);
    h.expect("/api/example", makeResponse());
    await h.example();
    assertEmpty(h);
    assert.equal(h.get("load-example").disabled, false);
    assert.equal(h.get("form-message").hidden, false);
    assert.ok(h.get("form-message").textContent.length > 0);
    for (const name of FEATURE_NAMES) assert.equal(h.input(name).value, String(VALUES[name]));
  });
}

const invalidInputs = [
  ["an empty integer", "place", ""], ["a non-finite integer", "place", "Infinity"],
  ["a NaN integer", "place", "NaN"], ["a fractional category", "catv", "1.25"],
  ["an empty coordinate", "lat", ""], ["a non-finite coordinate", "long", "-Infinity"],
  ["a positive float32 overflow", "lat", "1e39"], ["a negative float32 overflow", "long", "-1e39"],
  ["an integer float32 overflow", "vma", "1e39"]
];
for (const [name, field, value] of invalidInputs) {
  scenario(`Submission rejects ${name} before prediction fetch`, "Invalid numeric state must focus its field, show a message and prevent the prediction request.", async t => {
    const h = harness(t);
    await h.initial;
    h.fill();
    h.input(field).value = value;
    await h.submit();
    assert.equal(h.lastSubmit.defaultPrevented, true);
    assert.equal(h.calls.length, 1, "Only startup readiness may have requested anything");
    assert.equal(h.document.activeElement, h.input(field));
    const kind = field === "lat" || field === "long" ? "number" : "integer";
    assert.equal(h.get("form-message").textContent, `Enter a valid ${kind} for ${field}.`);
    assert.equal(h.get("form-message").hidden, false);
    assertEmpty(h);
    assertControls(h, false);
  });
}

for (const code of [0, 1]) {
  scenario(`Prediction code ${code} preserves the exact endpoint, payload and displayed identity`, "Decimal coordinates and ordered independent feature values reach the POST and its displayed result.", async t => {
    const h = harness(t);
    await h.initial;
    await showPrediction(h, code);
    const call = h.calls[1];
    assert.equal(call.path, "/api/predict");
    assert.equal(call.options.method, "POST");
    assert.equal(call.options.headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(call.options.body), {features: VALUES});
    assert.deepEqual(Object.keys(JSON.parse(call.options.body).features), FEATURE_NAMES);
    assert.equal(h.get("empty-result").hidden, true);
    assert.equal(h.get("model-hash").textContent, HASH);
    assert.equal(h.get("contract-version").textContent, CONTRACT);
    assert.equal(h.get("form-message").hidden, true);
    assertControls(h, false);
  });
}

scenario("Exact positive and negative float32 coordinate bounds are accepted", "Derive the finite float32 boundary independently and check the inclusive request boundary.", async t => {
  const h = harness(t);
  await h.initial;
  const maximum = (2 ** 24 - 1) * 2 ** 104;
  const values = {...VALUES, lat: maximum, long: -maximum};
  await showPrediction(h, 0, values);
  assert.deepEqual(JSON.parse(h.calls[1].options.body), {features: values});
});

const failedPredictions = [
  ["an HTTP rejection", () => reply({detail: "fixture validation rejection"}, 422), "HTTP 422: fixture validation rejection"],
  ["an HTTP error without detail", () => reply({}, 503), "HTTP 503: Request failed."],
  ["a network rejection", () => () => Promise.reject(new Error("prediction offline")), "prediction offline"],
  ["invalid JSON", () => invalidJson("prediction JSON failure"), "prediction JSON failure"],
  ["an unsupported prediction code", () => reply(prediction(2))],
  ["a string prediction code", () => reply(prediction("0"))],
  ["a missing prediction code", () => reply({model_sha256: HASH, contract_version: CONTRACT})],
  ["a wrong contract version", () => reply(prediction(1, {contract_version: "other"}))],
  ["a missing model hash", () => reply({prediction_code: 1, contract_version: CONTRACT})],
  ["a non-hexadecimal model hash", () => reply(prediction(1, {model_sha256: "z".repeat(64)}))],
  ["an uppercase model hash", () => reply(prediction(1, {model_sha256: "A".repeat(64)}))],
  ["a single-item model hash array", () => reply(prediction(1, {model_sha256: [HASH]}))],
  ["a null response contract", () => reply(null)]
];
for (const [name, makeResponse, expectedMessage] of failedPredictions) {
  scenario(`Prediction failure handles ${name} and hides stale output`, "A failed response must restore all prediction controls and leave no old or malformed result visible.", async t => {
    const h = harness(t);
    await h.initial;
    await showPrediction(h);
    h.expect("/api/predict", makeResponse());
    const pending = h.submit();
    assertEmpty(h);
    assertControls(h, true);
    await pending;
    assertEmpty(h);
    assertControls(h, false);
    assert.equal(h.get("form-message").hidden, false);
    assert.ok(h.get("form-message").textContent.length > 0);
    if (expectedMessage) assert.equal(h.get("form-message").textContent, expectedMessage);
    else if (name !== "a null response contract") assert.equal(h.get("form-message").textContent, "Unexpected prediction contract.");
  });
}

scenario("Prediction controls stay disabled through fetch and JSON settlement", "Drive both asynchronous boundaries with promises and await an explicit JSON-start checkpoint.", async t => {
  const h = harness(t);
  await h.initial;
  await showPrediction(h);
  const fetchGate = deferred(), jsonGate = deferred(), jsonStarted = deferred();
  h.expect("/api/predict", fetchGate.promise);
  const pending = h.submit();
  assertControls(h, true);
  assertEmpty(h);
  fetchGate.resolve({ok: true, status: 200, json() { jsonStarted.resolve(); return jsonGate.promise; }});
  await jsonStarted.promise;
  assertControls(h, true);
  assertEmpty(h);
  jsonGate.resolve(prediction(0));
  await pending;
  assertControls(h, false);
  assert.equal(h.get("prediction-code").textContent, "0");
  assert.equal(h.get("prediction-result").hidden, false);
});

scenario("A pending prediction network failure restores controls and keeps output hidden", "Reject the controlled fetch promise only after observing the disabled state.", async t => {
  const h = harness(t);
  await h.initial;
  await showPrediction(h);
  const gate = deferred();
  h.expect("/api/predict", gate.promise);
  const pending = h.submit();
  assertControls(h, true);
  assertEmpty(h);
  gate.reject(new Error("controlled pending failure"));
  await pending;
  assertControls(h, false);
  assertEmpty(h);
  assert.equal(h.get("form-message").textContent, "controlled pending failure");
});
