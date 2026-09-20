// Runtime smoke test for index.html's inline <script>. node --check only parses syntax and
// misses undefined-reference bugs entirely (e.g. a silently no-op string replace that drops a
// const declaration but leaves its usages) -- exactly the bug this file exists to catch. This
// loads the real script into a stubbed DOM and actually CALLS the functions a user interaction
// would trigger (including firing the real tab buttons' onclick handlers), rather than
// just checking that the file parses.
//
//   node driftlab/viz/dashboard_check.js driftlab/viz/index.html
//
// Minimal runtime harness: execute the dashboard's <script> body with a stubbed DOM
// and actually CALL the functions a user interaction would trigger, so undefined-
// reference bugs (which node --check cannot see) get caught before shipping.
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];

function fakeEl(id) {
  const el = {
    id, textContent: '', innerHTML: '', className: '', style: {}, value: '',
    checked: false, children: [],
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener() {}, setAttribute() {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 600, height: 200 }), appendChild(c) { el.children.push(c); return c; },
    querySelector: () => null, querySelectorAll: () => [],
    closest: () => null, after: () => {}, focus() {},
    parentElement: null,
  };
  el.parentElement = { querySelector: () => null, appendChild() {} };
  return el;
}

const elements = {};
const document = {
  getElementById: (id) => (elements[id] ||= fakeEl(id)),
  querySelector: (sel) => null,
  querySelectorAll: (sel) => [],
  createElement: (tag) => fakeEl('created-' + tag),
  activeElement: null,
};
const window = { addEventListener() {}, innerWidth: 1400 };
class FakeEventSource { constructor() { this.onopen = null; this.onerror = null; this.onmessage = null; } }
class FakeResizeObserver { constructor(cb) { this.cb = cb; } observe() {} }
global.document = document;
global.window = window;
global.fetch = async () => ({ json: async () => [] });
global.EventSource = FakeEventSource;
global.ResizeObserver = FakeResizeObserver;
global.requestAnimationFrame = () => {};
global.setInterval = () => {};
global.setTimeout = (fn, ms) => { /* don't actually schedule in this check */ };
global.navigator = {};

const sandbox = { document, window, fetch: global.fetch, EventSource: FakeEventSource,
                  ResizeObserver: FakeResizeObserver, requestAnimationFrame: () => {},
                  setInterval: () => {}, setTimeout: () => {}, console, Math, Object, Array, JSON, Number, Set, Map };
vm.createContext(sandbox);

// Execute the IIFE to define render/applyTutorial/etc. Since the script is an IIFE that
// runs immediately, we need to intercept it: append a line that stashes the internal
// functions onto a global we can call afterward. Easiest: eval with a trailing debugger
// hook isn't available, so instead we rewrite the closing `})();` to expose what we need.
const exposed = js.replace(/\}\)\(\);\s*$/, `
  globalThis.__test = { applyTutorial, render, ingest, setTab, setTop, analysisHTML, mdHTML };
})();`);

try {
  vm.runInContext(exposed, sandbox, { filename: 'dashboard.js' });
} catch (e) {
  console.error('FAIL: script threw during initial execution:', e.message);
  process.exit(1);
}

try {
  sandbox.__test.applyTutorial();
  console.log('applyTutorial(): OK');
} catch (e) {
  console.error('FAIL: applyTutorial() threw:', e.message);
  process.exit(1);
}

try {
  // simulate the checkbox handler's effect directly (tutorial is a closured let, so
  // flip it via the same code path applyTutorial reads)
  sandbox.__test.render(); // should not throw even with no active run
  console.log('render() with no active run: OK');
} catch (e) {
  console.error('FAIL: render() threw:', e.message);
  process.exit(1);
}


// fire the actual tab button handlers, exactly as a real click would
try {
  if (typeof elements['tabPerformance'].onclick !== 'function') throw new Error('no onclick handler was attached to the tab buttons');
  elements['tabPerformance'].onclick();
  elements['tabConfig'].onclick();
  elements['tabActivity'].onclick();
  console.log('tab switching fired without throwing');
} catch (e) {
  console.error('FAIL: tab switch threw:', e.message);
  process.exit(1);
}
console.log('ALL RUNTIME CHECKS PASSED');

// deep check: feed a realistic run through ingest -> render -> tab switches, then
// assert the configuration tab rendered the agent profile and humanized labels.
try {
  const T = sandbox.__test;
  T.ingest({ run: "agy__changing__seed0", kind: "run_start", T: 12, experiment: "exp02", world: "rule_world",
             instructions: "You work on the intake desk.",
             cell: { world: "rule_world", regime: "changing", seed: 0,
                     agent: { name: "agy_cli", type: "harness_cli", harness: "antigravity", model: "gemini-3.5-flash-medium" } } });
  T.ingest({ run: "agy__changing__seed0", kind: "step", t: 0, T: 12, observation: "Request: a large refund.", reply: "CHOICE: A",
             action: "A", reward: 1, feedback: "Accepted.", key: ["refund", "north", "large"], correct: "A",
             changes: [{ t: 0, kind: "latent", desc: "rule flip", affected: [["refund", "north", "large"]] }] });
  T.ingest({ run: "agy__changing__seed0", kind: "event", event_kind: "memory", msg: "notebook updated", memory: "notes text", memory_kind: "notebook" });
  T.ingest({ run: "agy__changing__seed0", kind: "analysis", experiment: "exp02", text: "table" });
  T.render();
  for (const tabName of ["Performance", "Config", "Activity"]) T.setTab(tabName);
  const prof = elements["agentProf"].innerHTML;
  if (!prof.includes("antigravity") || !prof.includes("model")) throw new Error("agent profile rows missing: " + prof);
  if (!elements["progWrap"].title.includes("agy · changing · seed 0")) throw new Error("progress title not humanized: " + elements["progWrap"].title);
  if (elements["instr"].textContent !== "You work on the intake desk.") throw new Error("instructions missing");
  console.log("DEEP CHECKS PASSED: ingest/render/tabs/config with a realistic run");
} catch (e) { console.error("DEEP FAIL:", e.message); process.exit(1); }

// analysis formatting: analyze() text should become HTML tables, and the analysis
// should be stored per experiment; top-level view switching must not throw
try {
  const T = sandbox.__test;
  T.ingest({ run: "agy__changing__seed0", kind: "analysis", experiment: "exp02",
             text: "Lags after a change (lower is better)\nagent              n  detection_lag  recovery_lag\n--------------------------------------------------\nagy                8          0.857         2.833\n" });
  const html = T.analysisHTML("head a  head b\n----------\nrow1a  1.00\n");
  if (!html.includes("<table")) throw new Error("analysis not table-formatted: " + html.slice(0, 120));
  T.setTop("research");
  T.setTop("home");
  T.setTop("runs");
  const md = T.mdHTML("# H003 — Title\n\nStatus: UNTESTED\n\nA **bold** claim with `code`.\n\n## Evidence\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- item one\n- item two");
  for (const frag of ["<h3>", "<strong>bold</strong>", "<code>code</code>", "<table", "<li>item one</li>"])
    if (!md.includes(frag)) throw new Error("mdHTML missing " + frag + ": " + md.slice(0, 200));
  console.log("ANALYSIS + TOP-LEVEL VIEW CHECKS PASSED");
} catch (e) { console.error("ANALYSIS/TOPNAV FAIL:", e.message); process.exit(1); }
