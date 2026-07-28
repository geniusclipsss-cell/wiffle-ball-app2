// Load ONNX Runtime Web locally
importScripts("ort.min.js");

const MODEL_INPUT_SIZE = 640;
let session = null;

async function loadModel() {
  try {
    // Load WASM from local folder
    ort.env.wasm.wasmPaths = "./";

    // Disable features that need extra files
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;

    // Simple WASM backend
    session = await ort.InferenceSession.create("best_fp16.onnx", {
      executionProviders: ["wasm"]
    });

    // Warm-up
    const dummy = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    const warm = new ort.Tensor("float32", dummy, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    await session.run({ images: warm });

    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
}
loadModel();

function decodeYOLO(rawData) {
  const numDetections = rawData.length / 84;
  const out = [];

  for (let i = 0; i < numDetections; i++) {
    const o = i * 84;

    let x = rawData[o + 0] / MODEL_INPUT_SIZE;
    let y = rawData[o + 1] / MODEL_INPUT_SIZE;
    let w = rawData[o + 2] / MODEL_INPUT_SIZE;
    let h = rawData[o + 3] / MODEL_INPUT_SIZE;
    let conf = rawData[o + 4];

    if (conf < 0.50) continue;
    if (w > 0.5 || h > 0.5) continue;
    if (w < 0.01 || h < 0.01) continue;

    const ratio = w / h;
    if (ratio < 0.5 || ratio > 1.8) continue;

    out.push({ x, y, w, h, confidence: conf });
  }

  return out;
}

onmessage = async (e) => {
  if (!session) return;

  try {
    const tensor = new ort.Tensor("float32", e.data, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    const results = await session.run({ images: tensor });

    const key = Object.keys(results)[0];
    const raw = results[key].data;

    const detections = decodeYOLO(raw);

    postMessage({ type: "detections", detections });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
};
