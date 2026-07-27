importScripts("ort.min.js");

const MODEL_INPUT_SIZE = 640;
let session = null;

async function loadModel() {
  // Tell ORT to look for WASM files in the same folder
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create("best_fp16.onnx", {
    executionProviders: ["wasm"]
  });

  // Warm-up
  const dummy = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const warm = new ort.Tensor("float32", dummy, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  await session.run({ images: warm });

  postMessage({ type: "ready" });
}
loadModel();

// YOLO decode (light filters; we can tune later)
function decodeYOLO(rawData) {
  const numDetections = rawData.length / 84;
  const out = [];

  for (let i = 0; i < numDetections; i++) {
    const o = i * 84;

    // Assume model outputs pixel coords (0–640); normalize to 0–1
    let x = rawData[o + 0] / MODEL_INPUT_SIZE;
    let y = rawData[o + 1] / MODEL_INPUT_SIZE;
    let w = rawData[o + 2] / MODEL_INPUT_SIZE;
    let h = rawData[o + 3] / MODEL_INPUT_SIZE;
    let conf = rawData[o + 4];

    if (conf < 0.50) continue;

    // Basic size filters
    if (w > 0.5 || h > 0.5) continue;   // ignore huge blobs
    if (w < 0.01 || h < 0.01) continue; // ignore tiny noise

    const ratio = w / h;
    if (ratio < 0.5 || ratio > 1.8) continue; // allow some blur

    out.push({ x, y, w, h, confidence: conf });
  }

  return out;
}

onmessage = async (e) => {
  if (!session) return;

  const tensor = new ort.Tensor("float32", e.data, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const results = await session.run({ images: tensor });

  const key = Object.keys(results)[0];
  const raw = results[key].data;

  const detections = decodeYOLO(raw);

  postMessage({ type: "detections", detections });
};
