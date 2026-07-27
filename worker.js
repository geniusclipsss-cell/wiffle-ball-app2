importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let session = null;
const MODEL_INPUT_SIZE = 640;

// Load model
async function loadModel() {
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

// YOLO decode
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

    if (conf < 0.60) continue;
    if (w > 0.2 || h > 0.2) continue;
    if (w < 0.02 || h < 0.02) continue;

    const ratio = w / h;
    if (ratio < 0.7 || ratio > 1.3) continue;

    out.push({ x, y, w, h, confidence: conf });
  }

  return out;
}

// Receive frames from main thread
onmessage = async (e) => {
  if (!session) return;

  const tensor = new ort.Tensor("float32", e.data, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const results = await session.run({ images: tensor });

  const key = Object.keys(results)[0];
  const raw = results[key].data;

  const detections = decodeYOLO(raw);

  postMessage({ type: "detections", detections });
};
