// Load ONNX Runtime Web locally
importScripts("ort.min.js");

const MODEL_INPUT_SIZE = 640;
let session = null;

async function loadModel() {
  try {
    // Load WASM from local folder
    ort.env.wasm.wasmPaths = "./";

    // Disable features requiring missing files
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;

    // Simple WASM backend
    session = await ort.InferenceSession.create("best_fp16.onnx", {
      executionProviders: ["wasm"]
    });

    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
}
loadModel();


// CORRECT DECODE FOR MODEL SHAPE [1, 5, 8400]
function decodeYOLO(rawData, dims) {
  const [batch, channels, count] = dims; // [1, 5, 8400]
  const out = [];

  for (let i = 0; i < count; i++) {
    const x_center = rawData[i * 5 + 0];
    const y_center = rawData[i * 5 + 1];
    const w = rawData[i * 5 + 2];
    const h = rawData[i * 5 + 3];
    const conf = rawData[i * 5 + 4];

    if (conf < 0.30) continue; // lower threshold for ball detection

    // Convert pixel coords → normalized 0–1
    const x = (x_center - w / 2) / MODEL_INPUT_SIZE;
    const y = (y_center - h / 2) / MODEL_INPUT_SIZE;
    const wn = w / MODEL_INPUT_SIZE;
    const hn = h / MODEL_INPUT_SIZE;

    // Sanity filters
    if (wn <= 0 || hn <= 0) continue;
    if (wn > 0.4 || hn > 0.4) continue; // ignore huge blobs

    out.push({ x, y, w: wn, h: hn, confidence: conf });
  }

  return out;
}


onmessage = async (e) => {
  if (!session) return;

  try {
    const tensor = new ort.Tensor("float32", e.data, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    const results = await session.run({ images: tensor });

    const key = Object.keys(results)[0];
    const output = results[key];

    const detections = decodeYOLO(output.data, output.dims);

    postMessage({ type: "detections", detections });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
};
