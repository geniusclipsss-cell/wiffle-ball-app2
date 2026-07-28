importScripts("ort.min.js");

const MODEL_INPUT_SIZE = 640;
let session = null;

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "./";
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;

    session = await ort.InferenceSession.create("best_fp16.onnx", {
      executionProviders: ["wasm"]
    });

    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
}
loadModel();

function decodeYOLO(rawData, dims) {
  const [batch, channels, count] = dims;
  const out = [];

  for (let i = 0; i < count; i++) {
    const x_center = rawData[i * 5 + 0];
    const y_center = rawData[i * 5 + 1];
    const w = rawData[i * 5 + 2];
    const h = rawData[i * 5 + 3];
    const conf = rawData[i * 5 + 4];

    if (conf < 0.40) continue;

    // Ball size sanity (wiffle ball is small)
    if (w < 10 || h < 10) continue;        // too small = noise
    if (w > 120 || h > 120) continue;      // too big = background

    // Convert to normalized
    const x = (x_center - w / 2) / MODEL_INPUT_SIZE;
    const y = (y_center - h / 2) / MODEL_INPUT_SIZE;
    const wn = w / MODEL_INPUT_SIZE;
    const hn = h / MODEL_INPUT_SIZE;

    // Position sanity
    if (x < 0 || y < 0 || x > 1 || y > 1) continue;
    if (wn <= 0 || hn <= 0) continue;

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
