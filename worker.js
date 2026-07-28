// Load ONNX Runtime Web locally
importScripts("ort.min.js");

const MODEL_INPUT_SIZE = 640;
let session = null;

async function loadModel() {
  try {
    // Load WASM from local folder
    ort.env.wasm.wasmPaths = "./";

    // Disable features that need extra helper files
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
    const warmResults = await session.run({ images: warm });

    const warmKey = Object.keys(warmResults)[0];
    console.log("Warm-up output keys:", Object.keys(warmResults));
    console.log("Warm-up output dims:", warmResults[warmKey].dims);
    console.log("Warm-up first 20 values:", warmResults[warmKey].data.slice(0, 20));

    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
}
loadModel();

// TEMP decode: just log raw output so we can see what the model is doing
function decodeYOLO(rawData, dims) {
  console.log("Model output dims:", dims);
  console.log("First 50 raw values:", rawData.slice(0, 50));

  // For now, return no detections until we understand the format
  return [];
}

onmessage = async (e) => {
  if (!session) return;

  try {
    const tensor = new ort.Tensor("float32", e.data, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    const results = await session.run({ images: tensor });

    const key = Object.keys(results)[0];
    const output = results[key];

    const raw = output.data;
    const dims = output.dims;

    const detections = decodeYOLO(raw, dims);

    postMessage({ type: "detections", detections });
  } catch (err) {
    postMessage({ type: "error", error: err.message || String(err) });
  }
};
