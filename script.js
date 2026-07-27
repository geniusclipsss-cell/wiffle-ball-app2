// -------------------------------
// CONFIG
// -------------------------------
const MIN_CONFIDENCE = 0.60;
const REQUIRED_FRAMES = 3;
const STRIKE_COOLDOWN = 2000;
const FREEZE_MS = 3000;

const INFERENCE_INTERVAL_MS = 250;
const MODEL_INPUT_SIZE = 640;

let lastStrikeTime = 0;
let consecutiveBallFrames = 0;
let freezeActive = false;
let lastStrikeFrame = null;

let strikeZoneWidthScale = 1.0;
let strikeZoneHeightScale = 1.0;

// -------------------------------
// SETUP VIDEO + CANVAS
// -------------------------------
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" }
  });
  video.srcObject = stream;
  await video.play();
}
setupCamera();

// -------------------------------
// LOAD MODEL
// -------------------------------
let session;

async function loadModel() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create("best_fp16.onnx", {
    executionProviders: ["wasm"]
  });
}
loadModel();

// -------------------------------
// PREPROCESS
// -------------------------------
function preprocessFrame() {
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = MODEL_INPUT_SIZE;
  tmpCanvas.height = MODEL_INPUT_SIZE;
  const tmpCtx = tmpCanvas.getContext("2d");

  tmpCtx.drawImage(video, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const { data } = tmpCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  const floatData = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  let idx = 0;

  for (let i = 0; i < data.length; i += 4) {
    floatData[idx++] = data[i] / 255;
    floatData[idx++] = data[i + 1] / 255;
    floatData[idx++] = data[i + 2] / 255;
  }

  return new ort.Tensor("float32", floatData, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
}

// -------------------------------
// YOLOv8 RAW OUTPUT DECODER
// -------------------------------
function decodeYOLO(rawData) {
  const numValues = rawData.length;

  // YOLOv8 output is usually [8400 * 84]
  const numDetections = numValues / 84;

  const detections = [];

  for (let i = 0; i < numDetections; i++) {
    const offset = i * 84;

    const x = rawData[offset + 0];
    const y = rawData[offset + 1];
    const w = rawData[offset + 2];
    const h = rawData[offset + 3];
    const confidence = rawData[offset + 4];

    if (confidence > MIN_CONFIDENCE) {
      detections.push({ x, y, w, h, confidence });
    }
  }

  return detections;
}

// -------------------------------
// AI LOOP
// -------------------------------
async function aiLoop() {
  if (!session || freezeActive) {
    setTimeout(aiLoop, INFERENCE_INTERVAL_MS);
    return;
  }

  const now = performance.now();

  const inputTensor = preprocessFrame();
  const feeds = { images: inputTensor };

  const results = await session.run(feeds);

  console.log(results); // keep for debugging

  const outputKey = Object.keys(results)[0];
  const rawTensor = results[outputKey];

  const rawData = rawTensor.data;

  const detections = decodeYOLO(rawData);
  const hasBall = detections.length > 0;

  if (hasBall) {
    consecutiveBallFrames++;
  } else {
    consecutiveBallFrames = 0;
  }

  if (consecutiveBallFrames >= REQUIRED_FRAMES) {
    const timeSinceLastStrike = now - lastStrikeTime;

    if (timeSinceLastStrike > STRIKE_COOLDOWN) {
      callStrike();
      lastStrikeTime = now;
    }

    consecutiveBallFrames = 0;
  }

  setTimeout(aiLoop, INFERENCE_INTERVAL_MS);
}
aiLoop();

// -------------------------------
// STRIKE EVENT
// -------------------------------
function callStrike() {
  freezeActive = true;

  lastStrikeFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
  ctx.putImageData(lastStrikeFrame, 0, 0);

  setTimeout(() => {
    freezeActive = false;
  }, FREEZE_MS);
}

// -------------------------------
// REPLAY
// -------------------------------
function replayStrike() {
  if (lastStrikeFrame) {
    freezeActive = true;
    ctx.putImageData(lastStrikeFrame, 0, 0);

    setTimeout(() => {
      freezeActive = false;
    }, FREEZE_MS);
  }
}
document.getElementById("replayButton").addEventListener("click", replayStrike);

// -------------------------------
// SLIDERS
// -------------------------------
document.getElementById("zoneWidthSlider").addEventListener("input", (e) => {
  strikeZoneWidthScale = parseFloat(e.target.value);
});
document.getElementById("zoneHeightSlider").addEventListener("input", (e) => {
  strikeZoneHeightScale = parseFloat(e.target.value);
});

// -------------------------------
// DRAW LOOP
// -------------------------------
function drawLoop() {
  if (!freezeActive) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const zoneWidth = canvas.width * 0.3 * strikeZoneWidthScale;
    const zoneHeight = canvas.height * 0.5 * strikeZoneHeightScale;
    const zoneX = (canvas.width - zoneWidth) / 2;
    const zoneY = (canvas.height - zoneHeight) / 2;

    ctx.strokeStyle = "rgba(0,255,0,0.7)";
    ctx.lineWidth = 3;
    ctx.strokeRect(zoneX, zoneY, zoneWidth, zoneHeight);
  }

  requestAnimationFrame(drawLoop);
}
drawLoop();
