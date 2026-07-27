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

let latestDetections = [];
let frameCounter = 0;

// -------------------------------
// SETUP VIDEO + CANVAS
// -------------------------------
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const strikeSound = document.getElementById("strikeSound");

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
// LOAD MODEL + WARM-UP
// -------------------------------
let session;

async function loadModel() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create("best_fp16.onnx", {
    executionProviders: ["wasm"]
  });

  // warm-up
  const dummy = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const warmTensor = new ort.Tensor("float32", dummy, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  await session.run({ images: warmTensor });
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
// AI LOOP (frame skipping, async)
// -------------------------------
async function aiLoop() {
  if (!session || freezeActive) {
    setTimeout(aiLoop, INFERENCE_INTERVAL_MS);
    return;
  }

  frameCounter++;
  // process every 2nd frame to reduce load
  if (frameCounter % 2 !== 0) {
    setTimeout(aiLoop, INFERENCE_INTERVAL_MS);
    return;
  }

  const now = performance.now();

  const inputTensor = preprocessFrame();
  const feeds = { images: inputTensor };

  const results = await session.run(feeds);

  const outputKey = Object.keys(results)[0];
  const rawTensor = results[outputKey];
  const rawData = rawTensor.data;

  latestDetections = decodeYOLO(rawData);
  const hasBall = latestDetections.length > 0;

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
// STRIKE EVENT (sound + freeze)
// -------------------------------
function callStrike() {
  freezeActive = true;

  lastStrikeFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
  ctx.putImageData(lastStrikeFrame, 0, 0);

  // play baseball umpire strike call
  if (strikeSound) {
    strikeSound.currentTime = 0;
    strikeSound.play().catch(() => {});
  }

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
// DRAW LOOP (smooth video + boxes)
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

    // draw bounding boxes from latestDetections
    for (const det of latestDetections) {
      const { x, y, w, h, confidence } = det;

      // assuming YOLO coords are normalized 0–1
      const bx = x * canvas.width;
      const by = y * canvas.height;
      const bw = w * canvas.width;
      const bh = h * canvas.height;

      ctx.strokeStyle = "rgba(255,0,0,0.8)";
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = "rgba(255,0,0,0.8)";
      ctx.font = "16px Arial";
      ctx.fillText(`Ball ${confidence.toFixed(2)}`, bx, by - 5);
    }
  }

  requestAnimationFrame(drawLoop);
}
drawLoop();
