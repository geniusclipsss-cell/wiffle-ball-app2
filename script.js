// -------------------------------
// CONFIG
// -------------------------------
const MIN_CONFIDENCE = 0.60;
const REQUIRED_FRAMES = 3;
const STRIKE_COOLDOWN = 2000;
const FREEZE_MS = 3000;

let lastStrikeTime = 0;
let consecutiveBallFrames = 0;
let freezeActive = false;
let lastStrikeFrame = null;

let strikeZoneWidthScale = 1.0;
let strikeZoneHeightScale = 1.0;

// -------------------------------
// SETUP VIDEO
// -------------------------------
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" }
  });
  video.srcObject = stream;
  await video.play();
}

setupCamera();

// -------------------------------
// LOAD MODEL (WASM ONLY)
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
// IMAGE PREPROCESSING (FIXED)
// -------------------------------
function preprocessFrame() {
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = 640;
  tmpCanvas.height = 640;
  const tmpCtx = tmpCanvas.getContext("2d");

  tmpCtx.drawImage(video, 0, 0, 640, 640);
  const { data } = tmpCtx.getImageData(0, 0, 640, 640);

  // Convert RGBA → normalized RGB float32 → NCHW
  const floatData = new Float32Array(1 * 3 * 640 * 640);
  let idx = 0;

  for (let i = 0; i < data.length; i += 4) {
    floatData[idx++] = data[i] / 255;     // R
    floatData[idx++] = data[i + 1] / 255; // G
    floatData[idx++] = data[i + 2] / 255; // B
  }

  return new ort.Tensor("float32", floatData, [1, 3, 640, 640]);
}

// -------------------------------
// AI LOOP
// -------------------------------
async function aiLoop() {
  if (!session || freezeActive) {
    requestAnimationFrame(aiLoop);
    return;
  }

  const now = performance.now();

  const inputTensor = preprocessFrame();
  const feeds = { images: inputTensor };

  const results = await session.run(feeds);
  const detections = results.output.data;

  const balls = detections.filter(d => d.confidence > MIN_CONFIDENCE);
  const hasBall = balls.length > 0;

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

  requestAnimationFrame(aiLoop);
}

aiLoop();

// -------------------------------
// STRIKE EVENT + FREEZE FRAME
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
// REPLAY LAST STRIKE
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
// STRIKE ZONE SLIDERS
// -------------------------------
document.getElementById("zoneWidthSlider").addEventListener("input", (e) => {
  strikeZoneWidthScale = parseFloat(e.target.value);
});

document.getElementById("zoneHeightSlider").addEventListener("input", (e) => {
  strikeZoneHeightScale = parseFloat(e.target.value);
});

// -------------------------------
// VIDEO DRAW LOOP
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
