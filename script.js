// -------------------------------
// CONFIG
// -------------------------------
const MIN_CONFIDENCE = 0.60;       // confidence threshold
const REQUIRED_FRAMES = 3;         // persistence requirement
const STRIKE_COOLDOWN = 2000;      // strike cooldown
const FREEZE_MS = 3000;            // freeze frame duration (3 seconds)

let lastStrikeTime = 0;
let consecutiveBallFrames = 0;
let freezeActive = false;
let lastStrikeFrame = null;

// Strike zone sliders (default scale = 1.0)
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
// LOAD MODEL
// -------------------------------
let session;

async function loadModel() {
  session = await ort.InferenceSession.create("best_fp16.onnx", {
    executionProviders: ["webgl"]
  });
}

loadModel();

// -------------------------------
// AI LOOP (runs every 120ms)
// -------------------------------
async function aiLoop() {
  if (!session || freezeActive) {
    requestAnimationFrame(aiLoop);
    return;
  }

  const now = performance.now();

  // Resize frame for inference
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = 640;
  tmpCanvas.height = 640;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(video, 0, 0, 640, 640);

  const imageData = tmpCtx.getImageData(0, 0, 640, 640);
  const inputTensor = new ort.Tensor("float32", imageData.data, [1, 640, 640, 4]);

  const feeds = { images: inputTensor };
  const results = await session.run(feeds);
  const detections = results.output.data;

  // -------------------------------
  // FILTER DETECTIONS
  // -------------------------------
  const balls = detections.filter(d => d.confidence > MIN_CONFIDENCE);
  const hasBall = balls.length > 0;

  // -------------------------------
  // PERSISTENCE CHECK
  // -------------------------------
  if (hasBall) {
    consecutiveBallFrames++;
  } else {
    consecutiveBallFrames = 0;
  }

  // -------------------------------
  // STRIKE LOGIC
  // -------------------------------
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
  console.log("STRIKE!");

  freezeActive = true;

  // Save freeze frame for replay
  lastStrikeFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Draw freeze frame
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

// Attach replay button
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
// VIDEO DRAW LOOP (always smooth)
// -------------------------------
function drawLoop() {
  if (!freezeActive) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw strike zone using slider scales
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
