const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const strikeSound = document.getElementById("strikeSound");

const MODEL_INPUT_SIZE = 640;

let latestDetections = [];
let freezeActive = false;
let lastStrikeFrame = null;
let lastStrikeTime = 0;
let consecutiveBallFrames = 0;

let strikeZoneWidthScale = 1.0;
let strikeZoneHeightScale = 1.0;

const worker = new Worker("worker.js");

// Handle messages from worker
worker.onmessage = (e) => {
  if (e.data.type === "ready") {
    console.log("Worker ready, model loaded.");
    return;
  }

  if (e.data.type === "error") {
    console.error("Worker error:", e.data.error);
    return;
  }

  if (e.data.type === "detections") {
    latestDetections = e.data.detections;

    if (latestDetections.length > 0) {
      consecutiveBallFrames++;
    } else {
      consecutiveBallFrames = 0;
    }

    const now = performance.now();
    if (consecutiveBallFrames >= 3 && now - lastStrikeTime > 2000) {
      callStrike();
      lastStrikeTime = now;
      consecutiveBallFrames = 0;
    }
  }
};

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" }
  });
  video.srcObject = stream;
  await video.play();
}
setupCamera();

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

function getStrikeZone() {
  const zoneWidth = canvas.width * 0.3 * strikeZoneWidthScale;
  const zoneHeight = canvas.height * 0.5 * strikeZoneHeightScale;
  const zoneX = (canvas.width - zoneWidth) / 2;
  const zoneY = (canvas.height - zoneHeight) / 2;
  return { zoneX, zoneY, zoneWidth, zoneHeight };
}

function preprocessFrame() {
  const tmp = document.createElement("canvas");
  tmp.width = MODEL_INPUT_SIZE;
  tmp.height = MODEL_INPUT_SIZE;
  const tctx = tmp.getContext("2d");

  tctx.drawImage(video, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const { data } = tctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  const out = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  let idx = 0;

  for (let i = 0; i < data.length; i += 4) {
    out[idx++] = data[i] / 255;
    out[idx++] = data[i + 1] / 255;
    out[idx++] = data[i + 2] / 255;
  }

  return out;
}

// AI loop → send frames to worker
setInterval(() => {
  if (!freezeActive && video.readyState >= 2) {
    const frame = preprocessFrame();
    worker.postMessage(frame);
  }
}, 200);

// Strike event
function callStrike() {
  freezeActive = true;

  lastStrikeFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
  ctx.putImageData(lastStrikeFrame, 0, 0);

  if (strikeSound) {
    strikeSound.currentTime = 0;
    strikeSound.play().catch(() => {});
  }

  setTimeout(() => {
    freezeActive = false;
  }, 3000);
}

// Replay
document.getElementById("replayButton").addEventListener("click", () => {
  if (lastStrikeFrame) {
    freezeActive = true;
    ctx.putImageData(lastStrikeFrame, 0, 0);
    setTimeout(() => {
      freezeActive = false;
    }, 3000);
  }
});

// Sliders
document.getElementById("zoneWidthSlider").addEventListener("input", (e) => {
  strikeZoneWidthScale = parseFloat(e.target.value);
});
document.getElementById("zoneHeightSlider").addEventListener("input", (e) => {
  strikeZoneHeightScale = parseFloat(e.target.value);
});

// Draw loop
function drawLoop() {
  if (!freezeActive) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const { zoneX, zoneY, zoneWidth, zoneHeight } = getStrikeZone();

    ctx.strokeStyle = "rgba(0,255,0,0.7)";
    ctx.lineWidth = 3;
    ctx.strokeRect(zoneX, zoneY, zoneWidth, zoneHeight);

    for (const det of latestDetections) {
      const bx = det.x * canvas.width;
      const by = det.y * canvas.height;
      const bw = det.w * canvas.width;
      const bh = det.h * canvas.height;

      ctx.strokeStyle = "rgba(255,0,0,0.8)";
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = "rgba(255,0,0,0.8)";
      ctx.font = "16px Arial";
      ctx.fillText(`Ball ${det.confidence.toFixed(2)}`, bx, by - 5);
    }
  }

  requestAnimationFrame(drawLoop);
}
drawLoop();
