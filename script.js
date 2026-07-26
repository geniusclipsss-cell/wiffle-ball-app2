const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const strikeZoneDiv = document.getElementById("strike-zone");

const startBtn = document.getElementById("start-btn");
const statusEl = document.getElementById("status");
const ballsEl = document.getElementById("balls-count");
const strikesEl = document.getElementById("strikes-count");

let balls = 0;
let strikes = 0;
let session = null;
let running = false;

// Strike zone (normalized 0–1, same idea as Python script)
const STRIKE_ZONE = {
  x_min: 0.35,
  x_max: 0.65,
  y_min: 0.25,
  y_max: 0.55
};

const CLOSE_THRESHOLD = 0.05; // fraction of frame

async function loadModel() {
  statusEl.textContent = "Status: loading model...";
  // Adjust path if you put model in /models
  session = await ort.InferenceSession.create("best_fp16.onnx");
  statusEl.textContent = "Status: model loaded";
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, // back camera on mobile
      audio: false
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Position strike zone overlay
    const w = canvas.width;
    const h = canvas.height;
    const sx = STRIKE_ZONE.x_min * w;
    const sy = STRIKE_ZONE.y_min * h;
    const sw = (STRIKE_ZONE.x_max - STRIKE_ZONE.x_min) * w;
    const sh = (STRIKE_ZONE.y_max - STRIKE_ZONE.y_min) * h;

    strikeZoneDiv.style.left = `${sx}px`;
    strikeZoneDiv.style.top = `${sy}px`;
    strikeZoneDiv.style.width = `${sw}px`;
    strikeZoneDiv.style.height = `${sh}px`;

    running = true;
    statusEl.textContent = "Status: running";
    loop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Status: camera error";
  }
}

function preprocessFrame() {
  // Resize to model input size (e.g., 640x640) – adjust to your model
  const inputSize = 640;
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = inputSize;
  tmpCanvas.height = inputSize;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(video, 0, 0, inputSize, inputSize);

  const imageData = tmpCtx.getImageData(0, 0, inputSize, inputSize);
  const { data } = imageData;

  // Convert to float32, normalize 0–1, shape [1, 3, H, W]
  const floatData = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < inputSize * inputSize; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    floatData[i] = r;
    floatData[i + inputSize * inputSize] = g;
    floatData[i + 2 * inputSize * inputSize] = b;
  }

  const inputTensor = new ort.Tensor("float32", floatData, [1, 3, inputSize, inputSize]);
  return { inputTensor, inputSize };
}

function parseOutputs(outputs, frameWidth, frameHeight) {
  // This depends on your ONNX export.
  // For YOLOv8 ONNX, usually there's a single output with shape [1, N, 6] (x1,y1,x2,y2,score,class).
  const outputName = Object.keys(outputs)[0];
  const out = outputs[outputName].data;
  const numDet = outputs[outputName].dims[1];

  const boxes = [];
  for (let i = 0; i < numDet; i++) {
    const base = i * 6;
    const x1 = out[base + 0] * frameWidth;
    const y1 = out[base + 1] * frameHeight;
    const x2 = out[base + 2] * frameWidth;
    const y2 = out[base + 3] * frameHeight;
    const score = out[base + 4];
    const cls = out[base + 5];

    if (score < 0.25) continue; // confidence threshold
    // If you have multiple classes, you can check cls here.
    boxes.push({ x1, y1, x2, y2, score, cls });
  }
  return boxes;
}

async function loop() {
  if (!running || !session) return;

  const { inputTensor } = preprocessFrame();

  try {
    const feeds = { images: inputTensor }; // adjust key name to your model input
    const outputs = await session.run(feeds);

    // Draw original frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const boxes = parseOutputs(outputs, canvas.width, canvas.height);

    boxes.forEach((box) => {
      const { x1, y1, x2, y2 } = box;
      const w = canvas.width;
      const h = canvas.height;

      const cx = (x1 + x2) / 2 / w;
      const cy = (y1 + y2) / 2 / h;
      const bw = (x2 - x1) / w;
      const bh = (y2 - y1) / h;

      const ballIsClose = bw > CLOSE_THRESHOLD || bh > CLOSE_THRESHOLD;
      const insideStrikeZone =
        STRIKE_ZONE.x_min <= cx &&
        cx <= STRIKE_ZONE.x_max &&
        STRIKE_ZONE.y_min <= cy &&
        cy <= STRIKE_ZONE.y_max;

      let call, color;
      if (ballIsClose && insideStrikeZone) {
        call = "STRIKE";
        strikes += 1;
        color = "lime";
      } else {
        call = "BALL";
        balls += 1;
        color = "red";
      }

      // Draw box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label
      ctx.fillStyle = color;
      ctx.font = "16px sans-serif";
      ctx.fillText(call, x1 + 4, y1 - 4);
    });

    ballsEl.textContent = balls;
    strikesEl.textContent = strikes;
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Status: inference error";
  }

  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", async () => {
  if (!session) {
    await loadModel();
  }
  await startCamera();
});
