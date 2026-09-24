let currentAudioBlob = null;
let currentFileName = "";
let mediaRecorder = null;
let recordedChunks = [];
let recordInterval = null;
let recordStartTime = 0;

document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  setupDropZone();
  fetchHistory();
});

async function checkApiHealth() {
  const statusEl = document.getElementById("api-status");
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      statusEl.textContent = data.model_loaded ? "Model Ready" : "API Online";
      statusEl.className = "badge badge-online";
    } else {
      throw new Error();
    }
  } catch (err) {
    statusEl.textContent = "Offline";
    statusEl.className = "badge badge-offline";
  }
}

function setupDropZone() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });
}

function handleFileSelection(file) {
  currentAudioBlob = file;
  currentFileName = file.name;
  document.getElementById("selected-file-label").textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  
  // Set up audio player preview
  const player = document.getElementById("audio-player");
  player.src = URL.createObjectURL(file);
  document.getElementById("playback-container").style.display = "block";
  hideAlert();
}

async function loadDemoSample(filename, category) {
  try {
    document.getElementById("selected-file-label").textContent = `Loading demo sample: ${filename}...`;
    const res = await fetch(`/api/samples/${filename}`);
    if (!res.ok) throw new Error("Could not load demo sample");
    
    const blob = await res.blob();
    currentAudioBlob = new File([blob], filename, { type: "audio/wav" });
    currentFileName = filename;
    document.getElementById("machine-select").value = category;
    
    document.getElementById("selected-file-label").textContent = `Selected Demo: ${filename}`;
    const player = document.getElementById("audio-player");
    player.src = URL.createObjectURL(blob);
    document.getElementById("playback-container").style.display = "block";
    hideAlert();
  } catch (err) {
    showAlert("Failed to load demo sample: " + err.message);
  }
}

async function toggleRecording() {
  const btn = document.getElementById("btn-record");
  const timer = document.getElementById("record-timer");

  if (mediaRecorder && mediaRecorder.state === "recording") {
    // Stop Recording
    mediaRecorder.stop();
    btn.textContent = "🔴 Start Recording";
    btn.classList.remove("recording");
    clearInterval(recordInterval);
  } else {
    // Start Recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType || "audio/webm";
        currentAudioBlob = new Blob(recordedChunks, { type: mimeType });
        currentFileName = "browser_recording.webm";
        
        document.getElementById("selected-file-label").textContent = 
          `Recorded Audio Clip (${(currentAudioBlob.size / 1024).toFixed(1)} KB)`;
        
        const player = document.getElementById("audio-player");
        player.src = URL.createObjectURL(currentAudioBlob);
        document.getElementById("playback-container").style.display = "block";

        // Stop all audio tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      btn.textContent = "⏹️ Stop Recording";
      btn.classList.add("recording");
      recordStartTime = Date.now();
      timer.textContent = "00:00";

      recordInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const secs = String(elapsed % 60).padStart(2, "0");
        timer.textContent = `${mins}:${secs}`;
        if (elapsed >= 10) {
          toggleRecording(); // Enforce 10s maximum duration
        }
      }, 500);

      hideAlert();
    } catch (err) {
      showAlert("Microphone access denied or unavailable: " + err.message);
    }
  }
}

async function submitAnalysis() {
  if (!currentAudioBlob) {
    showAlert("Please select an audio file, choose a demo clip, or record via microphone first.");
    return;
  }

  const btn = document.getElementById("btn-submit");
  btn.disabled = true;
  btn.textContent = "⏳ Analyzing Acoustic Signatures...";
  hideAlert();

  const formData = new FormData();
  formData.append("file", currentAudioBlob, currentFileName);
  formData.append("machine_category", document.getElementById("machine-select").value);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      showAlert(`[${data.error_code || "ERROR"}] ${data.detail || "Analysis failed."}`);
      return;
    }

    displayResults(data);
    fetchHistory();
  } catch (err) {
    showAlert("Network or server connection failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze Acoustic Health";
  }
}

function displayResults(data) {
  const card = document.getElementById("results-card");
  card.style.display = "block";

  // Status Badge
  const badge = document.getElementById("status-badge");
  badge.textContent = data.predicted_class.toUpperCase();
  badge.className = "status-badge status-" + data.predicted_class;

  // Metadata
  document.getElementById("res-timestamp").textContent = `ID #${data.analysis_id || "N/A"} • ${new Date(data.timestamp).toLocaleTimeString()}`;
  document.getElementById("res-summary").textContent = data.result_summary;
  document.getElementById("res-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
  document.getElementById("res-latency").textContent = `Turnaround: ${data.execution_time_ms}ms`;

  // Probability Bars
  const pNorm = data.probabilities.normal || 0.0;
  const pAbnorm = data.probabilities.abnormal || 0.0;
  document.getElementById("bar-normal").style.width = `${(pNorm * 100).toFixed(1)}%`;
  document.getElementById("bar-abnormal").style.width = `${(pAbnorm * 100).toFixed(1)}%`;
  document.getElementById("prob-normal-val").textContent = `${(pNorm * 100).toFixed(1)}%`;
  document.getElementById("prob-abnormal-val").textContent = `${(pAbnorm * 100).toFixed(1)}%`;

  // Visualizations
  if (data.visualizations) {
    document.getElementById("img-waveform").src = data.visualizations.waveform_image;
    document.getElementById("img-spectrogram").src = data.visualizations.spectrogram_image;
  }

  // Smooth scroll to results
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function fetchHistory() {
  try {
    const res = await fetch("/api/history?limit=15");
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById("history-tbody");

    if (!data.records || data.records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No records logged yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.records.map(r => `
      <tr>
        <td>#${r.analysis_id}</td>
        <td>${new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
        <td><span style="text-transform: capitalize;">${r.machine_category}</span></td>
        <td style="max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${r.audio_reference}</td>
        <td><span class="badge status-${r.predicted_class}">${r.predicted_class}</span></td>
        <td>${(r.confidence * 100).toFixed(1)}%</td>
      </tr>
    `).join("");
  } catch (err) {
    console.warn("Could not fetch history:", err);
  }
}

function showAlert(msg) {
  const alertEl = document.getElementById("alert-banner");
  alertEl.textContent = msg;
  alertEl.style.display = "block";
}

function hideAlert() {
  const alertEl = document.getElementById("alert-banner");
  alertEl.style.display = "none";
}
