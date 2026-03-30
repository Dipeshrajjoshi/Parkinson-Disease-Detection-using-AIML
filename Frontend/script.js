const BACKEND_URL = "http://127.0.0.1:5000";

// Demo values from first row of parkinsons.csv (status=1, Parkinson's)
const DEMO_VALUES = [
  119.992, 157.302, 74.997,          // Fo, Fhi, Flo
  0.00784, 0.00007, 0.00370, 0.00554, 0.01109, // Jitter
  0.04374, 0.42600, 0.02182, 0.03130, 0.02971, 0.06545, // Shimmer
  0.02211, 21.033,                   // NHR, HNR
  0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654 // Nonlinear
];

function loadDemoValues() {
  const inputs = document.querySelectorAll('.feat-input');
  inputs.forEach((inp, i) => {
    inp.value = DEMO_VALUES[i];
    inp.style.borderColor = 'rgba(139, 92, 246, 0.6)';
    setTimeout(() => inp.style.borderColor = '', 800);
  });
}

function clearResult() {
  const panel = document.getElementById('result-panel');
  panel.style.display = 'none';
  panel.className = 'result-panel';
  document.getElementById('graphs-section').style.display = 'none';
}

function getFeatures() {
  const inputs = document.querySelectorAll('.feat-input');
  const values = [];
  let valid = true;
  inputs.forEach(inp => {
    const v = parseFloat(inp.value);
    if (inp.value.trim() === '' || isNaN(v)) {
      inp.style.borderColor = '#ef4444';
      inp.style.boxShadow = '0 0 0 3px rgba(239,68,68,0.15)';
      valid = false;
    } else {
      inp.style.borderColor = '';
      inp.style.boxShadow = '';
      values.push(v);
    }
  });
  return valid ? values : null;
}

function getPatient() {
  return {
    name: document.getElementById('patientName').value.trim() || 'N/A',
    age: document.getElementById('patientAge').value.trim() || 'N/A',
    address: document.getElementById('patientAddress').value.trim() || 'N/A',
    doctor: document.getElementById('doctorName').value.trim() || 'N/A',
  };
}

document.getElementById('predict-form').addEventListener('submit', async function (e) {
  e.preventDefault();
  const features = getFeatures();
  if (!features) {
    return;
  }

  // Loading state
  const btn = document.getElementById('predict-btn');
  const loader = document.getElementById('btn-loader');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Predicting…';
  loader.style.display = 'inline-block';

  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Server error');
    }

    const data = await res.json();
    showResult(data, getPatient());
  } catch (err) {
    alert(` Error: ${err.message}\n\nMake sure the Flask server is running:\n  cd backend && python app.py`);
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Run Prediction';
    loader.style.display = 'none';
  }
});

function showResult(data, patient) {
  const panel = document.getElementById('result-panel');
  const isHealthy = data.prediction === 0;

  // Classes
  panel.className = `result-panel ${isHealthy ? 'state-healthy' : 'state-parkinsons'}`;

  // Icon & label
  document.getElementById('result-icon').textContent = isHealthy ? '✅' : '⚠️';
  const lbl = document.getElementById('result-label');
  lbl.textContent = data.result;
  lbl.className = `result-label ${isHealthy ? 'healthy' : 'parkinsons'}`;

  // Patient info
  document.getElementById('result-patient').textContent =
    `Patient: ${patient.name}${patient.age !== 'N/A' ? ', Age ' + patient.age : ''} · Dr. ${patient.doctor}`;

  // Confidence bars — animate after display
  panel.style.display = 'block';
  setTimeout(() => {
    document.getElementById('bar-healthy').style.width = `${data.prob_healthy}%`;
    document.getElementById('bar-park').style.width = `${data.prob_parkinsons}%`;
    document.getElementById('pct-healthy').textContent = `${data.prob_healthy}%`;
    document.getElementById('pct-park').textContent = `${data.prob_parkinsons}%`;
  }, 50);

  // Meta
  const now = new Date().toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
  document.getElementById('result-meta').textContent =
    `📍 ${patient.address} · 🕒 ${now} · Confidence: ${data.confidence}%`;

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function generateAndShowGraphs() {
  const graphsSection = document.getElementById('graphs-section');
  graphsSection.style.display = 'block';

  try {
    const res = await fetch(`${BACKEND_URL}/generate-plots`, { method: 'POST' });
    const result = await res.json();
    if (result.status === 'success') {
      const ts = Date.now(); // cache-bust
      document.getElementById('confusion-img').src = `${BACKEND_URL}/images/confusion.png?t=${ts}`;
      document.getElementById('importance-img').src = `${BACKEND_URL}/images/importance.png?t=${ts}`;
      graphsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      alert('Graph generation failed: ' + (result.error || 'Unknown error'));
    }
  } catch (err) {
    alert(` Could not load graphs: ${err.message}`);
  }
}
