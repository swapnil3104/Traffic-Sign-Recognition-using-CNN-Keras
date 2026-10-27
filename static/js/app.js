/**
 * Traffic Sign Recognition - Frontend Controller with Voice Module & Theme Switcher
 * Provides deep learning inference, audio explanation, webcam capture, and catalog search.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ==========================================
    // State & Storage
    // ==========================================
    let currentImageFile = null;
    let currentImageBase64 = null;
    let cameraStream = null;
    let catalogData = [];
    let activeCategory = 'ALL';
    let currentPrediction = null;

    // Voice State
    let speechRate = 1.0;
    let autoSpeak = true;
    let isSpeaking = false;

    // Load saved preferences
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    const savedAutoSpeak = localStorage.getItem('auto-speak');
    if (savedAutoSpeak !== null) {
        autoSpeak = savedAutoSpeak === 'true';
    }

    // ==========================================
    // DOM Elements
    // ==========================================
    // Theme
    const themeToggleBtn = document.getElementById('themeToggleBtn');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabViews = document.querySelectorAll('.tab-view');

    // Upload & Dropzone
    const dropArea = document.getElementById('dropArea');
    const fileSelector = document.getElementById('fileSelector');
    const dropPlaceholder = document.getElementById('dropPlaceholder');
    const previewWrapper = document.getElementById('previewWrapper');
    const previewImage = document.getElementById('previewImage');
    const btnRemoveImage = document.getElementById('btnRemoveImage');

    // Camera
    const webcamVideo = document.getElementById('webcamVideo');
    const webcamCanvas = document.getElementById('webcamCanvas');
    const btnStartCamera = document.getElementById('btnStartCamera');
    const btnCaptureCamera = document.getElementById('btnCaptureCamera');
    const btnStopCamera = document.getElementById('btnStopCamera');

    // Samples & Actions
    const samplesGrid = document.getElementById('samplesGrid');
    const btnClassify = document.getElementById('btnClassify');
    const classifySpinner = document.getElementById('classifySpinner');
    const btnReset = document.getElementById('btnReset');

    // Results Panel
    const statusPill = document.getElementById('statusPill');
    const emptyState = document.getElementById('emptyState');
    const resultDetails = document.getElementById('resultDetails');
    const resIcon = document.getElementById('resIcon');
    const resClassId = document.getElementById('resClassId');
    const resCategory = document.getElementById('resCategory');
    const resConfidenceBadge = document.getElementById('resConfidenceBadge');
    const resSignName = document.getElementById('resSignName');
    const resRisk = document.getElementById('resRisk');
    const barsList = document.getElementById('barsList');
    const resLatency = document.getElementById('resLatency');

    // Voice Module Elements
    const btnPlayVoice = document.getElementById('btnPlayVoice');
    const voiceBtnText = document.getElementById('voiceBtnText');
    const voicePulseIcon = document.getElementById('voicePulseIcon');
    const voiceStatusLabel = document.getElementById('voiceStatusLabel');
    const audioWaveform = document.getElementById('audioWaveform');
    const voiceTranscriptText = document.getElementById('voiceTranscriptText');
    const autoSpeakCheckbox = document.getElementById('autoSpeakCheckbox');
    const speedBtns = document.querySelectorAll('.speed-btn');

    // Catalog
    const catalogCardsGrid = document.getElementById('catalogCardsGrid');
    const catalogSearchInput = document.getElementById('catalogSearchInput');
    const categoryTabs = document.getElementById('categoryTabs');
    const appToast = document.getElementById('appToast');
    const toastText = document.getElementById('toastText');

    // Initialize Auto-speak checkbox
    if (autoSpeakCheckbox) {
        autoSpeakCheckbox.checked = autoSpeak;
        autoSpeakCheckbox.addEventListener('change', (e) => {
            autoSpeak = e.target.checked;
            localStorage.setItem('auto-speak', autoSpeak);
            showToast(autoSpeak ? 'Auto-voice guidance enabled' : 'Auto-voice guidance disabled');
        });
    }

    // ==========================================
    // Theme Switcher
    // ==========================================
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        showToast(`Switched to ${newTheme === 'light' ? 'White' : 'Dark'} Theme`);
    });

    // ==========================================
    // Voice Module (Text-to-Speech)
    // ==========================================
    function getBestVoice() {
        if (!('speechSynthesis' in window)) return null;
        const voices = window.speechSynthesis.getVoices();
        // Prefer natural English voices if available
        return voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Premium'))) ||
               voices.find(v => v.lang.startsWith('en')) ||
               voices[0] || null;
    }

    function speakSignInformation(prediction) {
        if (!('speechSynthesis' in window) || !prediction) return;

        // Cancel any active speech
        stopSpeaking();

        // Prepare clear, informative speech text
        const speechText = `Traffic sign identified: ${prediction.name}. Category: ${prediction.category}. Road directive: ${prediction.action}`;

        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.rate = speechRate;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        const bestVoice = getBestVoice();
        if (bestVoice) {
            utterance.voice = bestVoice;
        }

        utterance.onstart = () => {
            isSpeaking = true;
            updateVoiceUI(true, `Reading: ${prediction.name}`);
        };

        utterance.onend = () => {
            isSpeaking = false;
            updateVoiceUI(false, 'Voice explanation finished');
        };

        utterance.onerror = (e) => {
            console.warn('Speech error:', e);
            isSpeaking = false;
            updateVoiceUI(false, 'Voice explanation ready');
        };

        window.speechSynthesis.speak(utterance);
    }

    function stopSpeaking() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        isSpeaking = false;
        updateVoiceUI(false, 'Voice explanation ready');
    }

    function updateVoiceUI(active, statusText) {
        if (active) {
            voicePulseIcon.classList.add('active');
            audioWaveform.classList.remove('hidden');
            btnPlayVoice.classList.add('speaking');
            btnPlayVoice.querySelector('.play-icon').classList.add('hidden');
            btnPlayVoice.querySelector('.stop-icon').classList.remove('hidden');
            voiceBtnText.textContent = 'Stop Audio';
            voiceStatusLabel.textContent = statusText;
        } else {
            voicePulseIcon.classList.remove('active');
            audioWaveform.classList.add('hidden');
            btnPlayVoice.classList.remove('speaking');
            btnPlayVoice.querySelector('.play-icon').classList.remove('hidden');
            btnPlayVoice.querySelector('.stop-icon').classList.add('hidden');
            voiceBtnText.textContent = 'Speak Info';
            voiceStatusLabel.textContent = statusText;
        }
    }

    btnPlayVoice.addEventListener('click', () => {
        if (isSpeaking) {
            stopSpeaking();
        } else if (currentPrediction) {
            speakSignInformation(currentPrediction);
        } else {
            showToast('No sign recognized yet to read.');
        }
    });

    // Speed selector
    speedBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            speedBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            speechRate = parseFloat(btn.dataset.rate) || 1.0;
            if (isSpeaking && currentPrediction) {
                speakSignInformation(currentPrediction);
            }
        });
    });

    // ==========================================
    // Tab Navigation
    // ==========================================
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            tabViews.forEach(v => v.classList.remove('active'));

            btn.classList.add('active');
            if (tab === 'upload') document.getElementById('viewUpload').classList.add('active');
            if (tab === 'samples') document.getElementById('viewSamples').classList.add('active');
            if (tab === 'camera') document.getElementById('viewCamera').classList.add('active');

            if (tab !== 'camera' && cameraStream) {
                stopCamera();
            }
        });
    });

    // ==========================================
    // Drag & Drop / File Upload
    // ==========================================
    ['dragenter', 'dragover'].forEach(name => {
        dropArea.addEventListener(name, (e) => {
            e.preventDefault();
            dropArea.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropArea.addEventListener(name, (e) => {
            e.preventDefault();
            dropArea.classList.remove('drag-over');
        });
    });

    dropArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files[0] && files[0].type.startsWith('image/')) {
            loadImage(files[0]);
        }
    });

    dropArea.addEventListener('click', (e) => {
        if (e.target.closest('#btnRemoveImage')) return;
        fileSelector.click();
    });

    fileSelector.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            loadImage(e.target.files[0]);
        }
    });

    function loadImage(file) {
        currentImageFile = file;
        currentImageBase64 = null;

        const reader = new FileReader();
        reader.onload = (evt) => {
            previewImage.src = evt.target.result;
            dropPlaceholder.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
            btnClassify.disabled = false;
            statusPill.textContent = 'Image Ready';
            statusPill.className = 'status-pill ready';
            
            // Auto run classification for immediate response
            executeInference();
        };
        reader.readAsDataURL(file);
    }

    btnRemoveImage.addEventListener('click', (e) => {
        e.stopPropagation();
        resetInput();
    });

    btnReset.addEventListener('click', () => {
        resetInput();
        resetResults();
        stopSpeaking();
    });

    function resetInput() {
        currentImageFile = null;
        currentImageBase64 = null;
        fileSelector.value = '';
        previewImage.src = '';
        dropPlaceholder.classList.remove('hidden');
        previewWrapper.classList.add('hidden');
        btnClassify.disabled = true;
    }

    function resetResults() {
        emptyState.classList.remove('hidden');
        resultDetails.classList.add('hidden');
        statusPill.textContent = 'Standby';
        statusPill.className = 'status-pill';
        currentPrediction = null;
    }

    // ==========================================
    // Camera Stream
    // ==========================================
    btnStartCamera.addEventListener('click', async () => {
        try {
            btnStartCamera.textContent = 'Connecting...';
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
            });
            webcamVideo.srcObject = cameraStream;
            btnStartCamera.classList.add('hidden');
            btnCaptureCamera.classList.remove('hidden');
            btnStopCamera.classList.remove('hidden');
            showToast('Camera active');
        } catch (err) {
            console.error('Camera error:', err);
            btnStartCamera.textContent = 'Start Camera';
            showToast('Failed to access camera: ' + err.message);
        }
    });

    btnStopCamera.addEventListener('click', stopCamera);

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
            webcamVideo.srcObject = null;
            btnStartCamera.classList.remove('hidden');
            btnStartCamera.textContent = 'Start Camera';
            btnCaptureCamera.classList.add('hidden');
            btnStopCamera.classList.add('hidden');
        }
    }

    btnCaptureCamera.addEventListener('click', () => {
        if (!cameraStream) return;
        const video = webcamVideo;
        const size = Math.min(video.videoWidth, video.videoHeight);
        const startX = (video.videoWidth - size) / 2;
        const startY = (video.videoHeight - size) / 2;

        webcamCanvas.width = 300;
        webcamCanvas.height = 300;
        const ctx = webcamCanvas.getContext('2d');
        ctx.drawImage(video, startX, startY, size, size, 0, 0, 300, 300);

        const dataUrl = webcamCanvas.toDataURL('image/jpeg', 0.92);
        currentImageBase64 = dataUrl;
        currentImageFile = null;

        previewImage.src = dataUrl;
        dropPlaceholder.classList.add('hidden');
        previewWrapper.classList.remove('hidden');
        btnClassify.disabled = false;

        document.getElementById('tabUpload').click();
        executeInference();
    });

    // ==========================================
    // Test Bench Samples
    // ==========================================
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                renderSamples(data.samples);
            }
        } catch (err) {
            console.error('Failed to load samples:', err);
            samplesGrid.innerHTML = '<div class="empty-loading">Unable to load test samples.</div>';
        }
    }

    function renderSamples(samples) {
        samplesGrid.innerHTML = '';
        samples.forEach(sample => {
            const tile = document.createElement('div');
            tile.className = 'sample-tile';
            tile.title = `${sample.name} (Class #${sample.class_id})`;
            tile.innerHTML = `
                <img src="${sample.image_url}" alt="${sample.name}" loading="lazy">
                <span class="sample-name">${sample.name}</span>
            `;
            tile.addEventListener('click', async () => {
                try {
                    const res = await fetch(sample.image_url);
                    const blob = await res.blob();
                    const file = new File([blob], sample.filename, { type: blob.type || 'image/png' });
                    document.getElementById('tabUpload').click();
                    loadImage(file);
                } catch (e) {
                    console.error('Error loading sample image:', e);
                }
            });
            samplesGrid.appendChild(tile);
        });
    }

    // ==========================================
    // Model Inference Execution
    // ==========================================
    btnClassify.addEventListener('click', executeInference);

    async function executeInference() {
        if (!currentImageFile && !currentImageBase64) return;

        btnClassify.disabled = true;
        classifySpinner.classList.remove('hidden');
        statusPill.textContent = 'Evaluating...';

        const startTime = performance.now();

        try {
            let res;
            if (currentImageFile) {
                const fd = new FormData();
                fd.append('file', currentImageFile);
                res = await fetch('/api/predict', { method: 'POST', body: fd });
            } else if (currentImageBase64) {
                res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImageBase64 })
                });
            }

            const data = await res.json();
            const latencyMs = Math.round(performance.now() - startTime);

            if (data.success && data.prediction) {
                currentPrediction = data.prediction;
                displayResults(data.prediction, data.top_5, latencyMs);

                // Auto-voice trigger
                if (autoSpeak) {
                    speakSignInformation(data.prediction);
                }
            } else {
                throw new Error(data.error || 'Prediction request failed');
            }
        } catch (err) {
            console.error('Prediction failed:', err);
            showToast('Classification error: ' + err.message);
            statusPill.textContent = 'Error';
        } finally {
            btnClassify.disabled = false;
            classifySpinner.classList.add('hidden');
        }
    }

    function displayResults(prediction, top5, latency) {
        emptyState.classList.add('hidden');
        resultDetails.classList.remove('hidden');

        statusPill.textContent = 'Classified';
        statusPill.className = 'status-pill ready';

        // Primary Prediction Card
        resIcon.src = prediction.icon_url;
        resClassId.textContent = `Class #${prediction.class_id}`;
        resCategory.textContent = prediction.category;
        resConfidenceBadge.textContent = `${prediction.confidence_percent}% Match`;
        resSignName.textContent = prediction.name;
        resRisk.textContent = prediction.action;

        // Voice Transcript Preview
        voiceTranscriptText.textContent = `${prediction.name} (${prediction.category}): ${prediction.action}`;
        voiceStatusLabel.textContent = 'Information ready to speak';

        // Top-5 Probabilities
        renderProbabilities(top5);

        // Telemetry
        resLatency.textContent = `${latency} ms`;
    }

    function renderProbabilities(top5) {
        barsList.innerHTML = '';
        top5.forEach((item, index) => {
            const bar = document.createElement('div');
            bar.className = 'bar-item';
            bar.innerHTML = `
                <img class="bar-thumb" src="${item.icon_url}" alt="${item.name}">
                <div class="bar-info">
                    <div class="bar-meta">
                        <span class="bar-title">${index + 1}. ${item.name}</span>
                        <span class="bar-score">${item.confidence_percent}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
            barsList.appendChild(bar);

            // Animate fill bar
            setTimeout(() => {
                const fill = bar.querySelector('.bar-fill');
                if (fill) fill.style.width = `${item.confidence_percent}%`;
            }, 60 * index + 50);
        });
    }

    // ==========================================
    // Class Catalog & Search
    // ==========================================
    async function loadCatalog() {
        try {
            const res = await fetch('/api/classes');
            const data = await res.json();
            if (data.classes) {
                catalogData = data.classes;
                renderCatalog(catalogData);
            }
        } catch (err) {
            console.error('Failed to load classes catalog:', err);
            catalogCardsGrid.innerHTML = '<div class="empty-loading">Unable to load catalog.</div>';
        }
    }

    function renderCatalog(items) {
        catalogCardsGrid.innerHTML = '';
        if (items.length === 0) {
            catalogCardsGrid.innerHTML = '<div class="empty-loading">No traffic signs match your search filter.</div>';
            return;
        }

        items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'catalog-entry';
            card.innerHTML = `
                <img class="entry-icon" src="${item.icon_url}" alt="${item.name}" loading="lazy">
                <div class="entry-details">
                    <div class="entry-header">
                        <span class="entry-id">#${item.class_id}</span>
                        <span class="entry-cat">${item.category}</span>
                    </div>
                    <div class="entry-name" title="${item.name}">${item.name}</div>
                    <div class="entry-action" title="${item.action}">${item.action}</div>
                </div>
            `;

            card.addEventListener('click', async () => {
                try {
                    const res = await fetch(item.icon_url);
                    const blob = await res.blob();
                    const file = new File([blob], `${item.class_id}.png`, { type: 'image/png' });
                    document.getElementById('tabUpload').click();
                    loadImage(file);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    showToast(`Loaded ${item.name} into classifier`);
                } catch (e) {
                    console.error('Error loading sign icon:', e);
                }
            });

            catalogCardsGrid.appendChild(card);
        });
    }

    categoryTabs.addEventListener('click', (e) => {
        const pill = e.target.closest('.cat-pill');
        if (!pill) return;
        categoryTabs.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');

        activeCategory = pill.dataset.cat;
        applyFilters();
    });

    catalogSearchInput.addEventListener('input', applyFilters);

    function applyFilters() {
        const query = catalogSearchInput.value.trim().toLowerCase();
        const filtered = catalogData.filter(item => {
            const matchesCategory = (activeCategory === 'ALL' || item.category.toLowerCase().includes(activeCategory.toLowerCase()));
            const matchesQuery = query === '' ||
                item.name.toLowerCase().includes(query) ||
                item.category.toLowerCase().includes(query) ||
                item.action.toLowerCase().includes(query) ||
                item.class_id.toString() === query;
            return matchesCategory && matchesQuery;
        });
        renderCatalog(filtered);
    }

    // ==========================================
    // Toast Notification
    // ==========================================
    let toastTimer = null;
    function showToast(msg) {
        toastText.textContent = msg;
        appToast.classList.remove('hidden');
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            appToast.classList.add('hidden');
        }, 2800);
    }

    // Initial Load
    loadSamples().then(() => {
        // Query param handlers for demo / screenshots
        const params = new URLSearchParams(window.location.search);
        if (params.get('theme') === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else if (params.get('theme') === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        }

        if (params.get('tab') === 'samples') {
            const tabSamples = document.getElementById('tabSamples');
            if (tabSamples) tabSamples.click();
        }

        const demoSign = params.get('demo');
        if (demoSign) {
            // Pick sample by name or default to stop sign (sample_14_00093.png)
            const sampleUrl = demoSign === 'stop' || demoSign === '14'
                ? '/static/samples/sample_14_00093.png'
                : '/static/samples/sample_1_00001.png';
            fetch(sampleUrl)
                .then(r => r.blob())
                .then(blob => {
                    const file = new File([blob], 'demo_sample.png', { type: 'image/png' });
                    loadImage(file);
                })
                .catch(err => console.error('Demo sample load error:', err));
        }

        if (params.get('scroll') === 'matrix' || params.get('scroll') === 'catalog') {
            setTimeout(() => {
                const el = document.querySelector('.catalog-section');
                if (el) el.scrollIntoView({ behavior: 'instant' });
            }, 500);
        }
    });
    loadCatalog();
});
