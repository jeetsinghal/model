// AgriVision AI — Client Application & Prognostic Analytics Engine

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Input & Scanning
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const clearPreviewBtn = document.getElementById("clearPreviewBtn");
    const scanLaser = document.getElementById("scanLaser");
    const analyzeBtn = document.getElementById("analyzeBtn");

    // Quick Test Gallery & Filters
    const samplesContainer = document.getElementById("samplesContainer");
    const samplesCount = document.getElementById("samplesCount");
    const cropFilterBtns = document.querySelectorAll(".crop-filter-btn");

    // Results & Prognosis Panel
    const emptyState = document.getElementById("emptyState");
    const resultsContent = document.getElementById("resultsContent");
    const cropNamePill = document.getElementById("cropNamePill");
    const categoryPill = document.getElementById("categoryPill");
    const severityPill = document.getElementById("severityPill");
    const diagnosisTitle = document.getElementById("diagnosisTitle");
    const pathogenInfo = document.getElementById("pathogenInfo");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");

    // Future Risk Banner
    const futureRiskBanner = document.getElementById("futureRiskBanner");
    const riskPercentNumber = document.getElementById("riskPercentNumber");
    const riskBarFill = document.getElementById("riskBarFill");
    const contagionBadge = document.getElementById("contagionBadge");
    const neighborRiskText = document.getElementById("neighborRiskText");

    // Future Prognosis Tab
    const yieldLossRiskText = document.getElementById("yieldLossRiskText");
    const economicImpactText = document.getElementById("economicImpactText");
    const timelineContainer = document.getElementById("timelineContainer");
    const contagionRadiusText = document.getElementById("contagionRadiusText");

    // 7-Day Cure Roadmap Tab
    const roadmapGrid = document.getElementById("roadmapGrid");

    // Clinical Tabs
    const symptomsText = document.getElementById("symptomsText");
    const envFactorsText = document.getElementById("envFactorsText");
    const immediateActionText = document.getElementById("immediateActionText");
    const organicList = document.getElementById("organicList");
    const chemicalList = document.getElementById("chemicalList");
    const nutritionalText = document.getElementById("nutritionalText");
    const preventionText = document.getElementById("preventionText");
    const candidatesList = document.getElementById("candidatesList");

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Top Navigation & Modal
    const modelStatusPill = document.getElementById("modelStatusPill");
    const modelStatusText = document.getElementById("modelStatusText");
    const trainingPulseDot = document.getElementById("trainingPulseDot");
    const hardwarePillText = document.getElementById("hardwarePillText");
    const openTrainingModalBtn = document.getElementById("openTrainingModalBtn");
    const closeTrainingModalBtn = document.getElementById("closeTrainingModalBtn");
    const closeModalBottomBtn = document.getElementById("closeModalBottomBtn");
    const trainingModal = document.getElementById("trainingModal");

    // Modal KPIs & Chart Elements
    const kpiBestValAcc = document.getElementById("kpiBestValAcc");
    const kpiTargetEpochs = document.getElementById("kpiTargetEpochs");
    const kpiCurrentEpoch = document.getElementById("kpiCurrentEpoch");
    const kpiDevice = document.getElementById("kpiDevice");
    const liveTrainingBanner = document.getElementById("liveTrainingBanner");
    const liveTrainingStatusText = document.getElementById("liveTrainingStatusText");
    const liveProgressBar = document.getElementById("liveProgressBar");
    const convergenceSvg = document.getElementById("convergenceSvg");
    const historyTableBody = document.getElementById("historyTableBody");
    const startTrainingBtn = document.getElementById("startTrainingBtn");

    // Export Buttons & Toast
    const printReportBtn = document.getElementById("printReportBtn");
    const copySummaryBtn = document.getElementById("copySummaryBtn");
    const toast = document.getElementById("toast");

    let currentFile = null;
    let currentSamplePath = null;
    let latestAnalysisData = null;
    let allSamplesList = [];
    let activeCropFilter = "all";
    let trainingPollInterval = null;

    // Initialize App
    pollTrainingStatus();
    loadSamples();

    // ----------------------------------------------------
    // Model Status & Training Analytics Polling
    // ----------------------------------------------------
    function pollTrainingStatus() {
        fetch("/api/training_status")
            .then(res => res.json())
            .then(data => {
                updateTrainingUI(data);
            })
            .catch(err => {
                console.error("Training status fetch error:", err);
            });
    }

    function updateTrainingUI(data) {
        const historyData = data.history_data || {};
        const historyList = historyData.history || [];
        const isTraining = data.is_training === true;
        const totalTrained = historyData.total_epochs || historyList.length || 0;
        const bestAcc = historyData.best_val_acc ? `${historyData.best_val_acc}%` : "90.67%";

        // Hardware device
        if (data.device) {
            hardwarePillText.textContent = `${data.device.toUpperCase()} Accelerated`;
            kpiDevice.textContent = data.device.toUpperCase();
        }

        // Top Navigation Pill
        if (isTraining) {
            modelStatusText.textContent = `Training Epoch ${data.current_epoch || 1}/10 (${data.epoch_progress_pct || 0}%)`;
            trainingPulseDot.style.background = "#f59e0b";
            trainingPulseDot.style.boxShadow = "0 0 10px #f59e0b";
            liveTrainingBanner.classList.remove("hidden");
            liveTrainingStatusText.textContent = data.status_text || `Training Epoch ${data.current_epoch}/10...`;
            liveProgressBar.style.width = `${data.epoch_progress_pct || 10}%`;
            startTrainingBtn.disabled = true;
            startTrainingBtn.innerHTML = `
                <span class="spinner-sm"></span>
                <span>Training in Progress (${data.current_epoch}/10)...</span>
            `;
        } else {
            modelStatusText.textContent = `10-Epoch Vision: Active (${bestAcc})`;
            trainingPulseDot.style.background = "var(--primary-400)";
            trainingPulseDot.style.boxShadow = "0 0 10px var(--primary-400)";
            liveTrainingBanner.classList.add("hidden");
            startTrainingBtn.disabled = false;
            startTrainingBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                <span>${totalTrained >= 10 ? 'Re-Train 10 Epochs' : 'Train / Resume 10 Epochs'}</span>
            `;
        }

        // Modal KPIs
        kpiBestValAcc.textContent = bestAcc;
        kpiCurrentEpoch.textContent = `${totalTrained} / 10 Completed`;

        // Render Chart and Table
        renderConvergenceChart(historyList);
        renderHistoryTable(historyList, data.current_epoch, isTraining);
    }

    // Modal Listeners
    openTrainingModalBtn.addEventListener("click", () => {
        trainingModal.classList.remove("hidden");
        pollTrainingStatus();
        if (!trainingPollInterval) {
            trainingPollInterval = setInterval(pollTrainingStatus, 3000);
        }
    });

    function closeTrainingModal() {
        trainingModal.classList.add("hidden");
        if (trainingPollInterval) {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
        }
    }

    closeTrainingModalBtn.addEventListener("click", closeTrainingModal);
    closeModalBottomBtn.addEventListener("click", closeTrainingModal);
    trainingModal.addEventListener("click", (e) => {
        if (e.target === trainingModal) closeTrainingModal();
    });

    // Start Training Trigger
    startTrainingBtn.addEventListener("click", () => {
        startTrainingBtn.disabled = true;
        startTrainingBtn.innerHTML = `
            <span class="spinner-sm"></span>
            <span>Initializing 10-Epoch Pipeline...</span>
        `;
        showToast("Initiating 10-Epoch Training on MPS hardware...");

        fetch("/api/start_training", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs: 10 })
        })
        .then(res => res.json())
        .then(data => {
            showToast(data.message || "Training started successfully!");
            setTimeout(pollTrainingStatus, 1000);
        })
        .catch(err => {
            showToast(`Training error: ${err.message}`);
            startTrainingBtn.disabled = false;
        });
    });

    // ----------------------------------------------------
    // Dynamic SVG Convergence Graph Generator
    // ----------------------------------------------------
    function renderConvergenceChart(history) {
        if (!history || history.length === 0) {
            convergenceSvg.innerHTML = `<text x="350" y="120" text-anchor="middle" fill="#64748b" font-size="14">No training history available yet.</text>`;
            return;
        }

        const width = 700;
        const height = 240;
        const padding = { top: 20, right: 30, bottom: 30, left: 45 };
        const plotW = width - padding.left - padding.right;
        const plotH = height - padding.top - padding.bottom;

        const maxEpochs = Math.max(10, history.length);
        const getX = (epoch) => padding.left + ((epoch - 1) / (maxEpochs - 1)) * plotW;
        
        // Acc range: 50% to 100%
        const getYAcc = (acc) => padding.top + plotH - ((Math.min(100, Math.max(50, acc)) - 50) / 50) * plotH;
        // Loss range: 0.0 to 2.0
        const getYLoss = (loss) => padding.top + plotH - ((Math.min(2.0, Math.max(0.0, loss)) - 0) / 2.0) * plotH;

        let gridLines = "";
        // Horizontal grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (plotH / 4) * i;
            const accVal = 100 - i * 12.5;
            gridLines += `
                <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3 3"/>
                <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" fill="#64748b" font-size="10">${accVal}%</text>
            `;
        }

        // Vertical epoch lines
        for (let ep = 1; ep <= maxEpochs; ep++) {
            const x = getX(ep);
            gridLines += `
                <line x1="${x}" y1="${padding.top}" x2="${x}" y2="${padding.top + plotH}" stroke="rgba(255,255,255,0.04)"/>
                <text x="${x}" y="${height - 10}" text-anchor="middle" fill="#64748b" font-size="10">E${ep}</text>
            `;
        }

        // Build path data
        const valAccPoints = history.map(h => `${getX(h.epoch)},${getYAcc(h.val_acc)}`).join(" ");
        const trainAccPoints = history.map(h => `${getX(h.epoch)},${getYAcc(h.train_acc)}`).join(" ");
        const valLossPoints = history.map(h => `${getX(h.epoch)},${getYLoss(h.val_loss)}`).join(" ");
        const trainLossPoints = history.map(h => `${getX(h.epoch)},${getYLoss(h.train_loss)}`).join(" ");

        let dots = "";
        history.forEach(h => {
            const vx = getX(h.epoch);
            const vy = getYAcc(h.val_acc);
            dots += `
                <circle cx="${vx}" cy="${vy}" r="4" fill="#10b981" stroke="#04120c" stroke-width="2">
                    <title>Epoch ${h.epoch}: Val Acc ${h.val_acc}%</title>
                </circle>
            `;
        });

        convergenceSvg.innerHTML = `
            ${gridLines}
            <polyline fill="none" stroke="#a855f7" stroke-width="1.8" stroke-dasharray="4 2" points="${trainLossPoints}"/>
            <polyline fill="none" stroke="#f59e0b" stroke-width="2" points="${valLossPoints}"/>
            <polyline fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="4 2" points="${trainAccPoints}"/>
            <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="${valAccPoints}"/>
            ${dots}
        `;
    }

    function renderHistoryTable(history, currentEpoch, isTraining) {
        historyTableBody.innerHTML = "";
        (history || []).forEach(h => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>Epoch ${h.epoch}</strong></td>
                <td>${h.train_loss}</td>
                <td>${h.train_acc}%</td>
                <td>${h.val_loss}</td>
                <td class="highlight">${h.val_acc}%</td>
                <td>${h.duration_sec}s</td>
                <td><span class="severity-pill healthy">Completed</span></td>
            `;
            historyTableBody.appendChild(tr);
        });

        if (isTraining && currentEpoch) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>Epoch ${currentEpoch}</strong></td>
                <td colspan="4"><span class="spinner-sm"></span> Processing active epoch batches on MPS...</td>
                <td>--</td>
                <td><span class="severity-pill moderate">Active</span></td>
            `;
            historyTableBody.appendChild(tr);
        }
    }

    // ----------------------------------------------------
    // Quick Test Sample Gallery & Crop Filtering
    // ----------------------------------------------------
    function loadSamples() {
        fetch("/api/samples")
            .then(res => res.json())
            .then(data => {
                allSamplesList = data.samples || [];
                renderFilteredSamples();
            })
            .catch(err => {
                samplesCount.textContent = "Error loading";
                console.error("Samples error:", err);
            });
    }

    function renderFilteredSamples() {
        const filtered = activeCropFilter === "all" 
            ? allSamplesList 
            : allSamplesList.filter(s => s.crop.toLowerCase().includes(activeCropFilter.toLowerCase()));

        samplesCount.textContent = `${filtered.length} specimens`;
        samplesContainer.innerHTML = "";

        filtered.forEach(sample => {
            const card = document.createElement("div");
            card.className = "sample-card";
            card.innerHTML = `
                <img src="${sample.image_url}" alt="${sample.display_name}" class="sample-thumb" loading="lazy">
                <span class="sample-crop-tag">${sample.crop}</span>
                <span class="sample-name" title="${sample.display_name}">${sample.display_name}</span>
            `;
            card.addEventListener("click", () => {
                selectSample(sample, card);
            });
            samplesContainer.appendChild(card);
        });
    }

    // Crop Filter Tab Listeners
    cropFilterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            cropFilterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeCropFilter = btn.getAttribute("data-filter");
            renderFilteredSamples();
        });
    });

    function selectSample(sample, cardElement) {
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
        if (cardElement) cardElement.classList.add("active");

        currentFile = null;
        currentSamplePath = sample.sample_path;
        fileInput.value = "";

        imagePreview.src = sample.image_url;
        dropzonePrompt.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        analyzeBtn.disabled = false;

        runAnalysis();
    }

    // ----------------------------------------------------
    // Drag & Drop / File Upload Listeners
    // ----------------------------------------------------
    dropZone.addEventListener("click", (e) => {
        if (e.target !== clearPreviewBtn && !clearPreviewBtn.contains(e.target)) {
            if (previewContainer.classList.contains("hidden")) {
                fileInput.click();
            }
        }
    });

    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFileSelect(dt.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            showToast("Please upload a valid image file (JPG, PNG, WEBP).");
            return;
        }

        currentFile = file;
        currentSamplePath = null;
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearPreviewBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        currentFile = null;
        currentSamplePath = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        analyzeBtn.disabled = true;
        scanLaser.classList.remove("scanning");
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    });

    // ----------------------------------------------------
    // Run Neural Inference & Prognostic Analysis
    // ----------------------------------------------------
    analyzeBtn.addEventListener("click", runAnalysis);

    function runAnalysis() {
        if (!currentFile && !currentSamplePath) return;

        scanLaser.classList.add("scanning");
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `
            <span class="spinner-sm"></span>
            <span>Running Deep Neural Analysis & Prognosis...</span>
        `;

        const formData = new FormData();
        if (currentFile) {
            formData.append("file", currentFile);
        } else if (currentSamplePath) {
            formData.append("sample_path", currentSamplePath);
        }

        fetch("/api/predict", {
            method: "POST",
            body: formData
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(data => { throw new Error(data.error || "Analysis failed"); });
            }
            return res.json();
        })
        .then(data => {
            renderResults(data);
            showToast("Complete diagnostic & future prognosis generated!");
        })
        .catch(err => {
            console.error("Analysis Error:", err);
            showToast(`Error: ${err.message}`);
        })
        .finally(() => {
            scanLaser.classList.remove("scanning");
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                <span>Re-Run Neural Analysis</span>
            `;
        });
    }

    // ----------------------------------------------------
    // Render Results, Future Prognosis & 7-Day Cure Roadmap
    // ----------------------------------------------------
    function renderResults(data) {
        latestAnalysisData = data;
        const agro = data.agronomic_analysis;
        const future = agro.future_prediction || {};
        const isHealthy = (agro.severity || "").toLowerCase().includes("healthy") || (future.loss_percentage === 0);

        emptyState.classList.add("hidden");
        resultsContent.classList.remove("hidden");

        // Primary Hero Information
        cropNamePill.textContent = agro.crop || "Crop";
        categoryPill.textContent = agro.category || "Condition";
        diagnosisTitle.textContent = agro.display_name || data.prediction;
        
        if (agro.pathogen && agro.pathogen !== "N/A") {
            pathogenInfo.innerHTML = `Pathogen/Causal Agent: <em>${agro.pathogen}</em>`;
            pathogenInfo.classList.remove("hidden");
        } else {
            pathogenInfo.classList.add("hidden");
        }

        // Severity Pill
        const sev = (agro.severity || "Moderate").toLowerCase();
        severityPill.textContent = `Severity: ${agro.severity}`;
        severityPill.className = "severity-pill";
        if (sev.includes("critical")) severityPill.classList.add("critical");
        else if (sev.includes("high")) severityPill.classList.add("high");
        else if (sev.includes("healthy")) severityPill.classList.add("healthy");
        else severityPill.classList.add("moderate");

        // Confidence Bar
        confidenceValue.textContent = `${data.confidence}%`;
        setTimeout(() => {
            confidenceBar.style.width = `${data.confidence}%`;
        }, 50);

        // Future Risk Banner
        const lossPct = future.loss_percentage !== undefined ? future.loss_percentage : 50;
        riskPercentNumber.textContent = isHealthy ? "0% (Optimal)" : `${lossPct}%`;
        riskBarFill.style.width = `${lossPct}%`;
        if (isHealthy) {
            futureRiskBanner.classList.add("healthy-banner");
        } else {
            futureRiskBanner.classList.remove("healthy-banner");
        }

        contagionBadge.textContent = future.contagion_radius || "Standard Environmental Spread";
        neighborRiskText.textContent = future.neighbor_plot_risk || "Low transmission risk";

        // Tab: Future Prognosis
        yieldLossRiskText.textContent = future.yield_loss_risk || "No severe yield reduction expected.";
        economicImpactText.textContent = future.economic_impact || "Standard market quality expected.";
        contagionRadiusText.textContent = future.contagion_radius || "Local plot transmission.";

        // Stage-by-Stage Untreated Timeline
        timelineContainer.innerHTML = "";
        const timelineList = future.progression_timeline || [];
        timelineList.forEach(stage => {
            const card = document.createElement("div");
            const riskClass = (stage.risk_level || "moderate").toLowerCase();
            card.className = `timeline-stage-card risk-${riskClass}`;
            card.innerHTML = `
                <div class="timeline-stage-top">
                    <span class="timeline-phase-title">${stage.phase}</span>
                    <span class="timeline-risk-tag ${riskClass}">${stage.risk_level} Risk</span>
                </div>
                <p class="timeline-symptoms-text">${stage.symptoms}</p>
                <span class="timeline-loss-badge">Impact: ${stage.loss_trajectory}</span>
            `;
            timelineContainer.appendChild(card);
        });

        // Tab: 7-Day Cure Roadmap
        roadmapGrid.innerHTML = "";
        const roadmapList = agro.cure_roadmap || [];
        roadmapList.forEach(step => {
            const card = document.createElement("div");
            card.className = "roadmap-step-card";
            card.innerHTML = `
                <div class="roadmap-badge">${step.day}</div>
                <p class="roadmap-action-text">${step.action}</p>
            `;
            roadmapGrid.appendChild(card);
        });

        // Tab: Symptoms & Triggers
        symptomsText.textContent = agro.symptoms || "No specific symptoms reported.";
        envFactorsText.textContent = agro.environmental_factors || "Standard climatic conditions.";
        immediateActionText.textContent = agro.immediate_action || "Continue routine field monitoring.";

        // Tab: Organic List
        organicList.innerHTML = "";
        (agro.organic_control || []).forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            organicList.appendChild(li);
        });

        // Tab: Chemical List
        chemicalList.innerHTML = "";
        (agro.chemical_control || []).forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            chemicalList.appendChild(li);
        });

        // Tab: Nutrition & Soil
        nutritionalText.textContent = agro.nutritional_recovery || "Maintain standard balanced NPK fertilization.";

        // Tab: Prevention Text
        preventionText.textContent = agro.preventive_measures || "Adhere to recommended crop rotation and sanitation.";

        // Differential Candidates Breakdown
        candidatesList.innerHTML = "";
        (data.top_candidates || []).forEach(cand => {
            const row = document.createElement("div");
            row.className = "candidate-row";
            row.innerHTML = `
                <span class="candidate-name" title="${cand.class_name}">${cand.class_name}</span>
                <div class="candidate-track">
                    <div class="candidate-fill" style="width: ${cand.confidence}%;"></div>
                </div>
                <span class="candidate-percent">${cand.confidence}%</span>
            `;
            candidatesList.appendChild(row);
        });

        // Reset to First Tab (Prognosis)
        tabBtns.forEach(b => b.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));
        const firstTab = document.querySelector(`.tab-btn[data-tab="tab-prognosis"]`);
        if (firstTab) firstTab.classList.add("active");
        const firstPane = document.getElementById("tab-prognosis");
        if (firstPane) firstPane.classList.add("active");

        // Scroll to results if on smaller screen
        if (window.innerWidth <= 1040) {
            resultsContent.scrollIntoView({ behavior: "smooth" });
        }
    }

    // Tab Navigation Switcher
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add("active");
        });
    });

    // Print Report
    printReportBtn.addEventListener("click", () => {
        window.print();
    });

    // Copy Summary to Clipboard
    copySummaryBtn.addEventListener("click", () => {
        if (!latestAnalysisData) return;
        const agro = latestAnalysisData.agronomic_analysis;
        const future = agro.future_prediction || {};
        const roadmap = agro.cure_roadmap || [];

        const summary = `
AGRIVISION AI DIAGNOSTIC & PROGNOSTIC REPORT
============================================
Diagnosis: ${agro.display_name} (${latestAnalysisData.confidence}% confidence)
Crop: ${agro.crop} | Severity: ${agro.severity}
Pathogen: ${agro.pathogen || 'N/A'}

PROGNOSIS IF UNTREATED:
- Potential Yield Loss: ${future.loss_percentage || 0}% (${future.yield_loss_risk || 'N/A'})
- Economic Ramifications: ${future.economic_impact || 'N/A'}
- Contagion Radius & Spread: ${future.contagion_radius || 'N/A'}
- Neighboring Plot Risk: ${future.neighbor_plot_risk || 'N/A'}

7-DAY CURE ROADMAP:
${roadmap.map(r => `[${r.day}] ${r.action}`).join('\n')}

IMMEDIATE REMEDIATION (First 24-48 Hours):
${agro.immediate_action}

ORGANIC BIOLOGICAL SOLUTIONS:
${(agro.organic_control || []).map(x => '- ' + x).join('\n')}

TARGETED CHEMICAL FORMULATIONS:
${(agro.chemical_control || []).map(x => '- ' + x).join('\n')}

SOIL & NUTRITIONAL REMEDIATION:
${agro.nutritional_recovery}

LONG-TERM PREVENTATIVE SAFEGUARDS:
${agro.preventive_measures}
        `.trim();

        navigator.clipboard.writeText(summary)
            .then(() => showToast("Comprehensive diagnostic & prognostic report copied!"))
            .catch(() => showToast("Failed to copy report to clipboard."));
    });

    // Toast Helper
    let toastTimeout;
    function showToast(msg) {
        clearTimeout(toastTimeout);
        toast.textContent = msg;
        toast.classList.add("show");
        toastTimeout = setTimeout(() => {
            toast.classList.remove("show");
        }, 3200);
    }
});
