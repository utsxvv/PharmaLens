document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    
    const uploadSection = document.getElementById("upload-section");
    const previewSection = document.getElementById("preview-section");
    const resultSection = document.getElementById("result-section");
    
    const imagePreview = document.getElementById("image-preview");
    const laserScanner = document.getElementById("laser-scanner");
    const loaderContainer = document.getElementById("loader-container");
    
    // Result elements
    const statusBadge = document.getElementById("status-badge");
    const medicineName = document.getElementById("medicine-name");
    const genericName = document.getElementById("generic-name");
    const rawOcr = document.getElementById("raw-ocr");
    const confidencePercentage = document.getElementById("confidence-percentage");
    const confidenceBar = document.getElementById("confidence-bar");
    const infoAlert = document.getElementById("info-alert");
    const alertMessage = document.getElementById("alert-message");
    
    const resetBtn = document.getElementById("reset-btn");
    const copyAllBtn = document.getElementById("copy-all-btn");
    const copyBtns = document.querySelectorAll(".copy-btn");

    let isScanning = false;

    const ICONS = {
        success: `
            <svg class="alert-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
        `,
        warning: `
            <svg class="alert-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
        `,
        error: `
            <svg class="alert-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="15" y1="9" x2="9" y2="15"></line>
                <line x1="9" y1="9" x2="15" y2="15"></line>
            </svg>
        `
    };

    // Prevent default drag/drop behaviors globally to avoid browser navigating away
    window.addEventListener("dragover", (e) => {
        e.preventDefault();
    }, false);
    window.addEventListener("drop", (e) => {
        e.preventDefault();
    }, false);

    // ── DROPZONE HANDLERS ────────────────────────────────────────────────────

    // Click on dropzone triggers hidden file input
    dropzone.addEventListener("click", () => {
        if (!isScanning) fileInput.click();
    });

    // File input selection
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag-and-drop animations
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    // Drop file
    dropzone.addEventListener("drop", (e) => {
        if (isScanning) return;
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // ── FILE HANDLING & PREVIEW ──────────────────────────────────────────────

    function handleFile(file) {
        if (isScanning) return;
        // Validation: check file extension
        const validExtensions = ["image/jpeg", "image/jpg", "image/png"];
        if (!validExtensions.includes(file.type)) {
            alert("Unsupported format. Please upload a JPG, JPEG, or PNG image.");
            return;
        }

        // Validation: size check (limit to 5MB)
        if (file.size > 5 * 1024 * 1024) {
            alert("File size is too large. Please upload an image smaller than 5MB.");
            return;
        }

        // Display Image Preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            showPreviewState();
            uploadAndAnalyze(file);
        };
    }

    // ── UI STATE CONTROLLERS ─────────────────────────────────────────────────

    function showPreviewState() {
        isScanning = true;
        uploadSection.classList.add("hidden");
        previewSection.classList.remove("hidden");
        resultSection.classList.add("hidden");
        
        // Show scanning animation
        laserScanner.style.display = "block";
        loaderContainer.style.display = "flex";
        loaderContainer.classList.remove("hidden");
    }

    function showResultsState(result) {
        isScanning = false;
        // Stop scanning animation
        laserScanner.style.display = "none";
        loaderContainer.style.display = "none";
        loaderContainer.classList.add("hidden");
        
        // Show result section
        resultSection.classList.remove("hidden");

        const iconContainer = document.getElementById("alert-icon-container");

        // Format Status Badge
        statusBadge.className = "status-badge"; // reset classes
        if (result.status === "success") {
            statusBadge.classList.add("success");
            statusBadge.textContent = "Success";
            
            // Format info alert
            infoAlert.style.borderColor = "rgba(16, 185, 129, 0.2)";
            infoAlert.style.backgroundColor = "rgba(16, 185, 129, 0.02)";
            infoAlert.style.color = "var(--text-secondary)";
            if (iconContainer) iconContainer.innerHTML = ICONS.success;
            const iconSvg = infoAlert.querySelector('.alert-icon');
            if (iconSvg) iconSvg.style.color = "var(--success)";
        } else if (result.status === "low_quality") {
            statusBadge.classList.add("low_quality");
            statusBadge.textContent = "Low Quality";
            
            infoAlert.style.borderColor = "rgba(245, 158, 11, 0.2)";
            infoAlert.style.backgroundColor = "rgba(245, 158, 11, 0.02)";
            infoAlert.style.color = "var(--text-secondary)";
            if (iconContainer) iconContainer.innerHTML = ICONS.warning;
            const iconSvg = infoAlert.querySelector('.alert-icon');
            if (iconSvg) iconSvg.style.color = "var(--warning)";
        } else if (result.status === "error") {
            statusBadge.classList.add("not_found");
            statusBadge.textContent = "Scan Error";
            
            infoAlert.style.borderColor = "rgba(239, 68, 68, 0.3)";
            infoAlert.style.backgroundColor = "rgba(239, 68, 68, 0.03)";
            infoAlert.style.color = "var(--error)";
            if (iconContainer) iconContainer.innerHTML = ICONS.error;
            const iconSvg = infoAlert.querySelector('.alert-icon');
            if (iconSvg) iconSvg.style.color = "var(--error)";
        } else {
            statusBadge.classList.add("not_found");
            statusBadge.textContent = "Not Found";
            
            infoAlert.style.borderColor = "rgba(239, 68, 68, 0.3)";
            infoAlert.style.backgroundColor = "rgba(239, 68, 68, 0.03)";
            infoAlert.style.color = "var(--error)";
            if (iconContainer) iconContainer.innerHTML = ICONS.error;
            const iconSvg = infoAlert.querySelector('.alert-icon');
            if (iconSvg) iconSvg.style.color = "var(--error)";
        }

        // Populate details
        medicineName.textContent = result.medicine_name || "N/A";
        genericName.textContent = result.generic_name || "N/A";
        rawOcr.textContent = result.raw_prediction !== null ? result.raw_prediction : "—";
        
        // Confidence percentage and bar
        const confidence = result.confidence || 0;
        confidencePercentage.textContent = `${confidence}%`;
        confidenceBar.style.width = `${confidence}%`;
        
        // Alert message text
        alertMessage.textContent = result.message || "";
    }

    function resetApp() {
        isScanning = false;
        fileInput.value = "";
        imagePreview.src = "";
        
        uploadSection.classList.remove("hidden");
        previewSection.classList.add("hidden");
        resultSection.classList.add("hidden");
        
        // Reset loader container state
        loaderContainer.style.display = "none";
        loaderContainer.classList.add("hidden");
    }

    // ── API INTEGRATION ──────────────────────────────────────────────────────

    function uploadAndAnalyze(file) {
        const formData = new FormData();
        formData.append("image", file);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Server error occurred during prediction.");
            }
            return response.json();
        })
        .then(result => {
            showResultsState(result);
        })
        .catch(err => {
            showResultsState({
                status: "error",
                medicine_name: "N/A",
                generic_name: "N/A",
                raw_prediction: "—",
                confidence: 0,
                message: `Connection failed: ${err.message}`
            });
        });
    }

    // ── UTILITIES & CLIPBOARD ACTIONS ────────────────────────────────────────

    // Single item copy buttons
    copyBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-copy");
            const textToCopy = document.getElementById(targetId).textContent;
            
            if (textToCopy && textToCopy !== "—" && textToCopy !== "N/A") {
                copyTextToClipboard(textToCopy, btn);
            }
        });
    });

    // Copy all details button
    copyAllBtn.addEventListener("click", () => {
        const brand = medicineName.textContent;
        const gen = genericName.textContent;
        const ocr = rawOcr.textContent;
        const conf = confidencePercentage.textContent;

        const text = `PharmaLens Scan Details:\n` +
                     `- Brand Name: ${brand}\n` +
                     `- Generic Ingredient: ${gen}\n` +
                     `- Raw OCR Output: ${ocr}\n` +
                     `- Confidence: ${conf}`;

        copyTextToClipboard(text, copyAllBtn);
    });

    function copyTextToClipboard(text, button) {
        const successHandler = () => {
            // Flash feedback
            const originalText = button.innerHTML;
            if (button.tagName === "BUTTON" && button.classList.contains("btn")) {
                button.textContent = "Copied!";
                button.style.background = "var(--success)";
                button.style.color = "#ffffff";
                setTimeout(() => {
                    button.textContent = "Copy Details";
                    button.style.background = "";
                    button.style.color = "";
                }, 1500);
            } else {
                // Icon copy button
                button.innerHTML = `
                    <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                `;
                setTimeout(() => {
                    button.innerHTML = originalText;
                }, 1500);
            }
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(successHandler).catch(err => {
                fallbackCopyTextToClipboard(text, successHandler);
            });
        } else {
            fallbackCopyTextToClipboard(text, successHandler);
        }
    }

    function fallbackCopyTextToClipboard(text, callback) {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.position = "fixed";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            const successful = document.execCommand("copy");
            if (successful && callback) callback();
        } catch (err) {
            console.error("Fallback copy failed:", err);
        }
        document.body.removeChild(textArea);
    }

    // Reset button
    resetBtn.addEventListener("click", resetApp);
});
