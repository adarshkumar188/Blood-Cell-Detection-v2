// ─── Image Preview ────────────────────────────────────
function previewImage() {
    const input       = document.getElementById("imageInput");
    const preview     = document.getElementById("preview");
    const placeholder = document.getElementById("placeholder");
    const label       = document.querySelector(".file-label");

    const file = input.files[0];
    if (!file) return;

    label.textContent = "📂 " + file.name;
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    placeholder.style.display = "none";

    // Reset counts when new image chosen
    resetCounts();
}

// ─── Reset UI ─────────────────────────────────────────
function resetCounts() {
    ["rbcCount", "wbcCount", "plateletCount"].forEach(id => {
        document.getElementById(id).textContent = "—";
    });
    document.getElementById("totalBar").classList.add("hidden");
    document.getElementById("downloadLink").classList.add("hidden");
    document.getElementById("errorBanner").classList.add("hidden");
}

// ─── Upload & Analyze ─────────────────────────────────
function uploadImage() {
    const input = document.getElementById("imageInput");
    const file  = input.files[0];

    if (!file) {
        showError("Please select an image first.");
        return;
    }

    const btn    = document.getElementById("analyzeBtn");
    const loader = document.getElementById("loader");
    const errBanner = document.getElementById("errorBanner");

    // Show loading state
    btn.disabled = true;
    loader.classList.remove("hidden");
    errBanner.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(d => { throw new Error(d.error || "Server error " + res.status); });
        }
        return res.json();
    })
    .then(data => {
        if (data.error) throw new Error(data.error);

        // Update processed image
        const preview     = document.getElementById("preview");
        const placeholder = document.getElementById("placeholder");
        preview.src = data.image + "?t=" + Date.now();
        preview.style.display = "block";
        placeholder.style.display = "none";

        // Update counts with animation
        animateCount("rbcCount",      data.counts.RBC       || 0);
        animateCount("wbcCount",      data.counts.WBC       || 0);
        animateCount("plateletCount", data.counts.Platelets || 0);

        // Show total
        document.getElementById("totalCount").textContent = data.total || 0;
        document.getElementById("totalBar").classList.remove("hidden");

        // Show download
        document.getElementById("downloadLink").classList.remove("hidden");
    })
    .catch(err => {
        showError("❌ " + err.message);
    })
    .finally(() => {
        btn.disabled = false;
        loader.classList.add("hidden");
    });
}

// ─── Count animation ─────────────────────────────────
function animateCount(elementId, target) {
    const el = document.getElementById(elementId);
    const duration = 600;
    const steps = 20;
    const step = Math.ceil(target / steps);
    let current = 0;
    const timer = setInterval(() => {
        current = Math.min(current + step, target);
        el.textContent = current;
        if (current >= target) clearInterval(timer);
    }, duration / steps);
}

// ─── Error helper ─────────────────────────────────────
function showError(msg) {
    const banner = document.getElementById("errorBanner");
    banner.textContent = msg;
    banner.classList.remove("hidden");
}
