document.addEventListener("DOMContentLoaded", () => {
    const uploadBtn = document.getElementById("upload-btn");
    const fileInput = document.getElementById("file-input");
    const askBtn = document.getElementById("ask-btn");
    const questionInput = document.getElementById("question-input");
    const clearBtn = document.getElementById("clear-btn");
    const uploadMessage = document.getElementById("upload-message");

    // Process Documents
    uploadBtn.addEventListener("click", async () => {
        const files = fileInput.files;
        if (!files.length) {
            uploadMessage.innerHTML = "<span style='color: red;'>Please select files first.</span>";
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append("files", files[i]);
        }

        uploadMessage.innerHTML = "Processing documents...";
        
        try {
            const res = await fetch("/api/process", { method: "POST", body: formData });
            const data = await res.json();

            if (res.ok) {
                uploadMessage.innerHTML = `<span style='color: green;'>🚀 Index ready! ${data.chunks} chunks indexed.</span>`;
                updateStatus(true, data.files_count, data.chunks, data.files);
            } else {
                uploadMessage.innerHTML = `<span style='color: red;'>❌ ${data.detail}</span>`;
            }
        } catch (err) {
            uploadMessage.innerHTML = "<span style='color: red;'>❌ Server connection failed.</span>";
        }
    });

    // Ask Question
    askBtn.addEventListener("click", async () => {
        const question = questionInput.value.trim();
        if (!question) return;

        document.getElementById("answer-text").innerText = "Retrieving answer...";
        document.getElementById("answer-section").classList.remove("hidden");

        try {
            const res = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });
            const data = await res.json();

            if (res.ok) {
                document.getElementById("answer-text").innerText = data.answer;
                renderSources(data.chunks);
                renderChunks(data.chunks);
            } else {
                document.getElementById("answer-text").innerText = "Error: " + data.detail;
            }
        } catch (err) {
            document.getElementById("answer-text").innerText = "Error communicating with server.";
        }
    });

    // Reset Session
    clearBtn.addEventListener("click", async () => {
        await fetch("/api/reset", { method: "POST" });
        location.reload();
    });

    function updateStatus(ready, filesCount, chunksCount, filesList) {
        const pill = document.getElementById("status-pill");
        const details = document.getElementById("status-details");
        const qaDisabledMsg = document.getElementById("qa-disabled-msg");
        const qaForm = document.getElementById("qa-form");

        if (ready) {
            pill.className = "status-ready";
            pill.innerText = "✓ Ready";
            details.innerHTML = `<b>${filesCount}</b> file(s) · <b>${chunksCount}</b> chunks<br><br><b>Loaded files:</b><br>` + 
                filesList.map(f => `• <code>${f}</code>`).join("<br>");
            qaDisabledMsg.classList.add("hidden");
            qaForm.classList.remove("hidden");
        }
    }

    function renderSources(chunks) {
        const container = document.getElementById("source-badges");
        container.innerHTML = "";
        const seen = new Set();
        chunks.forEach(c => {
            const key = `${c.source}-${c.page}`;
            if (!seen.has(key)) {
                seen.add(key);
                const badge = document.createElement("span");
                badge.className = "source-badge";
                badge.innerText = `📄 ${c.source}` + (c.page ? ` — Page ${c.page}` : '');
                container.appendChild(badge);
            }
        });
    }

    function renderChunks(chunks) {
        const list = document.getElementById("chunks-list");
        list.innerHTML = "";
        chunks.forEach((c, idx) => {
            const div = document.createElement("div");
            div.className = "chunk-card";
            div.innerHTML = `
                <div class="chunk-meta">Chunk ${idx + 1} · ${c.source} · ${c.page ? 'Page ' + c.page : 'TXT'}</div>
                <div class="chunk-text">${c.text.substring(0, 800)}...</div>
                <div style="font-size:0.75rem; color:#aaa; margin-top:0.4rem;">Similarity score: ${c.score.toFixed(4)}</div>
            `;
            list.appendChild(div);
        });
    }
});