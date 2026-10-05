// Change this only if your backend runs on a different address
const API_URL = "http://127.0.0.1:8000";

// ---------- Page elements ----------
const statusBox = document.getElementById("status");
const statusText = document.getElementById("statusText");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadMessage = document.getElementById("uploadMessage");
const docList = document.getElementById("docList");
const docCount = document.getElementById("docCount");
const chat = document.getElementById("chat");
const empty = document.getElementById("empty");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");

// ---------- 1. Check backend  (GET /health) ----------
async function checkHealth() {
  try {
    const res = await fetch(API_URL + "/health");
    const data = await res.json();
    statusBox.className = "status online";
    statusText.textContent = data.status === "ok" ? "Backend online" : "Backend issue";
  } catch (err) {
    statusBox.className = "status offline";
    statusText.textContent = "Backend offline";
  }
}

// ---------- 2. Load documents  (GET /documents) ----------
async function loadDocuments() {
  try {
    const res = await fetch(API_URL + "/documents");
    const data = await res.json();

    docCount.textContent = data.count;
    docList.innerHTML = "";

    if (data.documents.length === 0) {
      docList.innerHTML = '<li class="none">No documents yet. Upload a PDF to start.</li>';
      return;
    }

    data.documents.forEach((doc) => {
      const li = document.createElement("li");
      li.innerHTML = '<span class="name"></span><span class="badge"></span>';
      li.querySelector(".name").textContent = doc.filename;
      li.querySelector(".badge").textContent = doc.status;
      docList.appendChild(li);
    });
  } catch (err) {
    docList.innerHTML = '<li class="none">Could not load documents.</li>';
  }
}

// ---------- 3. Upload PDF  (POST /document/upload_file) ----------
async function uploadFile(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  dropzone.classList.add("busy");
  showUploadMessage("Indexing " + file.name + "… large PDFs with images can take a minute.", "");

  try {
    const res = await fetch(API_URL + "/document/upload_file", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Upload failed (" + res.status + ")");

    const data = await res.json();
    showUploadMessage(
      "Indexed: " + data.chunks + " text chunks, " + data.tables + " tables, " + data.images + " images.",
      "success"
    );
    loadDocuments();
  } catch (err) {
    showUploadMessage(err.message, "error");
  }

  dropzone.classList.remove("busy");
  fileInput.value = "";
}

function showUploadMessage(text, type) {
  uploadMessage.hidden = false;
  uploadMessage.textContent = text;
  uploadMessage.className = "upload-message " + type;
}

// ---------- 4. Ask a question  (POST /ask) ----------
async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  empty.style.display = "none";
  addMessage(question, "user");
  questionInput.value = "";
  askBtn.disabled = true;

  const loading = addMessage("", "bot");
  loading.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

  try {
    const res = await fetch(API_URL + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Something went wrong");

    loading.textContent = data.answer;
    showImages(loading, data.images);
  } catch (err) {
    loading.textContent = err.message;
    loading.classList.add("error");
  }

  askBtn.disabled = false;
  chat.scrollTop = chat.scrollHeight;
}

// ---------- Helpers ----------
function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = "msg " + type;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

// Shows retrieved images. If the backend does not serve the file,
// the image path is shown as a small label instead.
function showImages(messageBox, images) {
  if (!images || images.length === 0) return;

  const wrap = document.createElement("div");
  wrap.className = "images";

  images.forEach((path) => {
    const img = document.createElement("img");
    img.src = API_URL + "/" + path.replace(/^\.?\//, "");
    img.onerror = () => {
      const label = document.createElement("span");
      label.className = "path";
      label.textContent = path;
      img.replaceWith(label);
    };
    wrap.appendChild(img);
  });

  messageBox.appendChild(wrap);
}

// ---------- Events ----------
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  uploadFile(e.dataTransfer.files[0]);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.textContent;
    askQuestion();
  });
});

// ---------- Start ----------
checkHealth();
loadDocuments();
