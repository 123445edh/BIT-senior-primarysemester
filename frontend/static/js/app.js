/* 主逻辑：样本上传、分类结果展示、历史记录 */
(function () {
    "use strict";

    // 允许的文件后缀
    const ALLOWED_EXT = [".txt", ".hex", ".bin"];

    // DOM 元素
    const fileInput = document.getElementById("file-input");
    const predictBtn = document.getElementById("predict-btn");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");
    const fileInfo = document.getElementById("file-info");
    const errorMsg = document.getElementById("error-msg");
    const resultSection = document.getElementById("result-section");
    const top5Section = document.getElementById("top5-section");
    const attentionSection = document.getElementById("attention-section");
    const attentionEmpty = document.getElementById("attention-empty");
    const predictedFamilyEl = document.getElementById("predicted-family");
    const confidenceEl = document.getElementById("confidence");
    const historyTbody = document.getElementById("history-tbody");
    const refreshHistoryBtn = document.getElementById("refresh-history-btn");
    const serviceStatus = document.getElementById("service-status");

    let selectedFile = null;
    let isPredicting = false;

    // ============ 工具函数 ============
    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove("d-none");
    }

    function clearError() {
        errorMsg.classList.add("d-none");
        errorMsg.textContent = "";
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + " KB";
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    function getExt(filename) {
        const idx = filename.lastIndexOf(".");
        return idx === -1 ? "" : filename.slice(idx).toLowerCase();
    }

    function setPredicting(predicting) {
        isPredicting = predicting;
        predictBtn.disabled = predicting || !selectedFile;
        fileInput.disabled = predicting;
        if (predicting) {
            btnText.textContent = "分类中…";
            btnSpinner.classList.remove("d-none");
        } else {
            btnText.textContent = "开始分类";
            btnSpinner.classList.add("d-none");
        }
    }

    // ============ 服务状态检测 ============
    async function checkHealth() {
        try {
            const resp = await fetch("/api/health");
            const data = await resp.json();
            if (data.status === "ok") {
                serviceStatus.textContent = "服务状态：在线";
                serviceStatus.className = "badge fs-6 status-ok";
            } else {
                throw new Error("状态异常");
            }
        } catch (e) {
            serviceStatus.textContent = "服务状态：离线";
            serviceStatus.className = "badge fs-6 status-error";
        }
    }

    // ============ 文件选择 ============
    fileInput.addEventListener("change", function () {
        clearError();
        const file = this.files && this.files[0];
        if (!file) {
            selectedFile = null;
            fileInfo.classList.add("d-none");
            predictBtn.disabled = true;
            return;
        }

        // 格式校验
        const ext = getExt(file.name);
        if (!ALLOWED_EXT.includes(ext)) {
            showError("不支持的文件格式，仅支持 .txt / .hex / .bin");
            selectedFile = null;
            fileInfo.classList.add("d-none");
            predictBtn.disabled = true;
            this.value = "";
            return;
        }

        selectedFile = file;
        fileInfo.innerHTML =
            `已选择文件：<strong>${file.name}</strong>　` +
            `大小：${formatFileSize(file.size)}　` +
            `类型：${ext}`;
        fileInfo.classList.remove("d-none");
        predictBtn.disabled = false;
    });

    // ============ 执行分类 ============
    predictBtn.addEventListener("click", async function () {
        if (!selectedFile) {
            showError("请先选择样本文件");
            return;
        }
        clearError();
        setPredicting(true);

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            });
            const data = await resp.json();

            if (data.status !== "success") {
                showError(data.message || "分类失败，请稍后重试");
                setPredicting(false);
                return;
            }

            // 展示结果
            predictedFamilyEl.textContent = data.predicted_family || "—";
            confidenceEl.textContent =
                typeof data.confidence === "number"
                    ? (data.confidence * 100).toFixed(2) + "%"
                    : "—";
            resultSection.classList.remove("d-none");

            // Top-5 柱状图
            if (data.top5 && data.top5.length > 0) {
                ChartModule.renderTop5(data.top5);
                top5Section.classList.remove("d-none");
            } else {
                top5Section.classList.add("d-none");
            }

            // Attention Map
            const hasAttention =
                data.attention_data &&
                Array.isArray(data.attention_data) &&
                data.attention_data.length > 0;
            if (hasAttention) {
                ChartModule.renderAttention(data.attention_data);
                attentionEmpty.classList.add("d-none");
            } else {
                attentionEmpty.classList.remove("d-none");
                ChartModule.renderAttention([]);
            }
            attentionSection.classList.remove("d-none");

            // 刷新历史记录
            loadHistory();
        } catch (e) {
            showError("网络异常或后端无响应，请检查后端服务是否启动");
        } finally {
            setPredicting(false);
        }
    });

    // ============ 历史记录 ============
    async function loadHistory() {
        historyTbody.innerHTML =
            '<tr><td colspan="4" class="text-center text-muted py-4">加载中…</td></tr>';
        try {
            const resp = await fetch("/api/history?limit=20");
            const data = await resp.json();
            const list = data.history || [];

            if (list.length === 0) {
                historyTbody.innerHTML =
                    '<tr><td colspan="4" class="text-center text-muted py-4">暂无历史记录</td></tr>';
                return;
            }

            historyTbody.innerHTML = list
                .map(
                    (item) => `
                <tr>
                    <td>${item.id}</td>
                    <td>${escapeHtml(item.filename)}</td>
                    <td><span class="badge bg-primary">${escapeHtml(item.result || "")}</span></td>
                    <td class="text-muted">${escapeHtml(item.timestamp || "")}</td>
                </tr>
            `
                )
                .join("");
        } catch (e) {
            historyTbody.innerHTML =
                '<tr><td colspan="4" class="text-center text-danger py-4">历史记录加载失败</td></tr>';
        }
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = String(str);
        return div.innerHTML;
    }

    refreshHistoryBtn.addEventListener("click", loadHistory);

    // ============ 初始化 ============
    window.addEventListener("DOMContentLoaded", function () {
        // 初始化图表
        ChartModule.initTop5Chart(document.getElementById("top5-chart"));
        ChartModule.initAttentionChart(document.getElementById("attention-chart"));

        // 检测服务状态 + 加载历史记录
        checkHealth();
        loadHistory();

        // 定时检测服务状态（每 30 秒）
        setInterval(checkHealth, 30000);
    });
})();
