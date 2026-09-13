/* 主逻辑：样本上传、分类结果展示、历史记录 */
(function () {
    "use strict";

    // 允许的文件后缀
    const ALLOWED_EXT = [".txt", ".hex", ".bin", ".bytes"];

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

    // 历史记录详情弹窗
    const historyDetailModalEl = document.getElementById("history-detail-modal");
    const detailLoading = document.getElementById("history-detail-loading");
    const detailError = document.getElementById("history-detail-error");
    const detailContent = document.getElementById("history-detail-content");
    const detailId = document.getElementById("detail-id");
    const detailFilename = document.getElementById("detail-filename");
    const detailFamily = document.getElementById("detail-family");
    const detailConfidence = document.getElementById("detail-confidence");
    const detailSize = document.getElementById("detail-size");
    const detailTime = document.getElementById("detail-time");
    const detailTop5Wrap = document.getElementById("detail-top5-wrap");
    const detailTop5Dom = document.getElementById("detail-top5-chart");
    const detailAttentionWrap = document.getElementById("detail-attention-wrap");
    const detailAttentionDom = document.getElementById("detail-attention-chart");
    const detailNoExtra = document.getElementById("detail-no-extra");

    let selectedFile = null;
    let isPredicting = false;

    // 历史详情状态
    let historyDetailModal = null; // Bootstrap Modal 实例（懒创建）
    let detailRecord = null; // 当前详情数据，供弹窗动画结束后渲染图表
    let detailTop5Chart = null;
    let detailAttentionChart = null;
    let detailRequestSeq = 0; // 防止连点不同行时旧响应覆盖新数据

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
            showError("不支持的文件格式，仅支持 " + ALLOWED_EXT.join(" / "));
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
            // 注意顺序：必须先让容器可见（去掉 d-none），再渲染。
            // 否则容器宽高为 0，ECharts 会退回默认 100px 宽度，图表缩成一小条。
            if (data.top5 && data.top5.length > 0) {
                top5Section.classList.remove("d-none");
                ChartModule.renderTop5(data.top5);
            } else {
                top5Section.classList.add("d-none");
            }

            // Attention Map（同样先显示容器再渲染）
            const hasAttention =
                data.attention_data &&
                Array.isArray(data.attention_data) &&
                data.attention_data.length > 0;
            attentionSection.classList.remove("d-none");
            if (hasAttention) {
                ChartModule.renderAttention(data.attention_data);
                attentionEmpty.classList.add("d-none");
            } else {
                attentionEmpty.classList.remove("d-none");
                ChartModule.renderAttention([]);
            }

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
            '<tr><td colspan="5" class="text-center text-muted py-4">加载中…</td></tr>';
        try {
            const resp = await fetch("/api/history?limit=20");
            const data = await resp.json();
            const list = data.history || [];

            if (list.length === 0) {
                historyTbody.innerHTML =
                    '<tr><td colspan="5" class="text-center text-muted py-4">暂无历史记录</td></tr>';
                return;
            }

            historyTbody.innerHTML = list
                .map(
                    (item) => `
                <tr class="history-row" data-id="${escapeHtml(item.id)}" tabindex="0"
                    title="点击查看该记录的详情">
                    <td>${escapeHtml(item.id)}</td>
                    <td>${escapeHtml(item.filename)}</td>
                    <td><span class="badge bg-primary">${escapeHtml(item.result || "—")}</span></td>
                    <td class="text-muted">${escapeHtml(item.timestamp || "")}</td>
                    <td class="text-end"><span class="detail-hint">详情 ›</span></td>
                </tr>
            `
                )
                .join("");
        } catch (e) {
            historyTbody.innerHTML =
                '<tr><td colspan="5" class="text-center text-danger py-4">历史记录加载失败</td></tr>';
        }
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = String(str);
        return div.innerHTML;
    }

    // ============ 历史记录详情弹窗 ============
    function getDetailModal() {
        if (!historyDetailModal && historyDetailModalEl && window.bootstrap) {
            historyDetailModal = new window.bootstrap.Modal(historyDetailModalEl);
        }
        return historyDetailModal;
    }

    function disposeDetailCharts() {
        if (detailTop5Chart) {
            if (!detailTop5Chart.isDisposed()) detailTop5Chart.dispose();
            detailTop5Chart = null;
        }
        if (detailAttentionChart) {
            if (!detailAttentionChart.isDisposed()) detailAttentionChart.dispose();
            detailAttentionChart = null;
        }
    }

    /** 渲染弹窗内的图表。调用时弹窗必须已可见，否则 ECharts 会量不到宽度。 */
    function renderDetailCharts() {
        const rec = detailRecord;
        if (!rec) return;
        disposeDetailCharts();

        const hasTop5 = Array.isArray(rec.top5) && rec.top5.length > 0;
        const hasAttention =
            Array.isArray(rec.attention_data) && rec.attention_data.length > 0;

        if (hasTop5) {
            detailTop5Wrap.classList.remove("d-none");
            detailTop5Chart = ChartModule.createTop5Chart(detailTop5Dom);
            ChartModule.renderTop5On(detailTop5Chart, rec.top5);
        } else {
            detailTop5Wrap.classList.add("d-none");
        }

        if (hasAttention) {
            detailAttentionWrap.classList.remove("d-none");
            detailAttentionChart = ChartModule.createAttentionChart(detailAttentionDom);
            ChartModule.renderAttentionOn(detailAttentionChart, rec.attention_data);
        } else {
            detailAttentionWrap.classList.add("d-none");
        }

        detailNoExtra.classList.toggle("d-none", hasTop5 || hasAttention);
    }

    function fillDetailFields(rec) {
        detailId.textContent = rec.id != null ? rec.id : "—";
        detailFilename.textContent = rec.filename || "—";
        detailFamily.textContent = rec.predicted_family || "—";
        detailConfidence.textContent =
            typeof rec.confidence === "number"
                ? (rec.confidence * 100).toFixed(2) + "%"
                : "—";
        detailSize.textContent =
            typeof rec.file_size === "number"
                ? formatFileSize(rec.file_size) + "（" + rec.file_size + " B）"
                : "—";
        detailTime.textContent = rec.timestamp || "—";
    }

    async function openHistoryDetail(recordId) {
        const modal = getDetailModal();
        if (!modal) return;

        // 先切到 loading 态并打开弹窗
        detailRecord = null;
        disposeDetailCharts();
        detailLoading.classList.remove("d-none");
        detailError.classList.add("d-none");
        detailContent.classList.add("d-none");
        modal.show();

        const seq = ++detailRequestSeq;
        try {
            const resp = await fetch("/api/history/" + encodeURIComponent(recordId));
            const data = await resp.json();
            if (seq !== detailRequestSeq) return; // 用户已点了别的行，丢弃过期响应

            if (!resp.ok || data.status !== "success" || !data.record) {
                throw new Error(data.message || "记录不存在");
            }

            detailRecord = data.record;
            fillDetailFields(detailRecord);
            detailLoading.classList.add("d-none");
            detailContent.classList.remove("d-none");

            // 弹窗若还在淡入动画中，交给 shown 事件渲染；已可见则立即渲染
            if (historyDetailModalEl.classList.contains("show")) {
                renderDetailCharts();
            }
        } catch (e) {
            if (seq !== detailRequestSeq) return;
            detailLoading.classList.add("d-none");
            detailContent.classList.add("d-none");
            detailError.textContent = "加载失败：" + (e && e.message ? e.message : "未知错误");
            detailError.classList.remove("d-none");
        }
    }

    if (historyDetailModalEl) {
        // 弹窗完全显示后再渲染，保证容器有实际宽高
        historyDetailModalEl.addEventListener("shown.bs.modal", renderDetailCharts);
        historyDetailModalEl.addEventListener("hidden.bs.modal", function () {
            disposeDetailCharts();
            detailRecord = null;
        });
    }

    // 点击历史行查看详情（事件委托，避免逐行绑定）
    historyTbody.addEventListener("click", function (e) {
        const row = e.target.closest("tr.history-row");
        if (!row) return;
        const id = row.getAttribute("data-id");
        if (id) openHistoryDetail(id);
    });

    // 键盘可达性：回车 / 空格
    historyTbody.addEventListener("keydown", function (e) {
        if (e.key !== "Enter" && e.key !== " ") return;
        const row = e.target.closest("tr.history-row");
        if (!row) return;
        e.preventDefault();
        const id = row.getAttribute("data-id");
        if (id) openHistoryDetail(id);
    });

    // 弹窗内图表跟随窗口尺寸变化
    window.addEventListener("resize", function () {
        [detailTop5Chart, detailAttentionChart].forEach(function (c) {
            if (c && !c.isDisposed()) c.resize();
        });
    });

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
