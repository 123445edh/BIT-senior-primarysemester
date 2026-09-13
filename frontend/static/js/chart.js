/* 图表渲染模块：Top-5 柱状图 + Attention Map 热力图
 *
 * 支持多实例：主页面一套、历史记录详情弹窗一套，互不干扰。
 * 每次渲染后都会 resize()，避免容器从 d-none / 弹窗隐藏态切为可见时
 * 被 ECharts 冻结在默认 100px 宽度（表现为"缩成一小条"）。
 */
(function (window) {
    "use strict";

    // 主页面图表实例
    var top5Chart = null;
    var attentionChart = null;

    // ============ Top-5 柱状图 ============

    /** Top-5 图表的基础配置（不含数据） */
    function buildTop5Option() {
        return {
            title: { text: "Top-5 候选家族概率", left: "center", textStyle: { fontSize: 14 } },
            tooltip: {
                trigger: "axis",
                axisPointer: { type: "shadow" },
                formatter: function (params) {
                    const p = params[0];
                    if (!p || typeof p.value !== "number") return "";
                    return `${p.name}<br/>概率：${(p.value * 100).toFixed(2)}%`;
                },
            },
            grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
            xAxis: {
                type: "category",
                data: [],
                axisLabel: { interval: 0, rotate: 0, hideOverlap: true },
            },
            yAxis: {
                type: "value",
                // 上限动态计算：初始给 1，渲染时按实际最大概率收紧，
                // 避免小概率样本的柱子被压成一条线
                max: 1,
                axisLabel: {
                    formatter: function (v) { return (v * 100).toFixed(0) + "%"; },
                },
            },
            series: [
                {
                    type: "bar",
                    data: [],
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: "#4facfe" },
                            { offset: 1, color: "#00f2fe" },
                        ]),
                    },
                    label: {
                        show: true,
                        position: "top",
                        formatter: function (p) {
                            return typeof p.value === "number"
                                ? (p.value * 100).toFixed(2) + "%"
                                : "";
                        },
                    },
                    barWidth: "40%",
                },
            ],
        };
    }

    /**
     * 创建一个独立的 Top-5 图表实例。
     * 调用前请确保 dom 已经可见（父容器不能是 display:none），否则会量不到宽度。
     */
    function createTop5Chart(dom) {
        const chart = echarts.init(dom);
        chart.setOption(buildTop5Option());
        return chart;
    }

    /**
     * 把 Top-5 数据渲染到指定实例上。
     * 传入的 chart 由 createTop5Chart 或 initTop5Chart 返回，可为任意实例。
     */
    function renderTop5On(chart, top5) {
        if (!chart || chart.isDisposed()) return false;
        if (!Array.isArray(top5) || top5.length === 0) return false;

        // 兼容 {family, score} 与 {family, probability} 两种字段命名
        const pickScore = (t) =>
            typeof t.score === "number"
                ? t.score
                : typeof t.probability === "number"
                  ? t.probability
                  : 0;

        const families = top5.map((t) => String(t.family ?? "未知"));
        const scores = top5.map(pickScore);

        // 按概率降序，保证图上从高到低
        const order = scores
            .map((v, i) => [v, i])
            .sort((a, b) => b[0] - a[0])
            .map((p) => p[1]);
        const sortedFamilies = order.map((i) => families[i]);
        const sortedScores = order.map((i) => scores[i]);

        // 动态 y 轴上限：留 25% 顶部余量给数据标签，同时保留 1 为上限的上限
        const maxScore = Math.max.apply(null, sortedScores);
        const yMax = Math.min(1, Math.max(0.1, maxScore * 1.25));

        chart.setOption({
            xAxis: { data: sortedFamilies },
            yAxis: { max: yMax },
            series: [{ data: sortedScores }],
        });

        // 容器可能刚从隐藏态切为可见，强制按当前尺寸重算
        chart.resize();
        return true;
    }

    /** 初始化主页面的 Top-5 图表 */
    function initTop5Chart(dom) {
        top5Chart = createTop5Chart(dom);
        window.addEventListener("resize", function () {
            if (top5Chart && !top5Chart.isDisposed()) top5Chart.resize();
        });
    }

    /** 渲染主页面的 Top-5 数据 */
    function renderTop5(top5) {
        return renderTop5On(top5Chart, top5);
    }

    // ============ Attention Map 热力图 ============

    /** 创建一个独立的 Attention 图表实例（调用前 dom 需可见） */
    function createAttentionChart(dom) {
        return echarts.init(dom);
    }

    /**
     * 渲染 Attention Map 热力图到指定实例
     * attentionData: 二维数组 [[row1...], [row2...], ...]
     */
    function renderAttentionOn(chart, attentionData) {
        if (!chart || chart.isDisposed()) return false;

        if (!attentionData || !Array.isArray(attentionData) || attentionData.length === 0) {
            chart.clear();
            return false;
        }

        const rows = attentionData.length;
        const cols = attentionData[0].length;

        // 转换为 [x, y, value] 格式
        const data = [];
        let maxVal = 0;
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                const v = attentionData[y][x];
                data.push([x, y, v]);
                if (v > maxVal) maxVal = v;
            }
        }

        // x 轴标签：列索引
        const xLabels = [];
        for (let i = 0; i < cols; i++) xLabels.push(String(i));
        // y 轴标签：行索引（如 Head 编号）
        const yLabels = [];
        for (let i = 0; i < rows; i++) yLabels.push("Head " + i);

        chart.setOption({
            title: { text: "Attention Map（注意力权重分布）", left: "center", textStyle: { fontSize: 14 } },
            tooltip: {
                position: "top",
                formatter: function (p) {
                    return `行：${yLabels[p.value[1]]}<br/>列：${xLabels[p.value[0]]}<br/>权重：${p.value[2].toFixed(4)}`;
                },
            },
            grid: { left: "8%", right: "4%", bottom: "12%", top: "15%" },
            xAxis: {
                type: "category",
                data: xLabels,
                splitArea: { show: true },
                axisLabel: { fontSize: 10, rotate: cols > 20 ? 45 : 0 },
            },
            yAxis: {
                type: "category",
                data: yLabels,
                splitArea: { show: true },
                axisLabel: { fontSize: 10 },
            },
            visualMap: {
                min: 0,
                max: maxVal > 0 ? maxVal : 1,
                calculable: true,
                orient: "horizontal",
                left: "center",
                bottom: "0%",
                inRange: {
                    color: ["#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"],
                },
            },
            series: [
                {
                    type: "heatmap",
                    data: data,
                    label: { show: false },
                    emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0, 0, 0, 0.5)" } },
                },
            ],
        });

        // 同 Top-5：容器从隐藏态切为可见后需重算尺寸
        chart.resize();
        return true;
    }

    /** 初始化主页面的 Attention 图表 */
    function initAttentionChart(dom) {
        attentionChart = createAttentionChart(dom);
        window.addEventListener("resize", function () {
            if (attentionChart && !attentionChart.isDisposed()) attentionChart.resize();
        });
    }

    /** 渲染主页面的 Attention 数据 */
    function renderAttention(attentionData) {
        return renderAttentionOn(attentionChart, attentionData);
    }

    // 暴露到全局
    window.ChartModule = {
        initTop5Chart,
        renderTop5,
        initAttentionChart,
        renderAttention,
        // 多实例接口（历史详情弹窗使用）
        createTop5Chart,
        renderTop5On,
        createAttentionChart,
        renderAttentionOn,
        // 模块自建的 resize 逻辑用不到时，外部可手动触发
        resizeAll: function () {
            [top5Chart, attentionChart].forEach(function (c) {
                if (c && !c.isDisposed()) c.resize();
            });
        },
    };
})(window);
