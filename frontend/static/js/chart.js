/* 图表渲染模块：Top-5 柱状图 + Attention Map 热力图 */
(function (window) {
    "use strict";

    let top5Chart = null;
    let attentionChart = null;

    /** 初始化 Top-5 柱状图 */
    function initTop5Chart(dom) {
        top5Chart = echarts.init(dom);
        top5Chart.setOption({
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
        });
        window.addEventListener("resize", () => top5Chart && top5Chart.resize());
    }

    /** 渲染 Top-5 数据 */
    function renderTop5(top5) {
        if (!top5Chart) return;
        if (!Array.isArray(top5) || top5.length === 0) return;

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

        top5Chart.setOption(
            {
                xAxis: { data: sortedFamilies },
                yAxis: { max: yMax },
                series: [{ data: sortedScores }],
            },
            // 显式不合并，防止切换样本时残留上一次的轴标签
            { notMerge: false, lazyUpdate: true }
        );
    }

    /** 初始化 Attention 热力图 */
    function initAttentionChart(dom) {
        attentionChart = echarts.init(dom);
        window.addEventListener("resize", () => attentionChart && attentionChart.resize());
    }

    /**
     * 渲染 Attention Map 热力图
     * attentionData: 二维数组 [[row1...], [row2...], ...]
     */
    function renderAttention(attentionData) {
        if (!attentionChart) return;

        if (!attentionData || !Array.isArray(attentionData) || attentionData.length === 0) {
            attentionChart.clear();
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

        attentionChart.setOption({
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
        return true;
    }

    // 暴露到全局
    window.ChartModule = {
        initTop5Chart,
        renderTop5,
        initAttentionChart,
        renderAttention,
    };
})(window);
