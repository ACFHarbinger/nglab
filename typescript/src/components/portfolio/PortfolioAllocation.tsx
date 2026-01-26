
import { useState } from "react";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import clsx from "clsx";

export function PortfolioAllocation() {
    const [allocation, setAllocation] = useState([
        { name: "BTC", y: 45, color: "#f7931a" },
        { name: "ETH", y: 30, color: "#627eea" },
        { name: "SOL", y: 15, color: "#14F195" },
        { name: "USDC", y: 10, color: "#2775ca" },
    ]);

    const [frontierPoints] = useState(() => {
        // Generate mock Efficient Frontier points
        const points = [];
        for (let i = 0; i < 50; i++) {
            const risk = 0.05 + i * 0.005;
            const return_ = Math.sqrt(risk) * 0.5 - 0.05 + Math.random() * 0.02;
            points.push([risk * 100, return_ * 100]); // in %
        }
        return points;
    });

    const pieOptions: Highcharts.Options = {
        chart: {
            type: "pie",
            backgroundColor: "transparent",
            height: 300,
        },
        title: { text: undefined },
        plotOptions: {
            pie: {
                allowPointSelect: true,
                cursor: "pointer",
                dataLabels: {
                    enabled: true,
                    format: "<b>{point.name}</b>: {point.percentage:.1f} %",
                    color: "#cbd5e1", // slate-300
                },
                borderWidth: 0,
            },
        },
        series: [{
            type: "pie",
            name: "Allocation",
            data: allocation,
        }],
        credits: { enabled: false },
    };

    const scatterOptions: Highcharts.Options = {
        chart: {
            type: "scatter",
            backgroundColor: "transparent",
            height: 300,
        },
        title: { text: undefined },
        xAxis: {
            title: { text: "Risk (Vol %)", style: { color: "#94a3b8" } },
            gridLineColor: "#334155",
            labels: { style: { color: "#94a3b8" } },
        },
        yAxis: {
            title: { text: "Return (%)", style: { color: "#94a3b8" } },
            gridLineColor: "#334155",
            labels: { style: { color: "#94a3b8" } },
        },
        legend: { enabled: false },
        plotOptions: {
            scatter: {
                marker: {
                    radius: 3,
                    fillColor: "rgba(99, 102, 241, 0.5)", // indigo-500
                }
            }
        },
        series: [
            {
                type: "scatter",
                name: "Portfolios",
                data: frontierPoints,
            },
            {
                type: "scatter",
                name: "Current",
                data: [[12.5, 8.2]], // Mock current pos
                marker: { radius: 6, fillColor: "#22c55e", symbol: "diamond" },
                dataLabels: { enabled: true, format: "You", color: "#fff" }
            }
        ],
        credits: { enabled: false },
    };

    return (
        <div className="flex h-full bg-slate-950 p-6 gap-6 overflow-hidden">
            {/* Left: Allocation */}
            <div className="w-1/2 flex flex-col gap-4">
                <div className="bg-slate-900 rounded-xl p-4 border border-slate-800 flex-1">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Current Allocation</h3>
                    <HighchartsReact highcharts={Highcharts} options={pieOptions} />

                    {/* Legend / Adjustment? */}
                    <div className="grid grid-cols-2 gap-2 mt-4">
                        {allocation.map(a => (
                            <div key={a.name} className="flex justify-between items-center bg-slate-800 p-2 rounded">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: a.color }} />
                                    <span className="text-sm font-bold text-slate-200">{a.name}</span>
                                </div>
                                <span className="text-sm text-slate-400">{a.y}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Right: Optimization */}
            <div className="w-1/2 flex flex-col gap-4">
                <div className="bg-slate-900 rounded-xl p-4 border border-slate-800 flex-1">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Efficient Frontier</h3>
                    <HighchartsReact highcharts={Highcharts} options={scatterOptions} />

                    <div className="mt-4 p-3 bg-slate-800/50 rounded border border-slate-700">
                        <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">Optimization Suggestion</h4>
                        <p className="text-sm text-slate-300">
                            To achieve <b>Target Return (10%)</b> with minimal risk, consider increasing ETH exposure by <b>5%</b> and reducing USDC.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
