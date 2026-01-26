
import { useState } from "react";
import { BacktestConfigPanel, BacktestConfig } from "./BacktestConfigPanel";
import { TerminalChart } from "../terminal/TerminalChart";
import { TrendingUp, TrendingDown, Activity, Clock } from "lucide-react";

export function BacktestDashboard() {
    const [config, setConfig] = useState<BacktestConfig>({
        startDate: "2023-01-01",
        endDate: "2023-12-31",
        initialCapital: 10000,
        symbol: "BTC-USDC",
        timeframe: "1h",
    });

    const [isRunning, setIsRunning] = useState(false);
    const [hasRun, setHasRun] = useState(false);
    const [resultData, setResultData] = useState<any[]>([]);

    const runBacktest = () => {
        setIsRunning(true);

        // Simulate async job
        setTimeout(() => {
            // Generate mock equity curve
            const data = [];
            let equity = config.initialCapital;
            const points = 100;
            const now = Math.floor(Date.now() / 1000);

            for (let i = 0; i < points; i++) {
                // Random walk with drift
                const change = (Math.random() - 0.45) * (equity * 0.02);
                equity += change;
                data.push({
                    time: now - (points - i) * 3600,
                    value: equity
                });
            }

            setResultData(data);
            setIsRunning(false);
            setHasRun(true);
        }, 1500);
    };

    // Calculate metrics from resultData
    const finalEquity = resultData.length > 0 ? resultData[resultData.length - 1].value : config.initialCapital;
    const totalReturn = ((finalEquity - config.initialCapital) / config.initialCapital) * 100;
    const maxDrawdown = 12.5; // Mock
    const winRate = 58; // Mock

    return (
        <div className="flex h-full bg-slate-950 p-6 gap-6 overflow-hidden">
            {/* Left Panel: Config */}
            <div className="w-80 flex-shrink-0">
                <div className="mb-6">
                    <h2 className="text-2xl font-bold text-slate-100 mb-1">Backtest</h2>
                    <p className="text-slate-400 text-sm">Validate strategies against history.</p>
                </div>
                <BacktestConfigPanel
                    config={config}
                    onChange={setConfig}
                    onRun={runBacktest}
                    isRunning={isRunning}
                />
            </div>

            {/* Right Panel: Results */}
            <div className="flex-1 flex flex-col gap-4 overflow-y-auto min-h-0">
                {/* KPI Cards */}
                <div className="grid grid-cols-4 gap-4">
                    <MetricCard
                        label="Total Return"
                        value={`${totalReturn.toFixed(2)}%`}
                        subValue={hasRun ? `+${(finalEquity - config.initialCapital).toFixed(2)}` : "-"}
                        color={totalReturn >= 0 ? "text-emerald-400" : "text-rose-400"}
                        icon={totalReturn >= 0 ? TrendingUp : TrendingDown}
                    />
                    <MetricCard
                        label="Max Drawdown"
                        value={`${maxDrawdown}%`}
                        color="text-rose-400"
                        icon={Activity}
                    />
                    <MetricCard
                        label="Win Rate"
                        value={`${winRate}%`}
                        color="text-indigo-400"
                        icon={Activity}
                    />
                    <MetricCard
                        label="Trades"
                        value="142"
                        color="text-slate-300"
                        icon={Clock}
                    />
                </div>

                {/* Equity Chart */}
                <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 p-4 min-h-[400px] flex flex-col">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Equity Curve</h3>
                    <div className="flex-1">
                        {hasRun ? (
                            <TerminalChart
                                data={resultData}
                                color={totalReturn >= 0 ? "#34d399" : "#fb7185"}
                            />
                        ) : (
                            <div className="h-full flex items-center justify-center text-slate-600 text-sm italic">
                                Run a backtest to see results
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

function MetricCard({ label, value, subValue, color, icon: Icon }: any) {
    return (
        <div className="bg-slate-900 rounded-xl p-4 border border-slate-800 flex items-center justify-between">
            <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">{label}</p>
                <div className={`text-xl font-bold ${color}`}>{value}</div>
                {subValue && <div className="text-xs text-slate-500 font-medium">{subValue}</div>}
            </div>
            <div className="p-2 bg-slate-800 rounded-lg">
                <Icon size={18} className="text-slate-400" />
            </div>
        </div>
    )
}
