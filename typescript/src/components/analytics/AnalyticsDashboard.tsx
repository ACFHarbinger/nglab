
import { useMemo } from "react";
import clsx from "clsx";
import { TrendingUp, TrendingDown, Activity, AlertTriangle, DollarSign, Percent } from "lucide-react";

interface StepInfo {
    portfolio_value: number;
    position: number;
    cash: number;
    sharpe_ratio: number;
    total_steps: number;
    pnl: number;
    return_pct: number;
    drawdown: number;
    max_drawdown: number;
    volatility: number;
}

interface AnalyticsDashboardProps {
    info: StepInfo;
}

export function AnalyticsDashboard({ info }: AnalyticsDashboardProps) {
    // Format helpers
    const fmtCurrency = (val: number) => `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    const fmtPct = (val: number) => `${(val * 100).toFixed(2)}%`;
    const fmtNum = (val: number) => val.toLocaleString(undefined, { maximumFractionDigits: 2 });

    return (
        <div className="flex flex-col h-full bg-slate-900 overflow-y-auto p-6 gap-6">
            <div className="flex justify-between items-center mb-2">
                <h2 className="text-xl font-bold text-slate-100 uppercase tracking-wide">Real-Time Analytics</h2>
                <span className="text-xs text-slate-500 font-mono">Step: {info.total_steps}</span>
            </div>

            {/* KPI Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                    label="Net P&L"
                    value={fmtCurrency(info.pnl)}
                    subValue={fmtPct(info.return_pct)}
                    icon={DollarSign}
                    trend={info.pnl >= 0 ? "up" : "down"}
                />
                <StatCard
                    label="Sharpe Ratio"
                    value={info.sharpe_ratio.toFixed(2)}
                    subValue="Rolling 30d"
                    icon={Activity}
                    color="blue"
                />
                <StatCard
                    label="Max Drawdown"
                    value={fmtPct(info.max_drawdown)}
                    subValue={`Current: ${fmtPct(info.drawdown)}`}
                    icon={TrendingDown}
                    color="rose"
                    inverse
                />
                <StatCard
                    label="Volatility"
                    value={fmtPct(info.volatility)}
                    subValue="Annualized"
                    icon={AlertTriangle}
                    color="amber"
                    inverse
                />
            </div>

            {/* Detailed Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                    <h3 className="text-sm font-semibold text-slate-400 mb-4 uppercase">Portfolio Status</h3>
                    <div className="space-y-3">
                        <Row label="Total Equity" value={fmtCurrency(info.portfolio_value)} />
                        <Row label="Cash Balance" value={fmtCurrency(info.cash)} />
                        <Row label="Position Value" value={fmtCurrency(info.position * 100.0)} /> {/* Est. Price derived? or just show raw pos for now */}
                        <Row label="Position Size" value={fmtNum(info.position)} />
                    </div>
                </div>

                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                    <h3 className="text-sm font-semibold text-slate-400 mb-4 uppercase">Risk Metrics</h3>
                    <div className="space-y-3">
                        <Row label="Current Drawdown" value={fmtPct(info.drawdown)} highlight={info.drawdown > 0.05} />
                        <Row label="Max Drawdown" value={fmtPct(info.max_drawdown)} />
                        <Row label="Sharpe Ratio" value={info.sharpe_ratio.toFixed(3)} />
                        <Row label="Volatility" value={fmtPct(info.volatility)} />
                    </div>
                </div>
            </div>
        </div>
    );
}

function StatCard({ label, value, subValue, icon: Icon, trend, color = "emerald", inverse = false }: any) {
    const isPositive = trend === "up";
    const isNegative = trend === "down";

    // Determine color based on trend or fixed color
    let valColor = "text-slate-100";
    if (trend) {
        valColor = isPositive ? "text-emerald-400" : "text-rose-400";
    } else if (color === "blue") valColor = "text-blue-400";
    else if (color === "amber") valColor = "text-amber-400";
    else if (color === "rose") valColor = "text-rose-400";

    // Inverse logic simply flips semantic meaning if needed, but here we just rely on explicit color/trend

    return (
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 flex items-start justify-between hover:bg-slate-750 transition-colors">
            <div>
                <p className="text-xs text-slate-400 font-medium uppercase mb-1">{label}</p>
                <div className={clsx("text-2xl font-bold font-mono tracking-tight", valColor)}>{value}</div>
                {subValue && <div className="text-xs text-slate-500 mt-1">{subValue}</div>}
            </div>
            <div className={clsx("p-2 rounded-lg bg-slate-700/50", valColor)}>
                <Icon size={20} />
            </div>
        </div>
    )
}

function Row({ label, value, highlight }: { label: string, value: string, highlight?: boolean }) {
    return (
        <div className="flex justify-between items-center py-2 border-b border-slate-700/50 last:border-0">
            <span className="text-sm text-slate-400">{label}</span>
            <span className={clsx("text-sm font-mono font-medium", highlight ? "text-rose-400" : "text-slate-200")}>{value}</span>
        </div>
    )
}
