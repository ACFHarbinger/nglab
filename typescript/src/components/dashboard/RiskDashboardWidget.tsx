/**
 * @module components/dashboard/RiskDashboardWidget
 * @description Real-time visualization of portfolio risk metrics including VaR and Drawdown.
 */
import { AlertTriangle, ShieldCheck, TrendingDown, Activity } from "lucide-react";
import clsx from "clsx";

/**
 * Props for the RiskDashboardWidget.
 */
interface RiskDashboardWidgetProps {
    /** Combined risk score (0-100), where 100 is max risk. */
    riskScore: number;
    /** Current portfolio drawdown (0.0 to 1.0). */
    drawdown: number;
    /** Value at Risk (VaR) estimate (e.g., 95% confidence). */
    varValue: number;
}

/**
 * Widget visualizing key portfolio risk metrics.
 * Displays Risk Score, Drawdown, and VaR with color-coded severity indicators.
 */
export function RiskDashboardWidget({ riskScore, drawdown, varValue }: RiskDashboardWidgetProps) {
    const getRiskColor = (score: number) => {
        if (score < 30) return "text-emerald-400";
        if (score < 70) return "text-yellow-400";
        return "text-rose-400";
    };

    const getRiskBg = (score: number) => {
        if (score < 30) return "bg-emerald-500/10 border-emerald-500/20";
        if (score < 70) return "bg-yellow-500/10 border-yellow-500/20";
        return "bg-rose-500/10 border-rose-500/20";
    };

    return (
        <div className={clsx("p-4 rounded-xl border transition-all duration-500", getRiskBg(riskScore))}>
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    {riskScore < 70 ? (
                        <ShieldCheck className="text-emerald-400" size={20} />
                    ) : (
                        <AlertTriangle className="text-rose-400" size={20} />
                    )}
                    <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                        Portfolio Risk
                    </h3>
                </div>
                <div className={clsx("text-2xl font-black font-mono", getRiskColor(riskScore))}>
                    {riskScore}
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <div className="flex items-center gap-1.5 text-[10px] font-bold text-slate-500 uppercase mb-1">
                        <TrendingDown size={12} />
                        Drawdown
                    </div>
                    <div className="text-lg font-bold text-rose-400 font-mono">
                        {(drawdown * 100).toFixed(2)}%
                    </div>
                </div>

                <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <div className="flex items-center gap-1.5 text-[10px] font-bold text-slate-500 uppercase mb-1">
                        <Activity size={12} />
                        VaR (95%)
                    </div>
                    <div className="text-lg font-bold text-yellow-400 font-mono">
                        {(varValue * 100).toFixed(2)}%
                    </div>
                </div>
            </div>

            <div className="mt-4 overflow-hidden h-1.5 bg-slate-800 rounded-full">
                <div
                    className={clsx("h-full transition-all duration-1000",
                        riskScore < 30 ? "bg-emerald-500" : riskScore < 70 ? "bg-yellow-500" : "bg-rose-500"
                    )}
                    style={{ width: `${riskScore}%` }}
                />
            </div>
        </div>
    );
}
