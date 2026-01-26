/**
 * @module components/terminal/PaperTradingDashboard
 * @description Dashboard for monitoring virtual account state in Paper Trading mode.
 */
import React from "react";
import { usePaperTrading } from "../../hooks/usePaperTrading";
import { TrendingUp, TrendingDown, DollarSign, Briefcase, History, RefreshCcw } from "lucide-react";

export function PaperTradingDashboard() {
    const { account, isActive, toggleMode, resetAccount } = usePaperTrading();

    if (!account) return null;

    const pnl = account.equity - 100000; // Assuming 100k start for now
    const isProfit = pnl >= 0;

    return (
        <div className="flex flex-col h-full bg-slate-950 p-6 space-y-6 overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-indigo-500 animate-pulse" />
                        Paper Trading Dashboard
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">Institutional-grade virtual order execution and portfolio tracking.</p>
                </div>
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => resetAccount(100000)}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-medium transition-all"
                    >
                        <RefreshCcw size={16} />
                        Reset Account
                    </button>
                    <div className="flex items-center gap-3 bg-slate-900 px-4 py-2 rounded-xl border border-slate-800">
                        <span className="text-sm font-medium text-slate-400">Paper Mode</span>
                        <button
                            onClick={() => toggleMode(!isActive)}
                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${isActive ? 'bg-indigo-600' : 'bg-slate-700'}`}
                        >
                            <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${isActive ? 'translate-x-6' : 'translate-x-1'}`} />
                        </button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard
                    label="Total Equity"
                    value={`$${account.equity.toLocaleString()}`}
                    subValue={pnl >= 0 ? `+${pnl.toLocaleString()}` : pnl.toLocaleString()}
                    isPositive={isProfit}
                    icon={<Briefcase className="text-indigo-400" />}
                />
                <StatCard
                    label="Cash Balance"
                    value={`$${account.balance.toLocaleString()}`}
                    subValue={`${((account.balance / account.equity) * 100).toFixed(1)}% liquidity`}
                    icon={<DollarSign className="text-emerald-400" />}
                />
                <StatCard
                    label="Open Positions"
                    value={Object.keys(account.positions).length.toString()}
                    subValue="Active Markets"
                    icon={<TrendingUp className="text-amber-400" />}
                />
            </div>

            <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-0">
                <div className="bg-slate-900/50 rounded-2xl border border-slate-800 flex flex-col min-h-0">
                    <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
                        <h3 className="font-bold flex items-center gap-2">
                            <TrendingUp size={18} className="text-indigo-400" />
                            Active Positions
                        </h3>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4">
                        {Object.keys(account.positions).length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-2">
                                <p>No open paper positions</p>
                                <p className="text-xs">Place an order with Paper Mode enabled.</p>
                            </div>
                        ) : (
                            <table className="w-full text-left text-sm">
                                <thead>
                                    <tr className="text-slate-500 border-b border-slate-800">
                                        <th className="pb-3 font-medium">Asset</th>
                                        <th className="pb-3 font-medium">Quantity</th>
                                        <th className="pb-3 font-medium text-right">Side</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/50">
                                    {Object.entries(account.positions).map(([symbol, qty]) => (
                                        <tr key={symbol}>
                                            <td className="py-4 font-bold text-white uppercase">{symbol}</td>
                                            <td className="py-4 font-mono">{qty.toLocaleString()}</td>
                                            <td className="py-4 text-right">
                                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${qty > 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
                                                    {qty > 0 ? "Long" : "Short"}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>

                <div className="bg-slate-900/50 rounded-2xl border border-slate-800 flex flex-col min-h-0">
                    <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
                        <h3 className="font-bold flex items-center gap-2">
                            <History size={18} className="text-indigo-400" />
                            Pending Paper Orders
                        </h3>
                        <span className="text-xs text-slate-500">{account.orders.length} Active</span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4">
                        {account.orders.length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-2">
                                <p>No pending paper orders</p>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {account.orders.map((order: any, idx: number) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800/50">
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${order.side === "Bid" ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
                                                    {order.side === "Bid" ? "Buy" : "Sell"}
                                                </span>
                                                <span className="font-bold text-white leading-none">${order.price.toFixed(3)}</span>
                                            </div>
                                            <p className="text-[10px] text-slate-500 mt-1 uppercase">Limit • {order.quantity} units</p>
                                        </div>
                                        <div className="text-right">
                                            <span className="text-[10px] text-slate-500 block">ID: #{order.id || idx}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

function StatCard({ label, value, subValue, icon, isPositive }: { label: string; value: string; subValue: string; icon: React.ReactNode; isPositive?: boolean }) {
    return (
        <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 hover:border-indigo-500/30 transition-all group">
            <div className="flex justify-between items-start mb-4">
                <span className="text-slate-400 text-sm font-medium uppercase tracking-wider">{label}</span>
                <div className="p-2 bg-slate-950 rounded-lg group-hover:scale-110 transition-transform">{icon}</div>
            </div>
            <div className="flex flex-col">
                <span className="text-3xl font-mono font-bold text-white">{value}</span>
                <div className="flex items-center gap-1.5 mt-1">
                    {isPositive !== undefined && (isPositive ? <TrendingUp size={12} className="text-emerald-400" /> : <TrendingDown size={12} className="text-rose-400" />)}
                    <span className={`text-xs font-semibold ${isPositive === undefined ? 'text-slate-500' : isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {subValue}
                    </span>
                </div>
            </div>
        </div>
    );
}
