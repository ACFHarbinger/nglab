import React, { useState } from 'react';
// Since I don't know where cn is, I'll use clsx and tailwind-merge directly or simplified.
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface OptionData {
    strike: number;
    expiry: string;
    bid: number;
    ask: number;
    last: number;
    vol: number;
    delta: number;
    gamma: number;
    iv: number;
}

interface OptionsChainProps {
    underlyingPrice: number;
    calls: OptionData[];
    puts: OptionData[];
}

export const OptionsChain: React.FC<OptionsChainProps> = ({ underlyingPrice, calls, puts }) => {
    const [selectedExpiry, setSelectedExpiry] = useState<string>('All');

    // Group by expiry
    const expiries = Array.from(new Set(calls.map(c => c.expiry))).sort();

    const filteredCalls = selectedExpiry === 'All' ? calls : calls.filter(c => c.expiry === selectedExpiry);
    const filteredPuts = selectedExpiry === 'All' ? puts : puts.filter(p => p.expiry === selectedExpiry);

    // Sort by strike
    const sortedStrikes = Array.from(new Set([...filteredCalls, ...filteredPuts].map(o => o.strike))).sort((a, b) => a - b);

    const getOption = (list: OptionData[], strike: number) => list.find(o => o.strike === strike);

    return (
        <div className="flex flex-col h-full bg-slate-900 text-slate-200 p-4 rounded-lg overflow-hidden">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-white">Options Chain</h2>
                <div className="flex gap-2 items-center">
                    <span className="text-sm text-slate-400">Expiry:</span>
                    <select
                        className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm outline-none focus:border-blue-500"
                        value={selectedExpiry}
                        onChange={(e) => setSelectedExpiry(e.target.value)}
                    >
                        <option value="All">All</option>
                        {expiries.map(e => <option key={e} value={e}>{e}</option>)}
                    </select>
                </div>
            </div>

            <div className="flex-1 overflow-auto">
                <table className="w-full text-xs text-right border-collapse">
                    <thead className="bg-slate-800 text-slate-400 sticky top-0 z-10">
                        <tr>
                            <th colSpan={5} className="py-2 px-2 text-center border-b border-slate-700 text-green-400 font-bold">Calls</th>
                            <th className="py-2 px-4 text-center border-b border-slate-700 text-white bg-slate-800">Strike</th>
                            <th colSpan={5} className="py-2 px-2 text-center border-b border-slate-700 text-red-400 font-bold">Puts</th>
                        </tr>
                        <tr>
                            <th className="py-1 px-2">Bid</th>
                            <th className="py-1 px-2">Ask</th>
                            <th className="py-1 px-2">IV</th>
                            <th className="py-1 px-2">Delta</th>
                            <th className="py-1 px-2">Vol</th>

                            <th className="py-1 px-4 text-center bg-slate-800"></th>

                            <th className="py-1 px-2">Bid</th>
                            <th className="py-1 px-2">Ask</th>
                            <th className="py-1 px-2">IV</th>
                            <th className="py-1 px-2">Delta</th>
                            <th className="py-1 px-2">Vol</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedStrikes.map(strike => {
                            const call = getOption(filteredCalls, strike);
                            const put = getOption(filteredPuts, strike);
                            const isITM_Call = underlyingPrice > strike;
                            const isITM_Put = underlyingPrice < strike;

                            return (
                                <tr key={strike} className="hover:bg-slate-800/50 border-b border-slate-800/50">
                                    {/* Calls */}
                                    <td className={cn("py-1.5 px-2", isITM_Call ? "bg-green-900/20 text-green-300" : "")}>{call?.bid.toFixed(2) || '-'}</td>
                                    <td className={cn("py-1.5 px-2", isITM_Call ? "bg-green-900/20 text-green-300" : "")}>{call?.ask.toFixed(2) || '-'}</td>
                                    <td className={cn("py-1.5 px-2 text-slate-500", isITM_Call ? "bg-green-900/20" : "")}>{call ? (call.iv * 100).toFixed(1) + '%' : '-'}</td>
                                    <td className={cn("py-1.5 px-2 text-slate-400", isITM_Call ? "bg-green-900/20" : "")}>{call?.delta.toFixed(2) || '-'}</td>
                                    <td className={cn("py-1.5 px-2", isITM_Call ? "bg-green-900/20" : "")}>{call?.vol || '-'}</td>

                                    {/* Strike */}
                                    <td className={cn(
                                        "py-1.5 px-4 text-center font-bold sticky left-0 z-5",
                                        strike === underlyingPrice ? "bg-blue-900 text-white" : "bg-slate-800 text-slate-300"
                                    )}>
                                        {strike}
                                    </td>

                                    {/* Puts */}
                                    <td className={cn("py-1.5 px-2", isITM_Put ? "bg-red-900/20 text-red-300" : "")}>{put?.bid.toFixed(2) || '-'}</td>
                                    <td className={cn("py-1.5 px-2", isITM_Put ? "bg-red-900/20 text-red-300" : "")}>{put?.ask.toFixed(2) || '-'}</td>
                                    <td className={cn("py-1.5 px-2 text-slate-500", isITM_Put ? "bg-red-900/20" : "")}>{put ? (put.iv * 100).toFixed(1) + '%' : '-'}</td>
                                    <td className={cn("py-1.5 px-2 text-slate-400", isITM_Put ? "bg-red-900/20" : "")}>{put?.delta.toFixed(2) || '-'}</td>
                                    <td className={cn("py-1.5 px-2", isITM_Put ? "bg-red-900/20" : "")}>{put?.vol || '-'}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
