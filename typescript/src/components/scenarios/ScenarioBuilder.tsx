import React, { useState } from 'react';
import { AlertTriangle, Play, RefreshCw, Zap } from 'lucide-react';

export const ScenarioBuilder: React.FC = () => {
    const [shockMagnitude, setShockMagnitude] = useState(0);
    const [volSpike, setVolSpike] = useState(0);
    const [selectedAsset, setSelectedAsset] = useState('BTC');

    const runScenario = (type: 'Shock' | 'Crash') => {
        console.log(`Running scenario: ${type} on ${selectedAsset}`, { shockMagnitude, volSpike });
        // TODO: Call backend API
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 text-slate-200 p-4 rounded-lg">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <Zap size={24} className="text-yellow-400" />
                Scenario Builder
            </h2>

            <div className="flex flex-col gap-6">
                {/* Asset Selection */}
                <div className="flex flex-col gap-2">
                    <label className="text-sm text-slate-400 font-bold">Target Asset</label>
                    <select
                        className="bg-slate-800 border border-slate-700 rounded p-2 text-white outline-none focus:border-blue-500"
                        value={selectedAsset}
                        onChange={(e) => setSelectedAsset(e.target.value)}
                    >
                        <option value="BTC">BTC</option>
                        <option value="ETH">ETH</option>
                        <option value="SOL">SOL</option>
                    </select>
                </div>

                {/* Price Shock Slider */}
                <div className="flex flex-col gap-2">
                    <div className="flex justify-between">
                        <label className="text-sm text-slate-400 font-bold">Price Shock</label>
                        <span className={shockMagnitude > 0 ? "text-green-400" : shockMagnitude < 0 ? "text-red-400" : "text-slate-400"}>
                            {shockMagnitude > 0 ? '+' : ''}{shockMagnitude}%
                        </span>
                    </div>
                    <input
                        type="range"
                        min="-50"
                        max="50"
                        step="1"
                        value={shockMagnitude}
                        onChange={(e) => setShockMagnitude(parseInt(e.target.value))}
                        className="w-full accent-blue-500"
                    />
                    <div className="flex justify-between text-xs text-slate-600 px-1">
                        <span>-50%</span>
                        <span>0%</span>
                        <span>+50%</span>
                    </div>
                </div>

                {/* Volatility Spike Input */}
                <div className="flex flex-col gap-2">
                    <div className="flex justify-between">
                        <label className="text-sm text-slate-400 font-bold">Volatility Spike (Abs)</label>
                        <span className="text-yellow-400">+{volSpike}%</span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={volSpike}
                        onChange={(e) => setVolSpike(parseInt(e.target.value))}
                        className="w-full accent-yellow-500"
                    />
                </div>

                {/* Actions */}
                <div className="grid grid-cols-2 gap-4 mt-4">
                    <button
                        onClick={() => runScenario('Shock')}
                        className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white p-3 rounded font-bold transition"
                    >
                        <Play size={18} /> Apply Shock
                    </button>
                    <button
                        onClick={() => runScenario('Crash')}
                        className="flex items-center justify-center gap-2 bg-red-600/20 hover:bg-red-600/40 text-red-400 border border-red-900 p-3 rounded font-bold transition"
                    >
                        <AlertTriangle size={18} /> Simulate Crash
                    </button>
                    <button
                        onClick={() => { setShockMagnitude(0); setVolSpike(0); }}
                        className="col-span-2 flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-400 p-2 rounded transition text-sm"
                    >
                        <RefreshCw size={14} /> Reset
                    </button>
                </div>
            </div>
        </div>
    );
};
