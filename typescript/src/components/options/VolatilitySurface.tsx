import React from 'react';

interface VolatilityPoint {
    strike: number;
    expiry: string; // "YYYY-MM-DD"
    iv: number;
}

interface VolatilitySurfaceProps {
    data: VolatilityPoint[];
}

export const VolatilitySurface: React.FC<VolatilitySurfaceProps> = ({ data }) => {
    // 1. Get unique strikes and expiries
    const strikes = Array.from(new Set(data.map(d => d.strike))).sort((a, b) => a - b);
    const expiries = Array.from(new Set(data.map(d => d.expiry))).sort();

    // 2. Helper to get IV
    const getIV = (strike: number, expiry: string) => {
        const point = data.find(d => d.strike === strike && d.expiry === expiry);
        return point ? point.iv : 0;
    };

    // 3. Color mapping for Heatmap
    const getColor = (iv: number) => {
        // Simple scale: 0.1 (10%) -> blue, 0.5 (50%) -> green, 1.0 (100%) -> red
        if (iv < 0.2) return `rgba(59, 130, 246, ${0.3 + iv * 2})`; // Low vol blue
        if (iv < 0.5) return `rgba(34, 197, 94, ${0.3 + iv})`; // Mid vol green
        return `rgba(239, 68, 68, ${Math.min(1, 0.3 + iv)})`; // High vol red
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 text-slate-200 p-4 rounded-lg">
            <h2 className="text-xl font-bold text-white mb-4">Volatility Surface (Heatmap)</h2>
            <div className="flex-1 overflow-auto">
                <div className="grid gap-1" style={{
                    gridTemplateColumns: `auto repeat(${strikes.length}, minmax(40px, 1fr))`
                }}>
                    {/* Header Row: Strikes */}
                    <div className="p-2 text-xs font-bold text-slate-500">Expiry \ Strike</div>
                    {strikes.map(k => (
                        <div key={k} className="p-2 text-center text-xs font-bold text-slate-400 bg-slate-800 rounded">
                            {k}
                        </div>
                    ))}

                    {/* Rows: Expiries */}
                    {expiries.map(expiry => (
                        <React.Fragment key={expiry}>
                            <div className="p-2 text-xs font-bold text-slate-400 bg-slate-800 rounded flex items-center">
                                {expiry}
                            </div>
                            {strikes.map(strike => {
                                const iv = getIV(strike, expiry);
                                return (
                                    <div
                                        key={`${expiry}-${strike}`}
                                        className="p-2 rounded text-center text-xs text-white font-mono flex items-center justify-center transition-all hover:scale-105 cursor-default"
                                        style={{ backgroundColor: getColor(iv) }}
                                        title={`IV: ${(iv * 100).toFixed(1)}%`}
                                    >
                                        {iv > 0 ? (iv * 100).toFixed(0) : '-'}
                                    </div>
                                );
                            })}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </div>
    );
};
