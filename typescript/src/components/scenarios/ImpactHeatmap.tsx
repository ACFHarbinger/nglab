import React from 'react';

export const ImpactHeatmap: React.FC = () => {
    // Mock Data for agents
    const agents = [
        { id: 'Momentum-1', activity: 0.8, pnl: 1200, impact: 'High' },
        { id: 'Noise-1', activity: 0.3, pnl: -200, impact: 'Low' },
        { id: 'MM-Alpha', activity: 0.9, pnl: 500, impact: 'Medium' },
        { id: 'Arb-Bot', activity: 0.1, pnl: 50, impact: 'Low' },
        { id: 'HFT-X', activity: 0.95, pnl: 3000, impact: 'High' },
    ];

    return (
        <div className="flex flex-col h-full bg-slate-900 text-slate-200 p-4 rounded-lg">
            <h2 className="text-xl font-bold text-white mb-4">Market Impact Heatmap</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {agents.map(agent => (
                    <div key={agent.id} className="bg-slate-800 p-4 rounded border border-slate-700 hover:border-blue-500 transition relative overflow-hidden">
                        <div className="flex justify-between items-start mb-2 relative z-10">
                            <span className="font-bold text-slate-300">{agent.id}</span>
                            <span className={`text-xs px-2 py-0.5 rounded ${agent.impact === 'High' ? 'bg-red-500/20 text-red-400' :
                                    agent.impact === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                        'bg-green-500/20 text-green-400'
                                }`}>
                                {agent.impact} Impact
                            </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-sm relative z-10">
                            <div>
                                <div className="text-slate-500 text-xs">Activity</div>
                                <div className="text-white font-mono">{(agent.activity * 100).toFixed(0)}%</div>
                            </div>
                            <div>
                                <div className="text-slate-500 text-xs">P&L</div>
                                <div className={`font-mono ${agent.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {agent.pnl >= 0 ? '+' : ''}{agent.pnl}
                                </div>
                            </div>
                        </div>

                        {/* Background Activity Bar */}
                        <div
                            className="absolute bottom-0 left-0 h-1 bg-blue-500 transition-all duration-1000"
                            style={{ width: `${agent.activity * 100}%` }}
                        ></div>
                        <div
                            className="absolute top-0 right-0 bottom-0 w-16 bg-gradient-to-l from-slate-700/10 to-transparent pointer-events-none"
                        ></div>
                    </div>
                ))}
            </div>
        </div>
    );
};
