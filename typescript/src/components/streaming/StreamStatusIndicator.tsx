
import React, { useMemo } from 'react';
import { Activity, Zap, RefreshCw, AlertCircle, Wifi, WifiOff } from 'lucide-react';
import { useStreaming } from '../../context/StreamingContext';

/**
 * A premium status indicator for real-time data streaming health.
 * Displays latency, message frequency, and connection status.
 */
export const StreamStatusIndicator: React.FC = () => {
    const { stats, isGlobalStreamingEnabled } = useStreaming();

    const statusConfig = useMemo(() => {
        switch (stats.status) {
            case 'connected':
                return {
                    color: 'text-emerald-400',
                    bgColor: 'bg-emerald-400/10',
                    borderColor: 'border-emerald-400/20',
                    icon: <Wifi className="w-4 h-4" />,
                    label: 'Live',
                    pulse: true
                };
            case 'connecting':
                return {
                    color: 'text-sky-400',
                    bgColor: 'bg-sky-400/10',
                    borderColor: 'border-sky-400/20',
                    icon: <RefreshCw className="w-4 h-4 animate-spin" />,
                    label: 'Connecting',
                    pulse: false
                };
            case 'retrying':
                return {
                    color: 'text-amber-400',
                    bgColor: 'bg-amber-400/10',
                    borderColor: 'border-amber-400/20',
                    icon: <RefreshCw className="w-4 h-4 animate-spin" />,
                    label: 'Retrying',
                    pulse: false
                };
            case 'idle':
            default:
                return {
                    color: 'text-slate-400',
                    bgColor: 'bg-slate-400/10',
                    borderColor: 'border-slate-400/20',
                    icon: <WifiOff className="w-4 h-4" />,
                    label: 'Idle',
                    pulse: false
                };
        }
    }, [stats.status]);

    if (!isGlobalStreamingEnabled) {
        return (
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50 text-slate-500 text-xs font-medium">
                <AlertCircle className="w-3.5 h-3.5" />
                <span>Streaming Disabled</span>
            </div>
        );
    }

    return (
        <div className="flex items-center space-x-4">
            {/* Main Status Badge */}
            <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full ${statusConfig.bgColor} border ${statusConfig.borderColor} ${statusConfig.color} transition-all duration-300`}>
                <div className="relative flex items-center justify-center">
                    {statusConfig.icon}
                    {statusConfig.pulse && (
                        <span className="absolute inset-0 rounded-full animate-ping bg-current opacity-20"></span>
                    )}
                </div>
                <span className="text-xs font-bold uppercase tracking-wider">{statusConfig.label}</span>
            </div>

            {/* Health Metrics (Only when connected) */}
            {stats.status === 'connected' && (
                <div className="hidden md:flex items-center space-x-4 text-[10px] uppercase tracking-widest font-bold">
                    <div className="flex items-center space-x-1.5 text-slate-400 hover:text-sky-400 transition-colors cursor-default group">
                        <Zap className="w-3.5 h-3.5 group-hover:animate-pulse" />
                        <span>{stats.latencyMs}ms</span>
                        <span className="text-slate-600 font-normal">Latency</span>
                    </div>

                    <div className="flex items-center space-x-1.5 text-slate-400 hover:text-emerald-400 transition-colors cursor-default group">
                        <Activity className="w-3.5 h-3.5 group-hover:animate-bounce" />
                        <span>{stats.msgsPerSec.toFixed(1)}/s</span>
                        <span className="text-slate-600 font-normal">Throughput</span>
                    </div>
                </div>
            )}

            {/* Status Message (For retrying/connecting) */}
            {(stats.status === 'retrying' || stats.status === 'connecting') && (
                <span className="text-[10px] text-slate-500 animate-pulse truncate max-w-[150px]">
                    {stats.statusMessage}
                </span>
            )}
        </div>
    );
};
