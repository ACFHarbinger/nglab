import { useState, useEffect } from "react";
import {
    Activity,
    Cpu,
    Database,
    Play,
    Square,
    RefreshCcw,
    Trash2,
    ChevronRight,
    ChevronDown,
    BarChart2,
    Clock,
    Settings
} from "lucide-react";
import clsx from "clsx";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from "recharts";

interface TrainingJob {
    id: string;
    model: string;
    status: "running" | "queued" | "completed" | "failed" | "stopped";
    progress: number;
    currentEpoch: number;
    totalEpochs: number;
    loss: number;
    reward: number;
    startTime: string;
    duration: string;
}

const mockJobs: TrainingJob[] = [
    {
        id: "job-001",
        model: "TransformerV2",
        status: "running",
        progress: 45,
        currentEpoch: 45,
        totalEpochs: 100,
        loss: 0.245,
        reward: 0.12,
        startTime: "2026-01-26 01:00",
        duration: "21m 15s"
    },
    {
        id: "job-002",
        model: "MambaTemporal",
        status: "queued",
        progress: 0,
        currentEpoch: 0,
        totalEpochs: 200,
        loss: 0,
        reward: 0,
        startTime: "-",
        duration: "-"
    }
];

const mockMetrics = Array.from({ length: 50 }, (_, i) => ({
    epoch: i,
    loss: 0.5 * Math.exp(-i / 10) + Math.random() * 0.05,
    reward: 0.2 * (1 - Math.exp(-i / 15)) + Math.random() * 0.02,
}));

export function TrainingDashboard() {
    const [jobs, setJobs] = useState<TrainingJob[]>(mockJobs);
    const [selectedJobId, setSelectedJobId] = useState<string>("job-001");
    const [logs, setLogs] = useState<string[]>([
        "[01:00:05] Initializing engine with CUDA 12.1...",
        "[01:00:10] Loading dataset: BTC-USD-1m-2025.parquet",
        "[01:00:15] Model architecture: Transformer (12 layers, 512 hidden)",
        "[01:01:00] Starting epoch 1/100...",
        "[01:01:45] Epoch 1 completed. Loss: 0.482, Reward: -0.05",
        "[01:21:00] Epoch 45 completed. Loss: 0.245, Reward: 0.12",
        "[01:21:24] Resuming training flow..."
    ]);

    const selectedJob = jobs.find(j => j.id === selectedJobId);

    return (
        <div className="flex flex-col h-full bg-slate-950 p-6 space-y-6 overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Activity className="text-indigo-400" />
                        Training Dashboard
                    </h2>
                    <p className="text-slate-400 text-sm">Monitor and manage your deep learning model training pipelines.</p>
                </div>
                <div className="flex items-center gap-3">
                    <button className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition-all shadow-lg shadow-indigo-500/20">
                        <Play size={16} /> New Training Job
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6 flex-1">
                {/* Left Column: Job Queue & Config */}
                <div className="col-span-12 lg:col-span-4 space-y-6">
                    {/* Job Queue */}
                    <div className="bg-slate-900/50 rounded-xl border border-slate-800 flex flex-col overflow-hidden">
                        <div className="px-4 py-3 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
                            <h3 className="text-sm font-semibold text-slate-200">Job Queue</h3>
                            <RefreshCcw size={14} className="text-slate-500 hover:text-slate-300 cursor-pointer transition-colors" />
                        </div>
                        <div className="divide-y divide-slate-800 max-h-[400px] overflow-y-auto">
                            {jobs.map(job => (
                                <div
                                    key={job.id}
                                    onClick={() => setSelectedJobId(job.id)}
                                    className={clsx(
                                        "p-4 cursor-pointer transition-all hover:bg-slate-800/50 border-l-2",
                                        selectedJobId === job.id ? "bg-indigo-500/10 border-indigo-500" : "border-transparent"
                                    )}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-medium text-slate-100">{job.model}</span>
                                        <StatusBadge status={job.status} />
                                    </div>
                                    <div className="flex items-center gap-4 text-xs text-slate-400">
                                        <div className="flex items-center gap-1">
                                            <Clock size={12} /> {job.duration}
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Database size={12} /> {job.progress}%
                                        </div>
                                    </div>
                                    {job.status === "running" && (
                                        <div className="mt-3 w-full bg-slate-800 rounded-full h-1">
                                            <div
                                                className="bg-indigo-500 h-1 rounded-full animate-pulse"
                                                style={{ width: `${job.progress}%` }}
                                            ></div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Hyperparameters */}
                    {selectedJob && (
                        <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4 space-y-4">
                            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                                <Settings size={16} className="text-slate-400" />
                                Hyperparameters
                            </h3>
                            <div className="grid grid-cols-2 gap-3">
                                <ParamItem label="Architecture" value={selectedJob.model} />
                                <ParamItem label="Optimizer" value="AdamW" />
                                <ParamItem label="Learning Rate" value="3e-4" />
                                <ParamItem label="Batch Size" value="64" />
                                <ParamItem label="Hidden Dim" value="256" />
                                <ParamItem label="Device" value="cuda:0" />
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column: Metrics & Logs */}
                <div className="col-span-12 lg:col-span-8 space-y-6">
                    {/* Real-time Metrics */}
                    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4 min-h-[400px] flex flex-col">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                                <BarChart2 size={16} className="text-slate-400" />
                                Training Metrics
                            </h3>
                            <div className="flex gap-4">
                                <div className="flex items-center gap-2 text-xs">
                                    <div className="w-3 h-3 bg-indigo-500 rounded-full"></div>
                                    <span className="text-slate-400">Loss</span>
                                </div>
                                <div className="flex items-center gap-2 text-xs">
                                    <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                                    <span className="text-slate-400">Reward</span>
                                </div>
                            </div>
                        </div>

                        <div className="flex-1 w-full">
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={mockMetrics}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis yAxisId="left" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                                        itemStyle={{ fontSize: '12px' }}
                                    />
                                    <Line
                                        yAxisId="left"
                                        type="monotone"
                                        dataKey="loss"
                                        stroke="#6366f1"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 4 }}
                                    />
                                    <Line
                                        yAxisId="right"
                                        type="monotone"
                                        dataKey="reward"
                                        stroke="#10b981"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 4 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Logs */}
                    <div className="bg-slate-900/50 rounded-xl border border-slate-800 flex flex-col h-[280px]">
                        <div className="px-4 py-3 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
                            <h3 className="text-sm font-semibold text-slate-200">Execution Logs</h3>
                            <div className="flex gap-2">
                                <button className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded border border-slate-700">Clear</button>
                                <button className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded border border-slate-700">Export</button>
                            </div>
                        </div>
                        <div className="flex-1 p-4 font-mono text-[11px] text-slate-300 overflow-y-auto space-y-1">
                            {logs.map((log, i) => (
                                <div key={i} className="hover:bg-slate-800/30 px-2 py-0.5 rounded transition-colors break-all">
                                    <span className="text-indigo-400 opacity-50 mr-2">{i + 1}</span>
                                    {log}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function StatusBadge({ status }: { status: TrainingJob["status"] }) {
    const styles = {
        running: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
        queued: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
        completed: "bg-blue-500/10 text-blue-400 border-blue-500/20",
        failed: "bg-rose-500/10 text-rose-400 border-rose-500/20",
        stopped: "bg-slate-500/10 text-slate-400 border-slate-500/20",
    };

    return (
        <span className={clsx("px-2 py-0.5 rounded text-[10px] font-bold uppercase border", styles[status])}>
            {status}
        </span>
    );
}

function ParamItem({ label, value }: { label: string; value: string | number }) {
    return (
        <div className="bg-slate-800/30 p-2 rounded border border-slate-800/50">
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">{label}</div>
            <div className="text-sm text-slate-100 font-medium truncate">{value}</div>
        </div>
    );
}
