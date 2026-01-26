
import { useState } from "react";
import clsx from "clsx";
import { ChevronDown, BarChart2, Hash, Clock } from "lucide-react";

export type ChartType = "Area" | "Candle" | "HeikinAshi";
export type Timeframe = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

interface ChartToolbarProps {
    chartType: ChartType;
    setChartType: (t: ChartType) => void;
    timeframe: Timeframe;
    setTimeframe: (t: Timeframe) => void;
}

export function ChartToolbar({ chartType, setChartType, timeframe, setTimeframe }: ChartToolbarProps) {
    return (
        <div className="flex items-center gap-2 p-1.5 bg-slate-900/80 backdrop-blur rounded border border-slate-800 absolute top-2 left-1/2 -translate-x-1/2 z-10 shadow-lg">
            <Dropdown
                value={chartType}
                onChange={(v) => setChartType(v as ChartType)}
                options={[
                    { value: "Area", label: "Area", icon: BarChart2 },
                    { value: "Candle", label: "Candles", icon: BarChart2 },
                    { value: "HeikinAshi", label: "Heikin Ashi", icon: Hash },
                ]}
            />

            <div className="w-px h-4 bg-slate-700" />

            <Dropdown
                value={timeframe}
                onChange={(v) => setTimeframe(v as Timeframe)}
                options={[
                    { value: "1m", label: "1m" },
                    { value: "5m", label: "5m" },
                    { value: "15m", label: "15m" },
                    { value: "1h", label: "1H" },
                    { value: "4h", label: "4H" },
                    { value: "1d", label: "1D" },
                ]}
                icon={Clock}
            />
        </div>
    );
}

function Dropdown({ value, onChange, options, icon: Icon }: any) {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <div className="relative group">
            <button
                onClick={() => setIsOpen(!isOpen)}
                onBlur={() => setTimeout(() => setIsOpen(false), 200)}
                className="flex items-center gap-1.5 px-2 py-1 hover:bg-slate-800 rounded text-xs font-medium text-slate-300 transition-colors"
            >
                {Icon && <Icon size={14} className="text-slate-500" />}
                <span>{options.find((o: any) => o.value === value)?.label || value}</span>
                <ChevronDown size={12} className="text-slate-600" />
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 mt-1 w-32 bg-slate-900 border border-slate-700 rounded shadow-xl overflow-hidden z-20 animate-in fade-in zoom-in-95 duration-100">
                    {options.map((opt: any) => (
                        <button
                            key={opt.value}
                            onClick={() => {
                                onChange(opt.value);
                                setIsOpen(false);
                            }}
                            className={clsx(
                                "w-full text-left px-3 py-2 text-xs hover:bg-slate-800 flex items-center gap-2",
                                value === opt.value ? "text-indigo-400 font-semibold" : "text-slate-400"
                            )}
                        >
                            {opt.icon && <opt.icon size={12} />}
                            {opt.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}
