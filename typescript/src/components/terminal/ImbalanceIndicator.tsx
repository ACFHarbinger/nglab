
import clsx from "clsx";

interface ImbalanceIndicatorProps {
    bidVolume: number;
    askVolume: number;
}

export function ImbalanceIndicator({ bidVolume, askVolume }: ImbalanceIndicatorProps) {
    const total = bidVolume + askVolume;
    const bidPct = total > 0 ? (bidVolume / total) * 100 : 50;
    const askPct = total > 0 ? (askVolume / total) * 100 : 50;

    return (
        <div className="flex flex-col gap-1 w-full px-2 py-2">
            <div className="flex justify-between text-[10px] text-slate-500 uppercase tracking-wider font-bold">
                <span className="text-emerald-500">{bidPct.toFixed(0)}% Bids</span>
                <span className="text-rose-500">{askPct.toFixed(0)}% Asks</span>
            </div>
            <div className="flex w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                    className="h-full bg-emerald-500 transition-all duration-500 ease-out"
                    style={{ width: `${bidPct}%` }}
                />
                <div
                    className="h-full bg-rose-500 transition-all duration-500 ease-out"
                    style={{ width: `${askPct}%` }}
                />
            </div>
        </div>
    );
}
