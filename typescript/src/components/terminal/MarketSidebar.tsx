import { Search, TrendingUp, TrendingDown, Star } from "lucide-react";
import clsx from "clsx";

interface Market {
    id: string;
    symbol: string;
    name: string;
    price: number;
    change24h: number;
    volume24h: number;
    isFavorite?: boolean;
}

interface MarketSidebarProps {
    markets: Market[];
    activeMarketId?: string;
    onSelectMarket: (id: string) => void;
}

export function MarketSidebar({ markets, activeMarketId, onSelectMarket }: MarketSidebarProps) {
    return (
        <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800 w-80">
            {/* Header / Search */}
            <div className="p-3 border-b border-slate-800">
                <div className="relative">
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-500" />
                    <input
                        type="text"
                        placeholder="Search markets..."
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    />
                </div>
            </div>

            {/* Market List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
                {markets.map((market) => (
                    <div
                        key={market.id}
                        onClick={() => onSelectMarket(market.id)}
                        className={clsx(
                            "group px-3 py-3 border-b border-slate-800/50 cursor-pointer transition-colors hover:bg-slate-800/50",
                            activeMarketId === market.id ? "bg-slate-800 border-l-2 border-l-indigo-500" : "border-l-2 border-l-transparent"
                        )}
                    >
                        <div className="flex justify-between items-start mb-1">
                            <div className="flex items-center gap-2">
                                <span className={clsx("font-bold text-sm", activeMarketId === market.id ? "text-white" : "text-slate-300 group-hover:text-white")}>
                                    {market.symbol}
                                </span>
                                {market.isFavorite && <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />}
                            </div>
                            <span className="font-mono text-sm text-white">
                                {market.price < 1 ? market.price.toFixed(3) : market.price.toFixed(2)}
                            </span>
                        </div>

                        <div className="flex justify-between items-center text-xs">
                            <span className="text-slate-500 truncate max-w-[120px]" title={market.name}>{market.name}</span>
                            <div className="flex items-center gap-1">
                                <span className="text-slate-600">${(market.volume24h / 1000).toFixed(1)}k</span>
                                <span className={clsx("flex items-center", market.change24h >= 0 ? "text-green-400" : "text-rose-400")}>
                                    {market.change24h >= 0 ? <TrendingUp size={10} className="mr-0.5" /> : <TrendingDown size={10} className="mr-0.5" />}
                                    {Math.abs(market.change24h).toFixed(2)}%
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
