
import { useState } from "react";
import { CloudDownload, Database, FileText, RefreshCw, Search } from "lucide-react";

interface Dataset {
    id: string;
    name: string;
    exchange: string;
    type: "OHLCV" | "Trades" | "OrderBook";
    range: string;
    size: string;
    status: "available" | "downloaded" | "downloading";
}

export function DataCatalog() {
    const [datasets, setDatasets] = useState<Dataset[]>([
        { id: "1", name: "BTC-USDC 1m Candles", exchange: "Binance", type: "OHLCV", range: "2020-2023", size: "450 MB", status: "downloaded" },
        { id: "2", name: "ETH-USDC 1m Candles", exchange: "Binance", type: "OHLCV", range: "2020-2023", size: "380 MB", status: "available" },
        { id: "3", name: "SOL-USDC 1m Candles", exchange: "Binance", type: "OHLCV", range: "2021-2023", size: "220 MB", status: "available" },
        { id: "4", name: "Polymarket Event Dump", exchange: "Polymarket", type: "Trades", range: "2024-Q1", size: "1.2 GB", status: "available" },
    ]);

    const [search, setSearch] = useState("");

    const handleDownload = (id: string) => {
        setDatasets(datasets.map(d => d.id === id ? { ...d, status: "downloading" } : d));
        setTimeout(() => {
            setDatasets(prev => prev.map(d => d.id === id ? { ...d, status: "downloaded" } : d));
        }, 2000);
    };

    return (
        <div className="bg-slate-950 p-6 h-full flex flex-col">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
                        <Database size={24} className="text-indigo-400" />
                        Data Catalog
                    </h2>
                    <p className="text-slate-400 text-sm">Manage historical datasets for backtesting.</p>
                </div>

                <button className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded text-sm text-slate-200 transition-colors">
                    <RefreshCw size={16} /> Sync Index
                </button>
            </div>

            {/* Filters */}
            <div className="mb-6 flex gap-4">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-2.5 text-slate-500" size={16} />
                    <input
                        type="text"
                        placeholder="Search datasets..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 outline-none focus:border-indigo-500"
                    />
                </div>
                <div className="flex gap-2">
                    <select className="bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-300 outline-none">
                        <option>All Exchanges</option>
                        <option>Binance</option>
                        <option>Polymarket</option>
                    </select>
                    <select className="bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-300 outline-none">
                        <option>All Types</option>
                        <option>OHLCV</option>
                        <option>Trades</option>
                    </select>
                </div>
            </div>

            {/* Table */}
            <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
                <table className="w-full text-left text-sm">
                    <thead className="bg-slate-800/50 text-slate-400 font-medium uppercase text-xs">
                        <tr>
                            <th className="px-4 py-3">Dataset Name</th>
                            <th className="px-4 py-3">Exchange</th>
                            <th className="px-4 py-3">Type</th>
                            <th className="px-4 py-3">Range</th>
                            <th className="px-4 py-3">Size</th>
                            <th className="px-4 py-3 text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {datasets.filter(d => d.name.toLowerCase().includes(search.toLowerCase())).map(d => (
                            <tr key={d.id} className="hover:bg-slate-800/30 transition-colors">
                                <td className="px-4 py-3 font-medium text-slate-200 flex items-center gap-2">
                                    <FileText size={16} className="text-slate-500" />
                                    {d.name}
                                </td>
                                <td className="px-4 py-3 text-slate-400">{d.exchange}</td>
                                <td className="px-4 py-3">
                                    <span className="px-2 py-0.5 bg-slate-800 border border-slate-700 rounded text-xs text-slate-300">
                                        {d.type}
                                    </span>
                                </td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs">{d.range}</td>
                                <td className="px-4 py-3 text-slate-500 text-xs">{d.size}</td>
                                <td className="px-4 py-3 text-right">
                                    {d.status === "downloaded" ? (
                                        <span className="text-emerald-400 text-xs font-bold px-3 py-1 bg-emerald-500/10 rounded-full border border-emerald-500/20">Ready</span>
                                    ) : d.status === "downloading" ? (
                                        <span className="text-indigo-400 text-xs font-bold animate-pulse">Downloading...</span>
                                    ) : (
                                        <button
                                            onClick={() => handleDownload(d.id)}
                                            className="text-indigo-400 hover:text-indigo-300 p-1 rounded hover:bg-indigo-500/10 transition-colors"
                                            title="Download"
                                        >
                                            <CloudDownload size={18} />
                                        </button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
