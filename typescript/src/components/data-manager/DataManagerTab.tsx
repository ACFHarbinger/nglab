import { DataCatalog } from "./DataCatalog";
import { Download, Database, BarChart3, ShieldCheck, AlertCircle } from "lucide-react";

/**
 * Main tab for Historical Data Management (Roadmap 4.3).
 * Integrates the Data Catalog and provides tools for data quality and export.
 */
export default function DataManagerTab() {
  return (
    <div className="h-full flex flex-col space-y-6 p-6 overflow-y-auto bg-slate-950/20">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Historical Data Management</h2>
          <p className="text-slate-400">
            Organize, validate, and manage your downloaded market datasets.
          </p>
        </div>
        <div className="flex gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors text-sm font-medium border border-slate-700">
                <BarChart3 className="w-4 h-4" />
                Analyze Quality
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors text-sm font-medium shadow-lg shadow-indigo-500/20">
                <Download className="w-4 h-4" />
                Batch Export
            </button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        
        {/* Left Side: Stats and Info */}
        <div className="xl:col-span-1 space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-4 text-indigo-400">
                    <Database className="w-5 h-5" />
                    <h3 className="font-semibold text-slate-200">Storage Overview</h3>
                </div>
                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-500">Local Cache Usage</span>
                            <span className="text-slate-300">12% full</span>
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-indigo-500 rounded-full w-[12%]" />
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 pt-2">
                        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
                             <p className="text-xs text-slate-500 mb-1">Files</p>
                             <p className="text-lg font-mono font-bold text-slate-200">15</p>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
                             <p className="text-xs text-slate-500 mb-1">Integrity</p>
                             <div className="flex items-center gap-1 text-emerald-400">
                                <ShieldCheck className="w-3.5 h-3.5" />
                                <span className="text-sm font-medium">98%</span>
                             </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-3 text-amber-400">
                    <AlertCircle className="w-5 h-5" />
                    <h3 className="font-semibold text-slate-200 text-sm">Gap Detection</h3>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                    System detected missing data points in <span className="text-amber-300 font-mono">polymarket-price-data-...173.csv</span>. Recommend re-scraping the period of 2024-12-10.
                </p>
                <button className="mt-4 w-full py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-colors border border-amber-500/20">
                    Run Repair Tool
                </button>
            </div>
        </div>

        {/* Right Side: Data Catalog */}
        <div className="xl:col-span-3 bg-slate-900/50 border border-slate-800 rounded-2xl p-6 h-fit min-h-[500px]">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-6 bg-indigo-500 rounded-full" />
                    <h3 className="text-lg font-semibold text-slate-200">Dataset Catalog</h3>
                </div>
                <div className="flex gap-2 text-[10px] font-mono uppercase text-slate-500">
                    <span>Source: data/polymarket</span>
                </div>
            </div>
            <DataCatalog />
        </div>

      </div>
    </div>
  );
}
