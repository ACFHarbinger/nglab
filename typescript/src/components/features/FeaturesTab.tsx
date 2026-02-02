/**
 * @module components/features/FeaturesTab
 * @description Feature Engineering tab with catalog, builder, and validation tools.
 */

import { useState } from "react";
import { FeatureCatalog } from "./FeatureCatalog";
import { FeatureBuilder } from "./FeatureBuilder";
import { FeatureValidation, MOCK_VALIDATION_RESULT } from "./FeatureValidation";
import { Beaker, Database, FileCode2, Activity, CheckSquare } from "lucide-react";
import clsx from "clsx";

type ViewMode = "catalog" | "validation";

export default function FeaturesTab() {
  const [showBuilder, setShowBuilder] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("catalog");

  return (
    <div className="h-full flex flex-col space-y-6 p-6 overflow-y-auto">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Feature Engineering</h2>
          <p className="text-slate-400">
            Manage, visualize, and construct features for model training.
          </p>
        </div>
        <div className="flex gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors text-sm font-medium">
                <FileCode2 className="w-4 h-4" />
                Export Definition
            </button>
            <button 
              onClick={() => setShowBuilder(true)}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors text-sm font-medium shadow-lg shadow-indigo-500/20"
            >
                <Beaker className="w-4 h-4" />
                Build Feature
            </button>
        </div>
      </div>

      {/* View Mode Tabs */}
      <div className="flex gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setViewMode("catalog")}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
            viewMode === "catalog"
              ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/50"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          )}
        >
          <Database size={16} />
          Feature Catalog
        </button>
        <button
          onClick={() => setViewMode("validation")}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
            viewMode === "validation"
              ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/50"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          )}
        >
          <CheckSquare size={16} />
          Validation
        </button>
      </div>

      {/* Main Content */}
      {viewMode === "catalog" && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-6">
              <Database className="w-5 h-5 text-indigo-400" />
              <h3 className="text-lg font-semibold text-slate-200">Feature Catalog</h3>
          </div>
          <FeatureCatalog />
        </div>
      )}

      {viewMode === "validation" && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-6">
              <CheckSquare className="w-5 h-5 text-indigo-400" />
              <h3 className="text-lg font-semibold text-slate-200">Feature Validation</h3>
              <span className="ml-auto text-xs text-slate-500">Showing: imbalance</span>
          </div>
          <FeatureValidation result={MOCK_VALIDATION_RESULT} />
        </div>
      )}
      
      {/* Bottom Grid - Importance Ranking */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 col-span-2">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Pipeline Visualization</h3>
            <div className="flex items-center justify-center h-48 border-2 border-dashed border-slate-800 rounded-xl text-slate-500">
                <span className="text-sm">Click "Build Feature" to create transformation pipelines</span>
            </div>
        </div>
        
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
             <div className="flex items-center gap-2 mb-4">
               <Activity className="w-5 h-5 text-purple-400" />
               <h3 className="text-lg font-semibold text-slate-200">Importance Ranking</h3>
             </div>
             <div className="space-y-3">
                 {[
                     { name: "imbalance", score: 0.85 },
                     { name: "log_ret", score: 0.62 },
                     { name: "volatility", score: 0.45 },
                     { name: "spread", score: 0.38 },
                     { name: "rsi", score: 0.28 },
                 ].map((feat, i) => (
                     <div key={feat.name} className="flex items-center gap-3">
                         <span className="text-slate-500 font-mono text-xs w-6">#{i+1}</span>
                         <div className="flex-1">
                             <div className="flex justify-between text-sm mb-1">
                                 <span className="text-slate-300">{feat.name}</span>
                                 <span className="text-slate-500">{(feat.score * 100).toFixed(0)}%</span>
                             </div>
                             <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                 <div 
                                    className={clsx(
                                      "h-full rounded-full",
                                      i === 0 ? "bg-indigo-500" : "bg-indigo-500/60"
                                    )}
                                    style={{ width: `${feat.score * 100}%` }}
                                 />
                             </div>
                         </div>
                     </div>
                 ))}
             </div>
        </div>
      </div>

      {/* Feature Builder Modal */}
      <FeatureBuilder
        isOpen={showBuilder}
        onClose={() => setShowBuilder(false)}
        onSave={(feature) => {
          console.log("Saved feature:", feature);
          setShowBuilder(false);
        }}
      />
    </div>
  );
}
