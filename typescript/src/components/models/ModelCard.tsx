
import { Brain, Calendar, Server, CheckCircle2, Square, CheckSquare, Download } from "lucide-react";
import clsx from "clsx";

/**
 * @module components/models/ModelCard
 * @description Display card for a trained ML model with deployment and selection controls.
 */

export interface ModelMetadata {
  name: string;
  filename: string;
  size_bytes: number;
  modified_ts: number;
  architecture: string;
}

interface ModelCardProps {
  model: ModelMetadata;
  isActive?: boolean;
  /** Whether the card is selected for comparison */
  isSelected?: boolean;
  /** Callback when the selection checkbox is toggled */
  onSelect?: () => void;
  onDeploy?: () => void;
  onRun?: () => void;
  /** Callback to export model to ONNX format */
  onExport?: () => void;
}

export function ModelCard({ model, isActive, isSelected, onSelect, onDeploy, onRun, onExport }: ModelCardProps) {
  const createdDate = new Date(model.modified_ts * 1000).toLocaleDateString();
  const sizeMb = (model.size_bytes / (1024 * 1024)).toFixed(2);
  
  return (
    <div className={clsx(
      "group relative bg-slate-900/50 border rounded-xl p-5 transition-all flex flex-col gap-4",
      isSelected
        ? "border-emerald-500 shadow-lg shadow-emerald-500/20 bg-emerald-500/5"
        : isActive 
          ? "border-indigo-500 shadow-lg shadow-indigo-500/20 bg-indigo-500/5" 
          : "border-slate-800 hover:border-indigo-500/50 hover:bg-slate-900 hover:shadow-lg hover:shadow-indigo-500/10"
    )}>
      {/* Selection Checkbox */}
      {onSelect && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          className="absolute top-2 left-2 p-1 text-slate-400 hover:text-emerald-400 transition-colors"
          title={isSelected ? "Deselect for comparison" : "Select for comparison"}
        >
          {isSelected ? (
            <CheckSquare size={18} className="text-emerald-400" />
          ) : (
            <Square size={18} />
          )}
        </button>
      )}

      {isActive && (
        <div className="absolute top-2 right-2">
            <span className="flex items-center gap-1 bg-indigo-500 text-white text-[10px] uppercase font-bold px-2 py-0.5 rounded-full shadow-lg shadow-indigo-500/20">
                <CheckCircle2 size={10} /> Active
            </span>
        </div>
      )}

      <div className="flex justify-between items-start">
        <div className={clsx(
            "p-2.5 rounded-lg transition-colors",
            isActive ? "bg-indigo-500/20" : "bg-indigo-500/10 group-hover:bg-indigo-500/20"
        )}>
          <Brain size={24} className={isActive ? "text-indigo-300" : "text-indigo-400"} />
        </div>
        {!isActive && (
            <div className="flex gap-2">
                <span className="px-2 py-1 rounded text-[10px] uppercase font-bold tracking-wider bg-slate-800 text-slate-400">
                    {model.architecture}
                </span>
            </div>
        )}
      </div>

      <div>
        <h3 className={clsx(
            "font-semibold text-lg transition-colors truncate",
            isActive ? "text-indigo-200" : "text-slate-200 group-hover:text-white"
        )} title={model.name}>
          {model.name}
        </h3>
        <p className="text-xs text-slate-500 mt-1 truncate">
           {model.filename}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 py-3 border-t border-slate-800/50 border-b">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
            <Calendar size={10} /> Modified
          </span>
          <span className="text-xs font-mono text-slate-300">{createdDate}</span>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
            <Server size={10} /> Size
          </span>
          <span className="text-xs font-mono text-indigo-400">{sizeMb} MB</span>
        </div>
      </div>

      <div className="flex gap-2 mt-auto">
        {isActive ? (
             <div className="flex-1 bg-indigo-500/20 text-indigo-300 py-2 rounded-lg text-xs font-semibold uppercase tracking-wide flex items-center justify-center gap-2 cursor-default border border-indigo-500/20">
                <CheckCircle2 size={14} /> Deployed
             </div>
        ) : (
            <button 
                onClick={onDeploy}
                className="flex-1 bg-slate-800 hover:bg-indigo-600 text-slate-300 hover:text-white py-2 rounded-lg text-xs font-semibold uppercase tracking-wide transition-all flex items-center justify-center gap-2 border border-slate-700 hover:border-indigo-500"
            >
                <Server size={14} /> Deploy
            </button>
        )}
        
        <button 
            onClick={onExport}
            className="p-2 bg-slate-800 hover:bg-emerald-600 text-slate-400 hover:text-white rounded-lg transition-colors border border-slate-700"
            title="Export to ONNX"
        >
            <Download size={16} />
        </button>
        
        <button 
            onClick={onRun}
            className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white rounded-lg transition-colors border border-slate-700"
            title="Inspect Model"
        >
            <Brain size={16} />
        </button>
      </div>
    </div>
  );
}
