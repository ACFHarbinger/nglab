
import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Brain, Search, Filter, Loader2, Plus, RefreshCw, GitCompare, ChevronDown } from "lucide-react";
import { ModelCard, ModelMetadata } from "./ModelCard";
import { ModelComparisonView } from "./ModelComparisonView";
import { ModelDocumentationPanel } from "./ModelDocumentationPanel";
import clsx from "clsx";

/**
 * @module components/models/ModelRegistryTab
 * @description Main tab for browsing, filtering, comparing, and deploying ML models.
 */

// Architecture options for filtering
const ARCHITECTURE_OPTIONS = ["PPO", "SAC", "DQN", "Mamba", "LSTM", "Transformer", "VAE", "Other"];

export default function ModelRegistryTab() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  // Multi-select for comparison
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [showComparison, setShowComparison] = useState(false);

  // Filter state
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [archFilters, setArchFilters] = useState<Set<string>>(new Set());

  // Documentation panel state
  const [inspectModelName, setInspectModelName] = useState<string | null>(null);

  const refreshModels = async () => {
    setIsLoading(true);
    try {
      const [list, active] = await Promise.all([
        invoke<ModelMetadata[]>("list_trained_models"),
        invoke<string | null>("get_active_model")
      ]);
      setModels(list);
      setActiveModel(active);
    } catch (err) {
      console.error("Failed to list models:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeploy = async (modelName: string) => {
    try {
      await invoke("set_active_model", { modelName });
      setActiveModel(modelName);
    } catch (err) {
      console.error("Failed to set active model:", err);
    }
  };

  const toggleModelSelection = (modelName: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelName)) {
        next.delete(modelName);
      } else {
        next.add(modelName);
      }
      return next;
    });
  };

  const toggleArchFilter = (arch: string) => {
    setArchFilters((prev) => {
      const next = new Set(prev);
      if (next.has(arch)) {
        next.delete(arch);
      } else {
        next.add(arch);
      }
      return next;
    });
  };

  useEffect(() => {
    refreshModels();
  }, []);

  // Apply filters
  const filteredModels = models.filter((m) => {
    const matchesSearch =
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.architecture.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesArch = archFilters.size === 0 || archFilters.has(m.architecture);
    return matchesSearch && matchesArch;
  });

  return (
    <div className="h-full flex flex-col space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="text-indigo-500" />
            Model Registry
          </h2>
          <p className="text-slate-400">
            Manage, compare, and deploy your trained machine learning agents.
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={refreshModels}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            title="Refresh List"
          >
            <RefreshCw size={20} className={isLoading ? "animate-spin" : ""} />
          </button>
          <button
            onClick={() => {
              // Future: Navigate to Training Tab with preset
            }}
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 transition-colors shadow-lg shadow-indigo-500/20"
          >
            <Plus size={16} /> Train New Model
          </button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex gap-4 p-4 bg-slate-900/50 border border-slate-800 rounded-xl">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search models..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-300 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>

        {/* Filter Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowFilterMenu(!showFilterMenu)}
            className={clsx(
              "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border",
              archFilters.size > 0
                ? "bg-indigo-600 text-white border-indigo-500"
                : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
            )}
          >
            <Filter size={16} />
            Filter {archFilters.size > 0 && `(${archFilters.size})`}
            <ChevronDown size={14} />
          </button>
          {showFilterMenu && (
            <div className="absolute right-0 top-full mt-2 w-56 bg-slate-900 border border-slate-700 rounded-xl shadow-xl z-20 p-3">
              <p className="text-xs uppercase text-slate-500 font-semibold mb-2">Architecture</p>
              <div className="space-y-1">
                {ARCHITECTURE_OPTIONS.map((arch) => (
                  <label
                    key={arch}
                    className="flex items-center gap-2 text-sm text-slate-300 hover:bg-slate-800 p-2 rounded-lg cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={archFilters.has(arch)}
                      onChange={() => toggleArchFilter(arch)}
                      className="rounded border-slate-600 bg-slate-800 text-indigo-500 focus:ring-indigo-500"
                    />
                    {arch}
                  </label>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-slate-800 flex justify-between">
                <button
                  onClick={() => setArchFilters(new Set())}
                  className="text-xs text-slate-400 hover:text-white"
                >
                  Clear All
                </button>
                <button
                  onClick={() => setShowFilterMenu(false)}
                  className="text-xs text-indigo-400 hover:text-indigo-300"
                >
                  Done
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Compare Button */}
        <button
          onClick={() => setShowComparison(true)}
          disabled={selectedModels.size < 2}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border",
            selectedModels.size >= 2
              ? "bg-emerald-600 hover:bg-emerald-500 text-white border-emerald-500"
              : "bg-slate-800 text-slate-500 border-slate-700 cursor-not-allowed"
          )}
        >
          <GitCompare size={16} />
          Compare ({selectedModels.size})
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto min-h-[400px]">
        {isLoading ? (
          <div className="h-full flex items-center justify-center flex-col gap-3">
            <Loader2 size={32} className="animate-spin text-indigo-500" />
            <p className="text-slate-500 animate-pulse">Scanning neural archive...</p>
          </div>
        ) : filteredModels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-6">
            {filteredModels.map((model) => (
              <ModelCard
                key={model.name}
                model={model}
                isActive={activeModel === model.name}
                isSelected={selectedModels.has(model.name)}
                onSelect={() => toggleModelSelection(model.name)}
                onDeploy={() => handleDeploy(model.name)}
                onRun={() => setInspectModelName(model.name)}
              />
            ))}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center flex-col gap-4">
            <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800">
              <Brain size={32} className="text-slate-700" />
            </div>
            <div className="text-center">
              <h3 className="text-lg font-medium text-slate-300">No Models Found</h3>
              <p className="text-slate-500 max-w-xs mx-auto mt-1">
                Train your first model in the Training tab to see it appear here.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Comparison Modal */}
      {showComparison && (
        <ModelComparisonView
          modelNames={Array.from(selectedModels)}
          onClose={() => setShowComparison(false)}
        />
      )}

      {/* Documentation Panel */}
      {inspectModelName && (
        <ModelDocumentationPanel
          modelName={inspectModelName}
          onClose={() => setInspectModelName(null)}
        />
      )}
    </div>
  );
}
