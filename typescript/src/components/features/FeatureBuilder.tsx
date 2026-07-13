/**
 * @module components/features/FeatureBuilder
 * @description Visual feature builder for creating derived features with transformations.
 */

import { useState } from "react";
import { X, Plus, Calculator, Clock, Layers } from "lucide-react";
import clsx from "clsx";

interface TransformationStep {
  id: string;
  type: "lag" | "lead" | "rolling" | "diff" | "math";
  params: Record<string, number | string>;
}

interface FeatureBuilderProps {
  isOpen: boolean;
  onClose: () => void;
  onSave?: (feature: { name: string; steps: TransformationStep[] }) => void;
}

const AVAILABLE_OPERATIONS = [
  { id: "lag", label: "Lag", icon: Clock, description: "Shift values back in time", defaultParams: { periods: 1 } },
  { id: "lead", label: "Lead", icon: Clock, description: "Shift values forward in time", defaultParams: { periods: 1 } },
  { id: "rolling", label: "Rolling", icon: Layers, description: "Rolling window aggregation", defaultParams: { window: 10, agg: "mean" } },
  { id: "diff", label: "Difference", icon: Calculator, description: "Compute difference", defaultParams: { periods: 1 } },
  { id: "math", label: "Math Op", icon: Calculator, description: "Mathematical operation", defaultParams: { op: "log" } },
];

export function FeatureBuilder({ isOpen, onClose, onSave }: FeatureBuilderProps) {
  const [featureName, setFeatureName] = useState("");
  const [baseFeature, setBaseFeature] = useState("close");
  const [steps, setSteps] = useState<TransformationStep[]>([]);

  if (!isOpen) return null;

  const addStep = (type: TransformationStep["type"]) => {
    const op = AVAILABLE_OPERATIONS.find(o => o.id === type);
    const defaultParams = op?.defaultParams ?? {};
    setSteps([
      ...steps,
      {
        id: `${type}-${Date.now()}`,
        type,
        params: defaultParams as Record<string, number | string>,
      }
    ]);
  };

  const removeStep = (id: string) => {
    setSteps(steps.filter(s => s.id !== id));
  };

  const updateStepParam = (id: string, key: string, value: number | string) => {
    setSteps(steps.map(s => s.id === id ? { ...s, params: { ...s.params, [key]: value } } : s));
  };

  const handleSave = () => {
    if (featureName && onSave) {
      onSave({ name: featureName, steps });
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-xl font-semibold text-white">Feature Builder</h2>
            <p className="text-sm text-slate-400">Create derived features with transformations</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6 space-y-6">
          {/* Feature Name */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Feature Name</label>
              <input
                type="text"
                value={featureName}
                onChange={(e) => setFeatureName(e.target.value)}
                placeholder="e.g., momentum_10d"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Base Feature</label>
              <select
                value={baseFeature}
                onChange={(e) => setBaseFeature(e.target.value)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
              >
                <option value="close">Close Price</option>
                <option value="open">Open Price</option>
                <option value="high">High</option>
                <option value="low">Low</option>
                <option value="volume">Volume</option>
                <option value="returns">Returns</option>
              </select>
            </div>
          </div>

          {/* Transformation Pipeline */}
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-3">Transformation Pipeline</label>
            
            <div className="space-y-3">
              {/* Base Feature Chip */}
              <div className="flex items-center gap-2">
                <div className="px-4 py-2 bg-indigo-500/20 border border-indigo-500/50 rounded-lg text-sm font-medium text-indigo-300">
                  {baseFeature}
                </div>
                {steps.length > 0 && <span className="text-slate-500">→</span>}
              </div>

              {/* Steps */}
              {steps.map((step, i) => (
                <div key={step.id} className="flex items-center gap-2">
                  <div className="flex-1 bg-slate-800 border border-slate-700 rounded-lg p-3 flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      {step.type === "lag" && <Clock size={16} className="text-amber-400" />}
                      {step.type === "lead" && <Clock size={16} className="text-cyan-400" />}
                      {step.type === "rolling" && <Layers size={16} className="text-purple-400" />}
                      {step.type === "diff" && <Calculator size={16} className="text-emerald-400" />}
                      {step.type === "math" && <Calculator size={16} className="text-rose-400" />}
                      <span className="text-sm font-medium text-slate-300 capitalize">{step.type}</span>
                    </div>
                    
                    {/* Parameters */}
                    <div className="flex-1 flex gap-3">
                      {Object.entries(step.params).map(([key, val]) => (
                        <div key={key} className="flex items-center gap-2">
                          <span className="text-xs text-slate-500">{key}:</span>
                          {typeof val === "number" ? (
                            <input
                              type="number"
                              value={val}
                              onChange={(e) => updateStepParam(step.id, key, parseInt(e.target.value) || 0)}
                              className="w-16 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-sm text-slate-300"
                            />
                          ) : (
                            <select
                              value={val}
                              onChange={(e) => updateStepParam(step.id, key, e.target.value)}
                              className="px-2 py-1 bg-slate-900 border border-slate-700 rounded text-sm text-slate-300"
                            >
                              {key === "agg" && (
                                <>
                                  <option value="mean">mean</option>
                                  <option value="std">std</option>
                                  <option value="sum">sum</option>
                                  <option value="min">min</option>
                                  <option value="max">max</option>
                                </>
                              )}
                              {key === "op" && (
                                <>
                                  <option value="log">log</option>
                                  <option value="sqrt">sqrt</option>
                                  <option value="abs">abs</option>
                                  <option value="square">square</option>
                                </>
                              )}
                            </select>
                          )}
                        </div>
                      ))}
                    </div>

                    <button
                      onClick={() => removeStep(step.id)}
                      className="p-1 hover:bg-slate-700 rounded text-slate-500 hover:text-red-400"
                    >
                      <X size={14} />
                    </button>
                  </div>
                  {i < steps.length - 1 && <span className="text-slate-500">→</span>}
                </div>
              ))}
            </div>

            {/* Add Step Buttons */}
            <div className="mt-4 flex flex-wrap gap-2">
              {AVAILABLE_OPERATIONS.map((op) => (
                <button
                  key={op.id}
                  onClick={() => addStep(op.id as TransformationStep["type"])}
                  className={clsx(
                    "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors border",
                    "bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700"
                  )}
                >
                  <Plus size={14} />
                  {op.label}
                </button>
              ))}
            </div>
          </div>

          {/* Preview */}
          <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
            <div className="text-xs text-slate-500 mb-2">Generated Expression:</div>
            <code className="text-sm text-indigo-300 font-mono">
              {steps.length === 0 
                ? baseFeature 
                : steps.reduce((expr, step) => {
                    switch (step.type) {
                      case "lag": return `lag(${expr}, ${step.params.periods})`;
                      case "lead": return `lead(${expr}, ${step.params.periods})`;
                      case "rolling": return `rolling_${step.params.agg}(${expr}, ${step.params.window})`;
                      case "diff": return `diff(${expr}, ${step.params.periods})`;
                      case "math": return `${step.params.op}(${expr})`;
                      default: return expr;
                    }
                  }, baseFeature)
              }
            </code>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!featureName}
            className={clsx(
              "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
              featureName
                ? "bg-indigo-600 hover:bg-indigo-500 text-white"
                : "bg-slate-700 text-slate-500 cursor-not-allowed"
            )}
          >
            Save Feature
          </button>
        </div>
      </div>
    </div>
  );
}
