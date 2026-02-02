import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { X, FileText, Cpu, Clock, HardDrive, Layers, ArrowRight, Box, AlertCircle } from "lucide-react";
import clsx from "clsx";

/**
 * @module components/models/ModelDocumentationPanel
 * @description Slide-out panel showing auto-generated model documentation including
 * architecture, training configuration, and input/output specifications.
 */

export interface ModelDocumentation {
  name: string;
  architecture: string;
  size_bytes: number;
  modified_ts: number;
  // Training configuration (if available)
  training_config?: {
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
    optimizer?: string;
    loss_function?: string;
  };
  // Input/output specifications
  input_spec?: {
    shape: number[];
    dtype: string;
    features?: string[];
  };
  output_spec?: {
    shape: number[];
    dtype: string;
    classes?: string[];
  };
}

interface ModelDocumentationPanelProps {
  /** Name of the model to display documentation for */
  modelName: string;
  /** Callback to close the panel */
  onClose: () => void;
}

export function ModelDocumentationPanel({ modelName, onClose }: ModelDocumentationPanelProps) {
  const [doc, setDoc] = useState<ModelDocumentation | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDocumentation = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await invoke<ModelDocumentation>("get_model_documentation", { modelName });
        setDoc(result);
      } catch (err) {
        console.error("Failed to fetch model documentation:", err);
        setError("Documentation not available for this model.");
        // Fallback to basic info
        try {
          const models = await invoke<ModelDocumentation[]>("list_trained_models");
          const found = models.find((m) => m.name === modelName);
          if (found) {
            setDoc(found);
          }
        } catch {
          // Ignore fallback errors
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchDocumentation();
  }, [modelName]);

  const formatSize = (bytes: number) => (bytes / (1024 * 1024)).toFixed(2) + " MB";
  const formatDate = (ts: number) => new Date(ts * 1000).toLocaleString();

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      
      {/* Panel */}
      <div className="relative w-full max-w-lg bg-slate-900 border-l border-slate-700 shadow-2xl flex flex-col animate-slide-in-right">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <FileText className="text-indigo-400" size={24} />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white truncate max-w-xs" title={modelName}>
                {modelName}
              </h2>
              <p className="text-sm text-slate-400">Model Documentation</p>
            </div>
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
          {isLoading ? (
            <div className="flex items-center justify-center h-32">
              <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
            </div>
          ) : error && !doc ? (
            <div className="flex items-center gap-2 text-amber-400 p-4 bg-amber-500/10 rounded-lg border border-amber-500/20">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          ) : doc ? (
            <>
              {/* Basic Info */}
              <section>
                <h3 className="text-sm uppercase text-slate-500 font-semibold mb-3 flex items-center gap-2">
                  <Box size={14} /> Overview
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                    <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                      <Cpu size={12} /> Architecture
                    </div>
                    <div className="text-white font-semibold">{doc.architecture}</div>
                  </div>
                  <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                    <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                      <HardDrive size={12} /> Size
                    </div>
                    <div className="text-white font-semibold">{formatSize(doc.size_bytes)}</div>
                  </div>
                  <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700 col-span-2">
                    <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                      <Clock size={12} /> Last Modified
                    </div>
                    <div className="text-white font-semibold">{formatDate(doc.modified_ts)}</div>
                  </div>
                </div>
              </section>

              {/* Training Configuration */}
              {doc.training_config && (
                <section>
                  <h3 className="text-sm uppercase text-slate-500 font-semibold mb-3 flex items-center gap-2">
                    <Layers size={14} /> Training Configuration
                  </h3>
                  <div className="bg-slate-800/50 rounded-lg border border-slate-700 divide-y divide-slate-700">
                    {Object.entries(doc.training_config).map(([key, value]) => (
                      <div key={key} className="flex justify-between items-center p-3">
                        <span className="text-slate-400 text-sm capitalize">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span className="text-white font-mono text-sm">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Input/Output Specs */}
              {(doc.input_spec || doc.output_spec) && (
                <section>
                  <h3 className="text-sm uppercase text-slate-500 font-semibold mb-3 flex items-center gap-2">
                    <ArrowRight size={14} /> Input / Output Specification
                  </h3>
                  <div className="space-y-3">
                    {doc.input_spec && (
                      <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                        <div className="text-xs text-slate-400 uppercase mb-2">Input</div>
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-slate-300">Shape:</span>
                          <code className="bg-slate-700 px-2 py-0.5 rounded text-indigo-300 font-mono">
                            [{doc.input_spec.shape.join(", ")}]
                          </code>
                          <span className="text-slate-500">({doc.input_spec.dtype})</span>
                        </div>
                        {doc.input_spec.features && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {doc.input_spec.features.slice(0, 10).map((f) => (
                              <span key={f} className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                                {f}
                              </span>
                            ))}
                            {doc.input_spec.features.length > 10 && (
                              <span className="px-2 py-0.5 text-xs text-slate-500">
                                +{doc.input_spec.features.length - 10} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                    {doc.output_spec && (
                      <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                        <div className="text-xs text-slate-400 uppercase mb-2">Output</div>
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-slate-300">Shape:</span>
                          <code className="bg-slate-700 px-2 py-0.5 rounded text-emerald-300 font-mono">
                            [{doc.output_spec.shape.join(", ")}]
                          </code>
                          <span className="text-slate-500">({doc.output_spec.dtype})</span>
                        </div>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Placeholder for performance metrics */}
              <section className="bg-slate-800/30 p-4 rounded-lg border border-dashed border-slate-700 text-center">
                <p className="text-slate-500 text-sm">
                  Performance metrics will appear here after model evaluation.
                </p>
              </section>
            </>
          ) : null}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
