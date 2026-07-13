import React from "react";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Brain, Zap, Layers } from "lucide-react";

interface FeatureImportance {
  feature: string;
  score: number;
}

interface AttentionMap {
  layer: number;
  head: number;
  matrix: number[][]; // [seq_len, seq_len]
}

export interface ExplanationData {
  importances: FeatureImportance[];
  attention?: AttentionMap;
}

interface ExplanationViewProps {
  data: ExplanationData | null;
  isLoading?: boolean;
}

export default function ExplanationView({ data, isLoading }: ExplanationViewProps) {
  if (isLoading) {
    return (
        <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 h-64 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2 text-slate-500">
                <Brain className="animate-pulse" size={32} />
                <span className="text-sm font-medium">Generating Explanations...</span>
            </div>
        </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      
      {/* Feature Importance Section */}
      <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Zap size={20} className="text-purple-500" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-200">Feature Contribution</h3>
            <p className="text-xs text-slate-400">Relative impact of input features on the latest prediction</p>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data.importances}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#334155" />
              <XAxis type="number" stroke="#94a3b8" fontSize={10} />
              <YAxis 
                type="category" 
                dataKey="feature" 
                stroke="#94a3b8" 
                fontSize={11} 
                width={100}
                tickFormatter={(value) => value.length > 15 ? value.substring(0, 15) + '...' : value}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {data.importances.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.score >= 0 ? '#10b981' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Attention Map Section (Conditional) */}
      {data.attention && (
        <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
             <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-indigo-500/10 rounded-lg">
                    <Layers size={20} className="text-indigo-500" />
                </div>
                <div>
                    <h3 className="font-semibold text-slate-200">Attention Weights</h3>
                    <p className="text-xs text-slate-400">Layer {data.attention.layer + 1}, Head {data.attention.head + 1}</p>
                </div>
            </div>
            
            <div className="grid place-items-center">
                <AttentionHeatmap matrix={data.attention.matrix} />
            </div>
        </div>
      )}
    </div>
  );
}

function AttentionHeatmap({ matrix }: { matrix: number[][] }) {
    if (!matrix || matrix.length === 0) return null;

    const size = matrix.length;
    // Simple canvas or grid rendering could work. For now, a CSS grid.
    // Limiting to max 20x20 for performance in CSS Grid
    const displaySize = Math.min(size, 20);
    const cellSize = 300 / displaySize;

    return (
        <div 
            className="grid gap-[1px] bg-slate-800 border border-slate-800"
            style={{ 
                gridTemplateColumns: `repeat(${displaySize}, 1fr)`,
                width: '300px',
                height: '300px'
            }}
        >
            {matrix.slice(0, displaySize).map((row, i) => 
                row.slice(0, displaySize).map((val, j) => (
                    <div 
                        key={`${i}-${j}`}
                        className="w-full h-full hover:border border-white/50 transition-colors"
                        style={{ 
                            backgroundColor: `rgba(99, 102, 241, ${Math.min(val * 5, 1)})`, // Scaled for visibility
                        }}
                        title={`Src: ${i}, Tgt: ${j}, Weight: ${val.toFixed(3)}`}
                    />
                ))
            )}
        </div>
    );
}
