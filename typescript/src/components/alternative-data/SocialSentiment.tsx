import { useState } from "react";
import { 
  TrendingUp, 
  TrendingDown,
  MessageSquare,
  Twitter,
  Hash
} from "lucide-react";
import clsx from "clsx";

interface MentionItem {
  id: string;
  symbol: string;
  mentions: number;
  change24h: number;
  sentiment: number; // -1 to 1
}

const MOCK_MENTIONS: MentionItem[] = [
  { id: "1", symbol: "BTC", mentions: 12450, change24h: 15.2, sentiment: 0.65 },
  { id: "2", symbol: "ETH", mentions: 8230, change24h: -5.1, sentiment: 0.42 },
  { id: "3", symbol: "SOL", mentions: 4120, change24h: 32.8, sentiment: 0.78 },
  { id: "4", symbol: "DOGE", mentions: 3890, change24h: 8.4, sentiment: 0.51 },
  { id: "5", symbol: "XRP", mentions: 2100, change24h: -12.3, sentiment: 0.28 },
];

export function SocialSentiment() {
  // Aggregate sentiment score (mock)
  const aggregateSentiment = 0.58; // 0 = bearish, 0.5 = neutral, 1 = bullish
  
  return (
    <div className="h-full p-6 overflow-y-auto">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
        
        {/* Aggregate Sentiment Gauge */}
        <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
            Aggregate Sentiment
          </h3>
          <div className="flex flex-col items-center">
            <SentimentGauge value={aggregateSentiment} />
            <div className="mt-4 text-center">
              <div className={clsx(
                "text-2xl font-bold",
                aggregateSentiment > 0.6 ? "text-emerald-400"
                  : aggregateSentiment < 0.4 ? "text-rose-400"
                  : "text-amber-400"
              )}>
                {aggregateSentiment > 0.6 ? "Bullish" 
                  : aggregateSentiment < 0.4 ? "Bearish" 
                  : "Neutral"}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Based on 24h social activity
              </div>
            </div>
          </div>
        </div>

        {/* Trend Chart Placeholder */}
        <div className="bg-slate-900 rounded-xl p-6 border border-slate-800 lg:col-span-2">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
            Sentiment Trend (24h)
          </h3>
          <div className="h-48 flex items-center justify-center text-slate-600 italic border border-dashed border-slate-800 rounded-lg">
            Rolling sentiment line chart
          </div>
        </div>

        {/* Top Mentions */}
        <div className="bg-slate-900 rounded-xl p-6 border border-slate-800 lg:col-span-3">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
            Top Mentions
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="px-4 py-2 text-[10px] font-bold text-slate-500 uppercase">Symbol</th>
                  <th className="px-4 py-2 text-[10px] font-bold text-slate-500 uppercase text-right">Mentions (24h)</th>
                  <th className="px-4 py-2 text-[10px] font-bold text-slate-500 uppercase text-right">Change</th>
                  <th className="px-4 py-2 text-[10px] font-bold text-slate-500 uppercase text-right">Sentiment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {MOCK_MENTIONS.map(m => (
                  <tr key={m.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Hash size={14} className="text-slate-500" />
                        <span className="font-bold text-slate-200">{m.symbol}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-sm text-slate-300">
                      {m.mentions.toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className={clsx(
                        "font-mono text-sm font-bold",
                        m.change24h >= 0 ? "text-emerald-400" : "text-rose-400"
                      )}>
                        {m.change24h >= 0 ? "+" : ""}{m.change24h}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <SentimentBar value={m.sentiment} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function SentimentGauge({ value }: { value: number }) {
  // Simple arc gauge visualization
  const rotation = -90 + (value * 180);
  
  return (
    <div className="relative w-32 h-16 overflow-hidden">
      <div className="absolute inset-x-0 bottom-0 h-32 w-32 rounded-full border-8 border-slate-800 mx-auto" 
           style={{ 
             background: `conic-gradient(from -90deg, 
               #f43f5e 0deg, 
               #f59e0b 90deg, 
               #10b981 180deg, 
               transparent 180deg)` 
           }}
      />
      <div className="absolute bottom-0 left-1/2 w-1 h-12 bg-white rounded-full origin-bottom shadow-lg"
           style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
      />
    </div>
  );
}

function SentimentBar({ value }: { value: number }) {
  const width = Math.abs((value - 0.5) * 2) * 100;
  const isBullish = value >= 0.5;
  
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden flex">
        <div 
          className={clsx(
            "h-full transition-all",
            isBullish ? "bg-emerald-500 ml-auto" : "bg-rose-500 mr-auto"
          )}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="text-[10px] font-mono text-slate-400 w-8">
        {(value * 100).toFixed(0)}
      </span>
    </div>
  );
}
