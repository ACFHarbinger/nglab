import { useState } from "react";
import { 
  ExternalLink, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  Filter,
  RefreshCw
} from "lucide-react";
import clsx from "clsx";

interface NewsItem {
  id: string;
  title: string;
  source: string;
  timestamp: string;
  sentiment: "bullish" | "bearish" | "neutral";
  keywords: string[];
  url: string;
}

const MOCK_NEWS: NewsItem[] = [
  {
    id: "1",
    title: "Bitcoin breaks $65K resistance as institutional demand surges",
    source: "CoinDesk",
    timestamp: "5m ago",
    sentiment: "bullish",
    keywords: ["BTC", "Institutional", "Resistance"],
    url: "#"
  },
  {
    id: "2",
    title: "SEC delays decision on spot ETH ETF applications",
    source: "The Block",
    timestamp: "22m ago",
    sentiment: "bearish",
    keywords: ["ETH", "SEC", "ETF"],
    url: "#"
  },
  {
    id: "3",
    title: "Fed minutes reveal hawkish stance on rate cuts",
    source: "Bloomberg",
    timestamp: "1h ago",
    sentiment: "bearish",
    keywords: ["Fed", "Rates", "Macro"],
    url: "#"
  },
  {
    id: "4",
    title: "Polymarket sees record volume on 2024 election markets",
    source: "Decrypt",
    timestamp: "2h ago",
    sentiment: "neutral",
    keywords: ["Polymarket", "Elections", "Volume"],
    url: "#"
  },
  {
    id: "5",
    title: "Major DeFi protocol announces Layer 2 migration",
    source: "DeFi Pulse",
    timestamp: "3h ago",
    sentiment: "bullish",
    keywords: ["DeFi", "L2", "Migration"],
    url: "#"
  }
];

export function NewsFeed() {
  const [filter, setFilter] = useState<"all" | "bullish" | "bearish" | "neutral">("all");
  
  const filteredNews = filter === "all" 
    ? MOCK_NEWS 
    : MOCK_NEWS.filter(n => n.sentiment === filter);

  return (
    <div className="h-full flex flex-col p-6">
      {/* Filters */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-slate-500" />
          <span className="text-xs text-slate-500 font-medium">Sentiment:</span>
          {(["all", "bullish", "bearish", "neutral"] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={clsx(
                "px-2 py-1 rounded text-[10px] font-bold uppercase transition-colors",
                filter === f
                  ? f === "bullish" ? "bg-emerald-500/20 text-emerald-400"
                    : f === "bearish" ? "bg-rose-500/20 text-rose-400"
                    : f === "neutral" ? "bg-slate-500/20 text-slate-400"
                    : "bg-indigo-500/20 text-indigo-400"
                  : "text-slate-600 hover:text-slate-400"
              )}
            >
              {f}
            </button>
          ))}
        </div>
        <button className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors">
          <RefreshCw size={12} />
          Refresh
        </button>
      </div>

      {/* News List */}
      <div className="flex-1 overflow-y-auto space-y-3">
        {filteredNews.map(item => (
          <NewsCard key={item.id} item={item} />
        ))}
      </div>
    </div>
  );
}

function NewsCard({ item }: { item: NewsItem }) {
  const SentimentIcon = item.sentiment === "bullish" ? TrendingUp 
    : item.sentiment === "bearish" ? TrendingDown 
    : Minus;
  
  const sentimentColor = item.sentiment === "bullish" ? "text-emerald-400"
    : item.sentiment === "bearish" ? "text-rose-400"
    : "text-slate-400";

  return (
    <div className="bg-slate-900 rounded-xl p-4 border border-slate-800 hover:border-slate-700 transition-all group">
      <div className="flex items-start gap-3">
        <div className={clsx("mt-1 p-1.5 rounded-lg bg-slate-800", sentimentColor)}>
          <SentimentIcon size={14} />
        </div>
        <div className="flex-1">
          <a 
            href={item.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="font-medium text-slate-200 hover:text-white transition-colors group-hover:underline"
          >
            {item.title}
            <ExternalLink size={12} className="inline ml-2 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-[10px] font-bold text-slate-500">{item.source}</span>
            <span className="text-slate-700">•</span>
            <span className="text-[10px] text-slate-600">{item.timestamp}</span>
          </div>
          <div className="flex flex-wrap gap-1 mt-2">
            {item.keywords.map(kw => (
              <span 
                key={kw} 
                className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] font-medium text-slate-400"
              >
                {kw}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
