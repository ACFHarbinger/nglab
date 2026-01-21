/**
 * @module components/NewsTab
 * @description Aggregated news and social media feed with sentiment analysis indicators.
 */
import { useState } from "react";
import {
  Newspaper,
  Twitter,
  Globe,
  MessageCircle,
  Filter,
  Search,
  ExternalLink,
  ThumbsUp,
  MessageSquare,
  Repeat2,
} from "lucide-react";
import clsx from "clsx";

/**
 * Supported news and social media sources configuration.
 * Categorized by type (News, Social, Market Data).
 */
const SOURCES = {
  "News & Analysis": [
    {
      id: "coindesk",
      name: "CoinDesk",
      icon: Newspaper,
      color: "text-indigo-400",
    },
    {
      id: "cointelegraph",
      name: "CoinTelegraph",
      icon: Newspaper,
      color: "text-yellow-400",
    },
    {
      id: "theblock",
      name: "The Block",
      icon: Newspaper,
      color: "text-purple-400",
    },
  ],
  "Social Media": [
    {
      id: "polynomial_x",
      name: "@Polymarket",
      icon: Twitter,
      color: "text-sky-400",
    },
    {
      id: "vitalik_x",
      name: "@VitalikButerin",
      icon: Twitter,
      color: "text-sky-400",
    },
    {
      id: "wsb_reddit",
      name: "r/WallStreetBets",
      icon: MessageCircle,
      color: "text-orange-500",
    },
  ],
  "Market Data": [
    {
      id: "polymarket_whales",
      name: "Whale Alerts",
      icon: Globe,
      color: "text-emerald-400",
    },
    {
      id: "rekt_news",
      name: "Liquidation Feed",
      icon: Globe,
      color: "text-rose-400",
    },
  ],
  // ... other categories
};

// Mock Feed Items
const MOCK_FEED = [
  {
    id: 1,
    sourceId: "coindesk",
    sourceName: "CoinDesk",
    type: "news",
    title: "Polymarket Volume Hits Record High Ahead of Election",
    snippet:
      "Prediction markets are seeing unprecedented activity as traders hedge their bets on the upcoming US election results...",
    time: "2 mins ago",
    sentiment: "bullish",
    url: "#",
  },
  {
    id: 2,
    sourceId: "vitalik_x",
    sourceName: "@VitalikButerin",
    type: "social",
    content:
      "Prediction markets are one of the most underrated applications of crypto. They provide a source of truth that is hard to manipulate.",
    time: "15 mins ago",
    stats: { likes: "12.5k", retweets: "3.2k", replies: "450" },
    url: "#",
  },
  {
    id: 3,
    sourceId: "polymarket_whales",
    sourceName: "Whale Alerts",
    type: "alert",
    title: "Large Position Opened",
    snippet: 'Whale "0x4a...9f" bought $500k NO on "Bitcoin > $100k by EOY"',
    time: "45 mins ago",
    sentiment: "bearish",
    url: "#",
  },
  {
    id: 4,
    sourceId: "theblock",
    sourceName: "The Block",
    type: "news",
    title: "SEC Approves New Bitcoin ETF Options",
    snippet:
      "Regulatory clarity continues to improve as the SEC gives the green light for options trading on spot Bitcoin ETFs.",
    time: "1 hour ago",
    sentiment: "bullish",
    url: "#",
  },
  {
    id: 5,
    sourceId: "wsb_reddit",
    sourceName: "r/WallStreetBets",
    type: "social",
    title: "YOLOing my life savings into Trump winning",
    snippet:
      "Just put 50k on Trump to win. The odds are too good to pass up. See you on the moon or at Wendy's.",
    time: "2 hours ago",
    stats: { upvotes: "4.2k", comments: "890" },
    url: "#",
  },
];

/**
 * Main component for viewing market-related news, social media posts, and alerts.
 * Supports filtering by source and switching between live feed and sentiment analysis views.
 */
export default function NewsTab() {
  const [selectedSources, setSelectedSources] = useState<Set<string>>(
    new Set([
      "coindesk",
      "cointelegraph",
      "polynomial_x",
      "vitalik_x",
      "polymarket_whales",
    ]),
  );
  const [activeTab, setActiveTab] = useState<"feed" | "sentiment">("feed");
  const [searchTerm, setSearchTerm] = useState("");

  const toggleSource = (id: string) => {
    const newSelected = new Set(selectedSources);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedSources(newSelected);
  };

  const filteredFeed = MOCK_FEED.filter((item) =>
    selectedSources.has(item.sourceId),
  );

  return (
    <div className="h-full flex bg-slate-950 text-slate-200 overflow-hidden">
      {/* Sidebar - Sources */}
      <div className="w-72 border-r border-slate-800 flex flex-col bg-slate-900/30">
        <div className="p-4 border-b border-slate-800">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Filter size={16} /> Sources
          </h2>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-6">
          {Object.entries(SOURCES).map(([category, sources]) => (
            <div key={category}>
              <h3 className="px-3 text-xs font-bold text-slate-500 uppercase mb-2">
                {category}
              </h3>
              <div className="space-y-0.5">
                {sources.map((source) => {
                  const Icon = source.icon;
                  const isSelected = selectedSources.has(source.id);
                  return (
                    <button
                      key={source.id}
                      onClick={() => toggleSource(source.id)}
                      className={clsx(
                        "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all",
                        isSelected
                          ? "bg-indigo-500/10 text-slate-100 hover:bg-indigo-500/20"
                          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
                      )}
                    >
                      <div
                        className={clsx(
                          "w-2 h-2 rounded-full",
                          isSelected ? "bg-indigo-500" : "bg-slate-700",
                        )}
                      />
                      <Icon size={16} className={source.color} />
                      <span className="truncate">{source.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content - Feed */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/20 backdrop-blur-sm">
          <div className="flex items-center gap-4">
            <div className="flex p-1 bg-slate-800 rounded-lg">
              <button
                onClick={() => setActiveTab("feed")}
                className={clsx(
                  "px-4 py-1.5 rounded-md text-sm font-medium transition-all",
                  activeTab === "feed"
                    ? "bg-indigo-600 text-white shadow-lg"
                    : "text-slate-400 hover:text-slate-200",
                )}
              >
                Live Feed
              </button>
              <button
                onClick={() => setActiveTab("sentiment")}
                className={clsx(
                  "px-4 py-1.5 rounded-md text-sm font-medium transition-all",
                  activeTab === "sentiment"
                    ? "bg-indigo-600 text-white shadow-lg"
                    : "text-slate-400 hover:text-slate-200",
                )}
              >
                Sentiment Analysis
              </button>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search news..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-64 bg-slate-800 border border-slate-700 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        </header>

        {/* Feed Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 bg-slate-950 relative">
          {/* Background Pattern */}
          <div
            className="absolute inset-0 opacity-[0.02] pointer-events-none"
            style={{
              backgroundImage: "radial-gradient(#4f46e5 1px, transparent 1px)",
              backgroundSize: "32px 32px",
            }}
          />

          <div className="max-w-3xl mx-auto space-y-4 relative z-10">
            {filteredFeed.length === 0 ? (
              <div className="text-center py-20 text-slate-500">
                <p>No items match your selected sources.</p>
              </div>
            ) : (
              filteredFeed.map((item) => (
                <div
                  key={item.id}
                  className="bg-slate-900/50 border border-slate-800 hover:border-slate-700 rounded-xl p-5 transition-all hover:shadow-lg hover:shadow-indigo-500/5 group"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                        {item.type === "news" && <Newspaper size={12} />}
                        {item.type === "social" && <Twitter size={12} />}
                        {item.type === "alert" && <Globe size={12} />}
                        {item.sourceName}
                      </span>
                      <span className="text-slate-600 text-[10px]">•</span>
                      <span className="text-xs text-slate-500">
                        {item.time}
                      </span>
                    </div>
                    <div className="flex gap-2">
                      {item.sentiment && (
                        <span
                          className={clsx(
                            "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
                            item.sentiment === "bullish"
                              ? "bg-green-500/10 text-green-400"
                              : "bg-red-500/10 text-red-400",
                          )}
                        >
                          {item.sentiment}
                        </span>
                      )}
                      <button className="text-slate-500 hover:text-indigo-400 opacity-0 group-hover:opacity-100 transition-opacity">
                        <ExternalLink size={14} />
                      </button>
                    </div>
                  </div>

                  {item.type === "social" ? (
                    <>
                      {item.title && (
                        <h3 className="text-lg font-semibold text-slate-200 mb-1">
                          {item.title}
                        </h3>
                      )}
                      <p className="text-slate-300 leading-relaxed text-sm mb-3">
                        "{item.content || item.snippet}"
                      </p>
                      <div className="flex gap-4 text-xs text-slate-500">
                        <span className="flex items-center gap-1 hover:text-rose-400 cursor-pointer transition-colors">
                          <ThumbsUp size={12} />{" "}
                          {item.stats?.likes || item.stats?.upvotes}
                        </span>
                        <span className="flex items-center gap-1 hover:text-green-400 cursor-pointer transition-colors">
                          <Repeat2 size={12} /> {item.stats?.retweets || "0"}
                        </span>
                        <span className="flex items-center gap-1 hover:text-sky-400 cursor-pointer transition-colors">
                          <MessageSquare size={12} />{" "}
                          {item.stats?.comments || item.stats?.replies}
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <h3 className="text-lg font-semibold text-indigo-100 mb-1 group-hover:text-indigo-300 transition-colors cursor-pointer">
                        {item.title}
                      </h3>
                      <p className="text-slate-400 text-sm leading-relaxed">
                        {item.snippet}
                      </p>
                    </>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
