/**
 * @module components/dashboard/GlobalActivityWidget
 * @description Real-time feed of market trades and OSINT/Social media intelligence.
 */
import { useState } from "react";
/**
 * @module components/dashboard/MarketStatsWidget
 * @description High-level overview of global market volume, active traders, and peak activity times.
 */
import { Activity, Users, Clock, Calendar, Zap, Eye } from "lucide-react";
import { ExternalLink, Twitter, Info, Radio } from "lucide-react";
import clsx from "clsx";

const ACTIVITY_TABS = ["All", "Insider", "Top Traders", "Tracked", "Friends"];
const INTEL_TABS = ["OSINT", "Twitter"];

// Market activity list data
const MARKET_ACTIVITY = [
  {
    id: "1",
    bid: "M",
    market: "XRP Up or Down - January 17, 5:45AM-6:...",
    price: "10¢",
    amount: "$1.00",
    color: "text-emerald-400",
  },
  {
    id: "2",
    bid: "N",
    market: "Ethereum Up or Down - January 17, 5:45...",
    price: "65¢",
    amount: "$3.25",
    color: "text-rose-400",
  },
  {
    id: "3",
    bid: "M",
    market: "Will the price of XRP be between $2.10...",
    price: "99¢",
    amount: "$40.00",
    color: "text-emerald-400",
  },
  {
    id: "4",
    bid: "N",
    market: "Bitcoin Up or Down - January 17, 6:00A...",
    price: "45¢",
    amount: "$2.25",
    color: "text-rose-400",
  },
  {
    id: "5",
    bid: "Y",
    market: "Will the price of Bitcoin be above $96...",
    price: "1¢",
    amount: "$0.00",
    color: "text-yellow-400",
  },
  {
    id: "6",
    bid: "Y",
    market: "Will the price of Bitcoin be above $96...",
    price: "1¢",
    amount: "$0.00",
    color: "text-yellow-400",
  },
  {
    id: "7",
    bid: "Y",
    market: "Will the price of Bitcoin be above $96...",
    price: "1¢",
    amount: "$0.00",
    color: "text-yellow-400",
  },
  {
    id: "8",
    bid: "Y",
    market: "Will the price of Bitcoin be above $96...",
    price: "1¢",
    amount: "$0.00",
    color: "text-yellow-400",
  },
  {
    id: "9",
    bid: "Y",
    market: "Will the price of Bitcoin be above $96...",
    price: "1¢",
    amount: "$0.00",
    color: "text-yellow-400",
  },
];

const NEWS_ITEMS = [
  {
    id: 1,
    title:
      "PLA Eastern Theatre Command Tracks US Naval Assets - Regional Tensions Rise",
    description:
      "The People's Liberation Army Eastern Theatre Command has deployed naval and air forces to monitor U.S. vessel movements in the region, indicating...",
    source: "Twitter",
    sentiment: "BULLISH",
    sentimentConf: "152 conf",
    time: "+2s ago",
    relatedMarkets: 3,
    via: "@FirstSquawk (100.6k followers)",
  },
  {
    id: 2,
    title:
      "US Naval Transit Through Taiwan Strait Heightens Regional Military Tensions",
    description:
      "Chinese military confirmed two US vessels - USS John Finn (destroyer) and USNS Mary Sears (survey ship) - conducted a freedom of navigation...",
    source: "Twitter",
    sentiment: "BULLISH",
    sentimentConf: "152 conf",
    time: "+5m ago",
    relatedMarkets: 2,
    via: "@FirstSquawk (429.0k followers)",
  },
];

export function GlobalActivityWidget() {
  const [activeTab, setActiveTab] = useState("All");
  const [activeIntelTab, setActiveIntelTab] = useState("OSINT");

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800 w-full">
      {/* Activity Section */}
      <div className="flex flex-col">
        {/* Activity Header / Tabs */}
        <div className="flex items-center justify-between px-2 py-1 border-b border-slate-800 bg-slate-950">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Activity
          </span>
          <span className="flex items-center gap-1.5 bg-green-500/20 text-green-400 text-[9px] font-bold px-1.5 py-0.5 rounded-full border border-green-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            LIVE
          </span>
        </div>
        <div className="flex border-b border-slate-800 bg-slate-950">
          {ACTIVITY_TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                "flex-1 py-2 text-[9px] font-bold uppercase tracking-wider transition-all",
                activeTab === tab
                  ? "text-indigo-400 border-b-2 border-indigo-500 bg-indigo-500/10"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-900",
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Market Activity List */}
        <div className="max-h-[250px] overflow-y-auto custom-scrollbar">
          {/* Column Headers */}
          <div className="flex items-center px-2 py-1.5 text-[8px] text-slate-600 font-bold uppercase border-b border-slate-800/50 bg-slate-950/50 sticky top-0">
            <span className="w-6">Bid</span>
            <span className="flex-1">Market</span>
            <span className="w-12 text-right">Price</span>
            <span className="w-14 text-right">$</span>
          </div>
          {MARKET_ACTIVITY.map((item, i) => (
            <div
              key={i}
              className="flex items-center px-2 py-1.5 border-b border-slate-800/30 hover:bg-slate-800/40 transition-colors cursor-pointer group"
            >
              <span className={clsx("w-6 text-[10px] font-bold", item.color)}>
                {item.bid}
              </span>
              <span className="flex-1 text-[10px] text-slate-400 group-hover:text-slate-200 truncate pr-2">
                {item.market}
              </span>
              <span className="w-12 text-[10px] text-slate-300 text-right font-mono">
                {item.price}
              </span>
              <span className="w-14 text-[10px] text-slate-500 text-right font-mono">
                {item.amount}
              </span>
            </div>
          ))}
        </div>
        <div className="px-2 py-1.5 text-[9px] text-slate-600 border-b border-slate-800 bg-slate-950/50">
          200 trades
        </div>
      </div>

      {/* Intel Section */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Intel Tabs */}
        <div className="flex items-center border-b border-slate-800 bg-slate-950">
          {INTEL_TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveIntelTab(tab)}
              className={clsx(
                "flex items-center gap-1 px-3 py-2 text-[10px] font-bold uppercase tracking-wider transition-all",
                activeIntelTab === tab
                  ? "text-indigo-400 border-b-2 border-indigo-500 bg-indigo-500/10"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-900",
              )}
            >
              {tab === "OSINT" && <Radio size={10} />}
              {tab === "Twitter" && <Twitter size={10} />}
              {tab}
              {tab === "OSINT" && (
                <span className="bg-rose-500 text-white text-[8px] px-1 rounded-full ml-1">
                  1
                </span>
              )}
            </button>
          ))}
          <button className="ml-auto px-2 text-slate-500 hover:text-slate-300">
            <Info size={12} />
          </button>
        </div>

        {/* Intel Feed */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <div className="flex flex-col gap-2 p-2">
            {NEWS_ITEMS.map((news) => (
              <div
                key={news.id}
                className="bg-slate-800/30 border border-slate-700/50 rounded-lg p-3 hover:border-indigo-500/30 transition-all cursor-pointer"
              >
                {/* Header */}
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center">
                    <Radio size={12} className="text-indigo-400" />
                  </div>
                  <h4 className="flex-1 text-[11px] font-semibold text-slate-200 leading-tight">
                    {news.title}
                  </h4>
                  <span className="text-[9px] text-slate-500 whitespace-nowrap">
                    {news.time}
                  </span>
                </div>

                {/* Description */}
                <p className="text-[10px] text-slate-400 mb-3 line-clamp-2 leading-relaxed">
                  {news.description}
                </p>

                {/* Tags Row */}
                <div className="flex items-center gap-2 mb-2">
                  <span className="flex items-center gap-1 text-[9px] bg-slate-800 text-sky-400 px-1.5 py-0.5 rounded border border-slate-700">
                    <Twitter size={8} /> {news.source}
                  </span>
                  <span
                    className={clsx(
                      "text-[9px] px-1.5 py-0.5 rounded font-bold uppercase",
                      news.sentiment === "BULLISH"
                        ? "bg-emerald-900/30 text-emerald-400"
                        : "bg-rose-900/30 text-rose-400",
                    )}
                  >
                    {news.sentiment}
                  </span>
                  <span className="text-[8px] text-slate-500">
                    {news.sentimentConf}
                  </span>
                </div>

                {/* Actions Row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button className="flex items-center gap-1 text-[9px] text-indigo-400 hover:text-indigo-300 transition-colors">
                      <ExternalLink size={10} /> {news.relatedMarkets} related
                      markets
                    </button>
                    <button className="text-[9px] bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded font-bold hover:bg-rose-500/30 transition-colors">
                      ACTION
                    </button>
                  </div>
                </div>

                {/* Via */}
                <div className="mt-2 text-[8px] text-slate-600">
                  via {news.via}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
