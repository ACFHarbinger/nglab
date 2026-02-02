import { useState } from "react";
import { 
  Newspaper, 
  MessageCircle, 
  LinkIcon, 
  CalendarDays,
  Activity,
  AlertTriangle,
  TrendingUp,
  TrendingDown
} from "lucide-react";
import clsx from "clsx";
import { NewsFeed } from "./NewsFeed";
import { SocialSentiment } from "./SocialSentiment";
import { EconomicCalendar } from "./EconomicCalendar";

type SubTab = "news" | "sentiment" | "onchain" | "calendar";

/**
 * AlternativeDataTab
 * 
 * Container for all alternative data sources:
 * - News Feed (RSS aggregation)
 * - Social Sentiment (Twitter/Reddit)
 * - On-Chain Data (Whale tracking)
 * - Economic Calendar
 */
export default function AlternativeDataTab() {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>("news");

  // Mock data quality KPIs
  const dataQuality = {
    sources: 12,
    latency: "1.2s",
    freshness: "Real-time",
    completeness: 94
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Header with Data Quality KPIs */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/40">
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
          <SubTabButton 
            active={activeSubTab === "news"} 
            onClick={() => setActiveSubTab("news")}
            icon={Newspaper}
            label="News"
          />
          <SubTabButton 
            active={activeSubTab === "sentiment"} 
            onClick={() => setActiveSubTab("sentiment")}
            icon={MessageCircle}
            label="Sentiment"
          />
          <SubTabButton 
            active={activeSubTab === "onchain"} 
            onClick={() => setActiveSubTab("onchain")}
            icon={LinkIcon}
            label="On-Chain"
          />
          <SubTabButton 
            active={activeSubTab === "calendar"} 
            onClick={() => setActiveSubTab("calendar")}
            icon={CalendarDays}
            label="Calendar"
          />
        </div>

        <div className="flex items-center gap-6 text-xs">
          <div className="flex items-center gap-2">
            <Activity size={14} className="text-emerald-400" />
            <span className="text-slate-500">Sources:</span>
            <span className="font-bold text-slate-200">{dataQuality.sources}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Latency:</span>
            <span className="font-bold text-slate-200">{dataQuality.latency}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Completeness:</span>
            <span className="font-bold text-emerald-400">{dataQuality.completeness}%</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeSubTab === "news" && <NewsFeed />}
        {activeSubTab === "sentiment" && <SocialSentiment />}
        {activeSubTab === "onchain" && (
          <div className="h-full flex items-center justify-center text-slate-500 italic">
            On-Chain data integration coming soon...
          </div>
        )}
        {activeSubTab === "calendar" && <EconomicCalendar />}
      </div>
    </div>
  );
}

function SubTabButton({ active, onClick, icon: Icon, label }: any) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all",
        active 
          ? "bg-slate-800 text-white shadow-sm" 
          : "text-slate-500 hover:text-slate-300"
      )}
    >
      <Icon size={14} />
      {label}
    </button>
  );
}
