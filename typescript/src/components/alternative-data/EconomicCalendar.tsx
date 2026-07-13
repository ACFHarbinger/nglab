import { useState, useEffect } from "react";
import { 
  Calendar,
  Clock,
  AlertTriangle,
  ChevronRight
} from "lucide-react";
import clsx from "clsx";

type Impact = "high" | "medium" | "low";

interface EconomicEvent {
  id: string;
  title: string;
  country: string;
  timestamp: Date;
  impact: Impact;
  previous: string;
  forecast: string;
  actual?: string;
}

const MOCK_EVENTS: EconomicEvent[] = [
  {
    id: "1",
    title: "FOMC Rate Decision",
    country: "US",
    timestamp: new Date(Date.now() + 2 * 60 * 60 * 1000), // 2 hours from now
    impact: "high",
    previous: "5.50%",
    forecast: "5.50%"
  },
  {
    id: "2",
    title: "Initial Jobless Claims",
    country: "US",
    timestamp: new Date(Date.now() + 5 * 60 * 60 * 1000),
    impact: "medium",
    previous: "215K",
    forecast: "220K"
  },
  {
    id: "3",
    title: "ECB Press Conference",
    country: "EU",
    timestamp: new Date(Date.now() + 24 * 60 * 60 * 1000),
    impact: "high",
    previous: "-",
    forecast: "-"
  },
  {
    id: "4",
    title: "UK CPI (YoY)",
    country: "GB",
    timestamp: new Date(Date.now() + 48 * 60 * 60 * 1000),
    impact: "medium",
    previous: "2.3%",
    forecast: "2.1%"
  },
  {
    id: "5",
    title: "Japan GDP (QoQ)",
    country: "JP",
    timestamp: new Date(Date.now() + 72 * 60 * 60 * 1000),
    impact: "low",
    previous: "-0.1%",
    forecast: "0.2%"
  }
];

export function EconomicCalendar() {
  return (
    <div className="h-full p-6 overflow-y-auto">
      <div className="max-w-4xl mx-auto">
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Calendar size={16} className="text-indigo-400" />
              <span className="font-bold text-slate-200">Upcoming Events</span>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                High
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-500" />
                Medium
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-slate-500" />
                Low
              </span>
            </div>
          </div>
          
          <table className="w-full text-left">
            <thead>
              <tr className="bg-slate-950/50 border-b border-slate-800">
                <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase">Time</th>
                <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase">Event</th>
                <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase text-center">Impact</th>
                <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase text-right">Previous</th>
                <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase text-right">Forecast</th>
                <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase text-right">Countdown</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {MOCK_EVENTS.map(event => (
                <EventRow key={event.id} event={event} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function EventRow({ event }: { event: EconomicEvent }) {
  const [countdown, setCountdown] = useState("");

  useEffect(() => {
    const updateCountdown = () => {
      const diff = event.timestamp.getTime() - Date.now();
      if (diff <= 0) {
        setCountdown("Now");
        return;
      }
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      setCountdown(`${hours}h ${minutes}m`);
    };
    
    updateCountdown();
    const interval = setInterval(updateCountdown, 60000);
    return () => clearInterval(interval);
  }, [event.timestamp]);

  const impactColor = event.impact === "high" ? "bg-rose-500"
    : event.impact === "medium" ? "bg-amber-500"
    : "bg-slate-500";

  return (
    <tr className="hover:bg-slate-800/30 transition-colors">
      <td className="px-4 py-4">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Clock size={12} />
          {event.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
        <div className="text-[10px] text-slate-600 mt-0.5">
          {event.timestamp.toLocaleDateString([], { month: 'short', day: 'numeric' })}
        </div>
      </td>
      <td className="px-4 py-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
            {event.country}
          </span>
          <span className="font-medium text-slate-200">{event.title}</span>
        </div>
      </td>
      <td className="px-4 py-4 text-center">
        <span className={clsx("w-3 h-3 rounded-full inline-block", impactColor)} />
      </td>
      <td className="px-4 py-4 text-right font-mono text-sm text-slate-400">
        {event.previous}
      </td>
      <td className="px-4 py-4 text-right font-mono text-sm text-slate-300">
        {event.forecast}
      </td>
      <td className="px-4 py-4 text-right">
        <span className={clsx(
          "font-mono text-sm font-bold px-2 py-1 rounded",
          countdown === "Now" 
            ? "bg-rose-500/20 text-rose-400 animate-pulse"
            : "text-indigo-400"
        )}>
          {countdown}
        </span>
      </td>
    </tr>
  );
}
