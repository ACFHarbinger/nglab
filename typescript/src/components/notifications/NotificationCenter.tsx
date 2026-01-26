import { useState, useEffect } from "react";
import {
    Bell,
    X,
    Trash2,
    Settings,
    Clock,
    AlertCircle,
    CheckCircle2,
    Info,
    Layers,
    Search
} from "lucide-react";
import clsx from "clsx";
import { invoke } from "@tauri-apps/api/core";

interface Alert {
    id: string;
    alert_type: string;
    target: string;
    value: number;
    active: boolean;
}

interface Notification {
    id: string;
    title: string;
    message: string;
    timestamp: string;
    priority: "Low" | "Medium" | "High";
}

interface NotificationCenterProps {
    onClose: () => void;
}

export function NotificationCenter({ onClose }: NotificationCenterProps) {
    const [view, setView] = useState<"Alerts" | "Notifications">("Alerts");
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [loading, setLoading] = useState(false);

    const fetchAlerts = async () => {
        try {
            const data: Alert[] = await invoke("get_alerts");
            setAlerts(data);
        } catch (e) {
            console.error("Failed to fetch alerts", e);
        }
    };

    const fetchNotifications = async () => {
        try {
            const data: Notification[] = await invoke("get_notifications");
            setNotifications(data);
        } catch (e) {
            console.error("Failed to fetch notifications", e);
        }
    };

    const clearNotifications = async () => {
        try {
            await invoke("clear_notifications");
            setNotifications([]);
        } catch (e) {
            console.error("Failed to clear notifications", e);
        }
    };

    useEffect(() => {
        fetchAlerts();
        fetchNotifications();
        // Poll for updates every 5 seconds
        const interval = setInterval(() => {
            if (view === "Alerts") fetchAlerts();
            else fetchNotifications();
        }, 5000);
        return () => clearInterval(interval);
    }, [view]);

    return (
        <div className="flex flex-col h-full bg-slate-950 border-l border-slate-800 w-80 shadow-2xl">
            {/* Header */}
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                <div className="flex items-center gap-2">
                    <Bell size={18} className="text-indigo-400" />
                    <h2 className="font-bold text-white">Activity Center</h2>
                </div>
                <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
                    <X size={20} />
                </button>
            </div>

            {/* View Switcher */}
            <div className="flex p-2 bg-slate-900/30">
                <button
                    onClick={() => setView("Alerts")}
                    className={clsx(
                        "flex-1 py-1.5 rounded-md text-xs font-bold transition-all",
                        view === "Alerts" ? "bg-slate-800 text-white shadow" : "text-slate-500 hover:text-slate-300"
                    )}
                >
                    Alerts ({alerts.length})
                </button>
                <button
                    onClick={() => setView("Notifications")}
                    className={clsx(
                        "flex-1 py-1.5 rounded-md text-xs font-bold transition-all",
                        view === "Notifications" ? "bg-slate-800 text-white shadow" : "text-slate-500 hover:text-slate-300"
                    )}
                >
                    History ({notifications.length})
                </button>
            </div>

            {/* List Area */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                {view === "Alerts" ? (
                    <div className="space-y-3">
                        {alerts.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-12 text-slate-600">
                                <Search size={32} className="mb-2 opacity-20" />
                                <p className="text-sm">No active alerts</p>
                            </div>
                        ) : (
                            alerts.map(alert => (
                                <AlertItem key={alert.id} alert={alert} />
                            ))
                        )}
                    </div>
                ) : (
                    <div className="space-y-3">
                        {notifications.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-12 text-slate-600">
                                <Layers size={32} className="mb-2 opacity-20" />
                                <p className="text-sm">Notification history empty</p>
                            </div>
                        ) : (
                            notifications.map(notification => (
                                <NotificationItem key={notification.id} notification={notification} />
                            ))
                        )}
                    </div>
                )}
            </div>

            {/* Footer */}
            {view === "Notifications" && notifications.length > 0 && (
                <div className="p-4 border-t border-slate-800 bg-slate-900/20">
                    <button
                        onClick={clearNotifications}
                        className="w-full flex items-center justify-center gap-2 py-2 text-xs font-bold text-slate-400 hover:text-rose-400 transition-colors"
                    >
                        <Trash2 size={14} /> Clear History
                    </button>
                </div>
            )}
        </div>
    );
}

function AlertItem({ alert }: { alert: Alert }) {
    return (
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 group hover:border-slate-700 transition-all">
            <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">{alert.alert_type}</span>
                <div className={clsx(
                    "w-2 h-2 rounded-full",
                    alert.active ? "bg-emerald-500 animate-pulse" : "bg-slate-700"
                )} />
            </div>
            <div className="flex items-center justify-between">
                <div>
                    <span className="text-sm font-bold text-white block">{alert.target}</span>
                    <span className="text-xs text-slate-400">${alert.value.toFixed(4)}</span>
                </div>
                <button className="text-slate-600 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-all">
                    <X size={14} />
                </button>
            </div>
        </div>
    );
}

function NotificationItem({ notification }: { notification: Notification }) {
    const priorityColors = {
        Low: "text-blue-400 bg-blue-400/10 border-blue-400/20",
        Medium: "text-amber-400 bg-amber-400/10 border-amber-400/20",
        High: "text-rose-400 bg-rose-400/10 border-rose-400/20"
    };

    return (
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
                <span className={clsx(
                    "px-1.5 py-0.5 rounded text-[9px] font-bold uppercase border",
                    priorityColors[notification.priority]
                )}>
                    {notification.priority}
                </span>
                <span className="text-[10px] text-slate-600">{notification.timestamp}</span>
            </div>
            <div>
                <span className="text-xs font-bold text-slate-200 block">{notification.title}</span>
                <p className="text-[11px] text-slate-500 leading-relaxed mt-1">{notification.message}</p>
            </div>
        </div>
    );
}
