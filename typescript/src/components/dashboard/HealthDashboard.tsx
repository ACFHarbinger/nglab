/**
 * Health Dashboard Widget for NGLab.
 *
 * Displays real-time health status of application components including
 * Arena, OrderBook, Polymarket scraper, and Python bindings.
 */

import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState, useCallback } from "react";

/** Component health status from the Rust backend. */
interface ComponentHealth {
    arena: boolean;
    orderbook: boolean;
    polymarket_scraper: boolean;
    python_binding: boolean;
}

/** Overall health status response from health_check command. */
interface HealthStatus {
    status: "healthy" | "degraded" | "unhealthy";
    version: string;
    uptime_seconds: number;
    components: ComponentHealth;
}

/** System information from get_system_info command. */
interface SystemInfo {
    cpu_count: number;
    os_name: string;
    arch: string;
}

/** Props for the HealthDashboard component. */
interface HealthDashboardProps {
    /** Refresh interval in milliseconds. Default: 10000 (10s) */
    refreshInterval?: number;
    /** Compact mode for sidebar display. Default: false */
    compact?: boolean;
}

/**
 * Format seconds as human-readable uptime string.
 * @param seconds - Uptime in seconds.
 * @returns Formatted string like "2h 30m 15s".
 */
function formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
}

/**
 * Get status color based on health status.
 * @param status - Overall health status.
 * @returns CSS color class name.
 */
function getStatusColor(status: string): string {
    switch (status) {
        case "healthy":
            return "#22c55e"; // green-500
        case "degraded":
            return "#eab308"; // yellow-500
        case "unhealthy":
            return "#ef4444"; // red-500
        default:
            return "#6b7280"; // gray-500
    }
}

/**
 * Health Dashboard Widget.
 *
 * Displays real-time health status of all application components
 * with auto-refresh functionality.
 *
 * @example
 * ```tsx
 * <HealthDashboard refreshInterval={5000} compact={false} />
 * ```
 */
export function HealthDashboard({
    refreshInterval = 10000,
    compact = false,
}: HealthDashboardProps) {
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchHealthData = useCallback(async () => {
        try {
            const [healthData, sysInfo] = await Promise.all([
                invoke<HealthStatus>("health_check"),
                invoke<SystemInfo>("get_system_info"),
            ]);
            setHealth(healthData);
            setSystemInfo(sysInfo);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to fetch health data");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchHealthData();
        const interval = setInterval(fetchHealthData, refreshInterval);
        return () => clearInterval(interval);
    }, [fetchHealthData, refreshInterval]);

    if (loading) {
        return (
            <div className="health-dashboard loading">
                <div className="spinner"></div>
                <span>Loading health status...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="health-dashboard error">
                <span className="error-icon">⚠️</span>
                <span>{error}</span>
                <button onClick={fetchHealthData}>Retry</button>
            </div>
        );
    }

    if (!health) return null;

    const components = [
        { name: "Arena", healthy: health.components.arena, icon: "🎯" },
        { name: "OrderBook", healthy: health.components.orderbook, icon: "📊" },
        { name: "Polymarket", healthy: health.components.polymarket_scraper, icon: "🌐" },
        { name: "Python", healthy: health.components.python_binding, icon: "🐍" },
    ];

    if (compact) {
        return (
            <div className="health-dashboard compact">
                <div
                    className="status-indicator"
                    style={{ backgroundColor: getStatusColor(health.status) }}
                    title={`Status: ${health.status}`}
                />
                <span className="version">v{health.version}</span>
            </div>
        );
    }

    return (
        <div className="health-dashboard">
            <div className="header">
                <h3>System Health</h3>
                <span
                    className="status-badge"
                    style={{ backgroundColor: getStatusColor(health.status) }}
                >
                    {health.status.toUpperCase()}
                </span>
            </div>

            <div className="info-row">
                <span className="label">Version:</span>
                <span className="value">{health.version}</span>
            </div>

            <div className="info-row">
                <span className="label">Uptime:</span>
                <span className="value">{formatUptime(health.uptime_seconds)}</span>
            </div>

            {systemInfo && (
                <div className="info-row">
                    <span className="label">System:</span>
                    <span className="value">
                        {systemInfo.os_name}/{systemInfo.arch} ({systemInfo.cpu_count} CPUs)
                    </span>
                </div>
            )}

            <div className="components">
                <h4>Components</h4>
                <div className="component-grid">
                    {components.map(({ name, healthy, icon }) => (
                        <div
                            key={name}
                            className={`component ${healthy ? "healthy" : "unhealthy"}`}
                        >
                            <span className="icon">{icon}</span>
                            <span className="name">{name}</span>
                            <span
                                className="dot"
                                style={{ backgroundColor: healthy ? "#22c55e" : "#ef4444" }}
                            />
                        </div>
                    ))}
                </div>
            </div>

            <div className="footer">
                <button onClick={fetchHealthData} className="refresh-btn">
                    🔄 Refresh
                </button>
            </div>

            <style>{`
        .health-dashboard {
          background: linear-gradient(145deg, #1a1a2e, #16213e);
          border-radius: 12px;
          padding: 16px;
          color: #e0e0e0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .health-dashboard.compact {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
        }
        .health-dashboard.loading,
        .health-dashboard.error {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          min-height: 100px;
        }
        .health-dashboard .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }
        .health-dashboard h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }
        .health-dashboard h4 {
          margin: 12px 0 8px;
          font-size: 14px;
          font-weight: 500;
          color: #a0a0a0;
        }
        .status-badge {
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
        }
        .status-indicator {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }
        .info-row {
          display: flex;
          justify-content: space-between;
          padding: 8px 0;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .info-row .label {
          color: #888;
        }
        .info-row .value {
          font-weight: 500;
        }
        .component-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 8px;
        }
        .component {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
        }
        .component .dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-left: auto;
        }
        .footer {
          margin-top: 16px;
          text-align: center;
        }
        .refresh-btn {
          background: rgba(255,255,255,0.1);
          border: none;
          padding: 8px 16px;
          border-radius: 8px;
          color: #e0e0e0;
          cursor: pointer;
          transition: background 0.2s;
        }
        .refresh-btn:hover {
          background: rgba(255,255,255,0.2);
        }
        .error-icon {
          font-size: 24px;
        }
        .spinner {
          width: 20px;
          height: 20px;
          border: 2px solid rgba(255,255,255,0.2);
          border-top-color: #fff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    );
}

export default HealthDashboard;
