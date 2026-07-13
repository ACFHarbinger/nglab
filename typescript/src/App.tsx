import { useState, useEffect } from "react";
import { PriceChart } from "./components/charts/PriceChart";
import { OrderBook } from "./components/dashboard/OrderBook";
import ScraperTab from "./components/ScraperTab";
import AnalysisTab from "./components/AnalysisTab";
import PricingTab from "./components/PricingTab";
import PredictionTab from "./components/PredictionTab";
import TrainingTab from "./components/TrainingTab";
import NewsTab from "./components/NewsTab";
import VaultTab from "./components/VaultTab";
import ModelRegistryTab from "./components/models/ModelRegistryTab";
import FeaturesTab from "./components/features/FeaturesTab";
import DataManagerTab from "./components/data-manager/DataManagerTab";
import PortfolioTab from "./components/portfolio/PortfolioTab";
import AlternativeDataTab from "./components/alternative-data/AlternativeDataTab";
import { AccountTab } from "./components/AccountTab";
import { LoginModal } from "./components/LoginModal";
import { FavoritesTab } from "./components/FavoritesTab";
import { useArena } from "./hooks/useArena";
import { useFavorites } from "./hooks/useFavorites";
import { useStreaming } from "./context/StreamingContext";
import {
  Play,
  Square,
  RotateCcw,
  Activity,
  LineChart,
  Download,
  Brain,
  Database,
  Calculator,
  ArrowUpRight,
  LayoutDashboard,
  Search,
  Star,
  Users,
  ShoppingCart,
  LogIn,
  User,
  GraduationCap,
  Newspaper,
  ShieldCheck,
  AlertCircle,
  Wand2,
  HardDrive,
  Briefcase,
  Compass,
  Radio,
} from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { usePolymarket } from "./hooks/usePolymarket";
import clsx from "clsx";
import { TerminalLayout } from "./components/terminal/TerminalLayout";
import { DashboardOverview } from "./components/dashboard/DashboardOverview";
import { StreamStatusIndicator } from "./components/streaming/StreamStatusIndicator";

/**
 * The root component of the NGLab application.
 *
 * Manages global application state including the active tab, simulation logs,
 * and real-time arena data. Provides the main layout and navigation.
 */
function App() {
  const { data: arenaData, history, isRunning, start, stop } = useArena();
  const {
    livePrices,
    isStreaming,
    activeMarket,
    startStream,
    stopStream,
    setActiveMarket,
  } = usePolymarket();
  const { setIsLoggedIn } = useStreaming();
  const [logs, setLogs] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<
    | "dashboard"
    | "simulation"
    | "scraper"
    | "analysis"
    | "prediction"
    | "pricing"
    | "terminal"
    | "terminal"
    | "training"
    | "models"
    | "features"
    | "portfolio"
    | "explorer"
    | "altdata"
    | "news"
    | "vault"
    | "account"
    | "data"
    | "favorites"
  >("dashboard");

  const {
    favorites,
    favoriteIds,
    addFavorite,
    removeFavorite,
    isFavorite,
    toggleFavorite,
    refreshFavorites,
    lastMessage,
    isError,
    setLastMessage,
  } = useFavorites();

  // Clear toast after delay
  useEffect(() => {
    if (lastMessage) {
      const timer = setTimeout(() => setLastMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [lastMessage, setLastMessage]);

  const [showLoginModal, setShowLoginModal] = useState(false);
  const [currentUser, setCurrentUser] = useState<string | null>(null);

  // --- SESSION RESTORATION: Check for active session on mount ---
  useEffect(() => {
    const restoreSession = async () => {
      try {
        const response = await invoke<{
          success: boolean;
          message: string;
          username: string | null;
        }>("get_current_user");
        if (response.success && response.username) {
          console.log("👤 Restored active session for:", response.username);
          setCurrentUser(response.username);
          refreshFavorites();
        }
      } catch (err) {
        console.warn("Session restoration failed:", err);
      }
    };
    restoreSession();
  }, [refreshFavorites]);

  // --- LOGIN SYNC: Notify StreamingContext about login status ---
  useEffect(() => {
    setIsLoggedIn(currentUser !== null);
  }, [currentUser, setIsLoggedIn]);

  // Listen for logs
  useEffect(() => {
    const unlisten = listen("arena-update", (event: any) => {
      if (event.payload.message) {
        setLogs((prev) => [event.payload.message, ...prev].slice(0, 50));
      }
    });

    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  /**
   * Logs the current user out, clearing both frontend and backend session state.
   * Locks the secure vault to protect user data.
   */
  const handleLogout = async () => {
    try {
      await invoke("logout");
      await invoke("lock_vault");
      setCurrentUser(null);
      // Favorites will be cleared automatically since the vault is locked
      // and useFavorites will fail its next refresh, but we can clear them manually too
    } catch (e) {
      console.error("Logout failed:", e);
    }
  };

  /**
   * Triggers the start of the simulation via the `useArena` hook.
   */
  const handleStart = async () => {
    start();
  };

  /**
   * Triggers the stop of the simulation via the `useArena` hook.
   */
  const handleStop = async () => {
    stop();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30 flex flex-col h-screen">
      {/* Header - PVE.trade Style */}
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2 bg-slate-950 z-10">
        {/* Left: Logo + Main Nav */}
        <div className="flex items-center gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-lg font-bold tracking-tight">
              <span className="text-white">NGL</span>
              <span className="text-indigo-400">.trade</span>
            </h1>
          </div>

          {/* Main Navigation */}
          <nav className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab("dashboard")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "dashboard"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <LayoutDashboard size={16} /> Dashboard
            </button>
            <button
              onClick={() => setActiveTab("prediction")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "prediction"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Brain size={16} /> Intelligence
            </button>
            <button
              onClick={() => setActiveTab("terminal")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "terminal"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <ArrowUpRight size={16} /> Markets
            </button>

            <button
              onClick={() => setActiveTab("portfolio")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "portfolio"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Briefcase size={16} /> Portfolio
            </button>
            <button
              onClick={() => setActiveTab("explorer")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "explorer"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Compass size={16} /> Explorer
            </button>
            <button
              onClick={() => setActiveTab("altdata")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "altdata"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Radio size={16} /> Alt Data
            </button>
            <button
              onClick={() => setActiveTab("favorites")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "favorites"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Star size={16} /> Favorites
              {favorites.length > 0 && (
                <span className="text-[10px] bg-yellow-500/20 text-yellow-500 px-1.5 py-0.5 rounded-full">
                  {favorites.length}
                </span>
              )}
            </button>
            <button className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition-all">
              <Users size={16} /> Friends
            </button>
            <button
              onClick={() => setActiveTab("account")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "account"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <User size={16} /> Account
            </button>
            {/* Separator */}
            <div className="w-px h-6 bg-slate-700 mx-2" />
            {/* Tool tabs moved from secondary nav */}
            <button
              onClick={() => setActiveTab("simulation")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "simulation"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <LineChart size={16} /> Simulation
            </button>
            <button
              onClick={() => setActiveTab("scraper")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "scraper"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Download size={16} /> Scraper
            </button>
            <button
              onClick={() => setActiveTab("pricing")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "pricing"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Calculator size={16} /> Pricing
            </button>
            <button
                onClick={() => setActiveTab("models")}
                className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "models"
                    ? "text-white bg-slate-800"
                    : "text-slate-400 hover:text-white hover:bg-slate-800/50",
                )}
            >
                <Database size={16} /> Models
            </button>
            <button
                onClick={() => setActiveTab("features")}
                className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "features"
                    ? "text-white bg-slate-800"
                    : "text-slate-400 hover:text-white hover:bg-slate-800/50",
                )}
            >
                <Wand2 size={16} /> Features
            </button>
            <button
              onClick={() => setActiveTab("training")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "training"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <GraduationCap size={16} /> Training
            </button>
            <button
                onClick={() => setActiveTab("data")}
                className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "data"
                    ? "text-white bg-slate-800"
                    : "text-slate-400 hover:text-white hover:bg-slate-800/50",
                )}
            >
                <HardDrive size={16} /> Data
            </button>
            <button
              onClick={() => setActiveTab("news")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "news"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <Newspaper size={16} /> News
            </button>
            <button
              onClick={() => setActiveTab("vault")}
              className={clsx(
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                activeTab === "vault"
                  ? "text-white bg-slate-800"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )}
            >
              <ShieldCheck size={16} /> Vault
            </button>
          </nav>
        </div>

        {/* Right: Search + Actions */}
        <div className="flex items-center gap-6">
          <StreamStatusIndicator />
          
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="search_disabled"
              className="w-48 bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-4 py-1.5 text-sm text-slate-400 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          {/* Simulation Controls (when in simulation tab) */}
          {activeTab === "simulation" && (
            <div className="flex gap-2">
              {!isRunning ? (
                <button
                  onClick={handleStart}
                  className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-1.5 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
                >
                  <Play size={16} fill="currentColor" /> Start
                </button>
              ) : (
                <button
                  onClick={handleStop}
                  className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 px-3 py-1.5 rounded-lg transition-colors font-medium text-sm"
                >
                  <Square size={16} fill="currentColor" /> Stop
                </button>
              )}
              <button className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700">
                <RotateCcw size={16} />
              </button>
            </div>
          )}

          {/* Login Button */}
          {currentUser ? (
            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-300">
                Hi,{" "}
                <span className="font-medium text-white">{currentUser}</span>
              </span>
              <button
                onClick={handleLogout}
                className="text-sm text-slate-400 hover:text-white transition-colors"
              >
                Logout
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowLoginModal(true)}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded-lg transition-colors font-medium text-sm"
            >
              <LogIn size={16} /> Login
            </button>
          )}
        </div>
      </header>

      {/* Login Modal */}
      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onLoginSuccess={(username) => {
          setCurrentUser(username);
          refreshFavorites();
        }}
      />

      {/* Live Indicator Bar (only shows when streaming) */}
      {isStreaming && activeMarket && (
        <div className="border-b border-slate-800 bg-slate-900/50 w-full px-4 py-1.5">
          <span className="flex items-center gap-1.5 text-[10px] text-green-500 font-mono tracking-wide uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Polymarket Live: {activeMarket.title.slice(0, 40)}...
          </span>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 overflow-hidden relative">
        {activeTab === "dashboard" ? (
          <DashboardOverview
            onNavigateToTerminal={(id) => {
              console.log("Navigating to market:", id);
              setActiveTab("terminal");
              // Future: Set active market by ID here
            }}
            livePrices={livePrices}
            favorites={favorites}
            favoriteIds={favoriteIds}
            toggleFavorite={toggleFavorite}
            onNavigateToFavorites={() => setActiveTab("favorites")}
            riskMetrics={
              arenaData
                ? {
                    riskScore: arenaData.risk_score,
                    drawdown: arenaData.current_drawdown,
                    varValue: arenaData.current_var,
                  }
                : undefined
            }
          />
        ) : activeTab === "simulation" ? (
          <div className="grid grid-cols-12 gap-6 h-full p-4">
            {/* Left: Charts & Orderbook */}
            <div className="col-span-8 flex flex-col gap-6 h-full overflow-hidden">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 relative overflow-hidden flex flex-col">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    Live Price (Polymarket)
                  </h3>
                  {isStreaming && (
                    <div className="flex gap-2">
                      {Object.entries(livePrices)
                        .slice(0, 3)
                        .map(([id, price]) => (
                          <span
                            key={id}
                            className="text-xs font-mono bg-slate-800 px-1.5 py-0.5 rounded text-green-400"
                          >
                            {activeMarket?.outcomes.find((o) => o.id === id)
                              ?.name || id.slice(0, 4)}
                            : ${price.toFixed(3)}
                          </span>
                        ))}
                    </div>
                  )}
                </div>
                <div className="flex-1 w-full bg-slate-950/50 rounded-lg overflow-hidden">
                  <PriceChart data={history} />
                </div>
              </div>

              <div className="h-64 bg-slate-900/50 border border-slate-800 rounded-xl p-4 overflow-hidden">
                <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">
                  Order Book Depth
                </h3>
                <OrderBook
                  book={
                    arenaData?.orderbook || { bids: {}, asks: {}, timestamp: 0 }
                  }
                />
              </div>
            </div>

            {/* Right: Stats & Logs */}
            <div className="col-span-4 flex flex-col gap-6 h-full overflow-hidden">
              {/* Stats Panel */}
              <div className="bg-indigo-600/10 border border-indigo-500/20 rounded-xl p-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-indigo-300 uppercase font-bold">
                      Current Step
                    </p>
                    <p className="text-2xl font-mono">{arenaData?.step || 0}</p>
                  </div>
                  <div>
                    <p className="text-xs text-indigo-300 uppercase font-bold">
                      Portfolio Value
                    </p>
                    <p
                      className={`text-2xl font-mono ${(arenaData?.portfolio_value || 0) >= 10000 ? "text-emerald-400" : "text-rose-400"}`}
                    >
                      {arenaData?.portfolio_value?.toFixed(2) || "10000.00"}
                    </p>
                  </div>
                </div>
              </div>

              {/* Log Panel */}
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col overflow-hidden">
                <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                  Agent Logs
                </h3>
                <div className="flex-1 overflow-y-auto font-mono text-xs space-y-1 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                  {logs.length === 0 && (
                    <p className="text-slate-600 italic">
                      Waiting for simulation data...
                    </p>
                  )}
                  {logs.map((log, i) => (
                    <div
                      key={i}
                      className="text-slate-300 border-l border-slate-700 pl-2 py-0.5"
                    >
                      <span className="text-indigo-400 opacity-50 mr-2">
                        [{new Date().toLocaleTimeString()}]
                      </span>
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : activeTab === "scraper" ? (
          <div className="h-full bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
            <ScraperTab
              livePrices={livePrices}
              isStreaming={isStreaming}
              startStream={startStream}
              stopStream={stopStream}
              activeMarket={activeMarket}
              setActiveMarket={setActiveMarket}
            />
          </div>
        ) : activeTab === "portfolio" ? (
          <PortfolioTab />
        ) : activeTab === "explorer" ? (
          <AnalysisTab
            livePrices={livePrices}
            isStreaming={isStreaming}
            activeMarket={activeMarket}
          />
        ) : activeTab === "altdata" ? (
          <AlternativeDataTab />
        ) : activeTab === "prediction" ? (
          <PredictionTab
            livePrices={livePrices}
            isStreaming={isStreaming}
            activeMarket={activeMarket}
          />
        ) : activeTab === "terminal" ? (
          <TerminalLayout
            livePrices={livePrices}
            activeMarket={activeMarket}
            startStream={startStream}
            stopStream={stopStream}
            favoriteIds={favoriteIds}
            favorites={favorites}
            toggleFavorite={toggleFavorite}
          />
        ) : activeTab === "data" ? (
          <DataManagerTab />
        ) : activeTab === "models" ? (
          <ModelRegistryTab />
        ) : activeTab === "features" ? (
          <FeaturesTab />
        ) : activeTab === "training" ? (
          <TrainingTab />
        ) : activeTab === "news" ? (
          <NewsTab />
        ) : activeTab === "pricing" ? (
          <PricingTab />
        ) : activeTab === "vault" ? (
          <VaultTab />
        ) : activeTab === "account" ? (
          <AccountTab
            isStreaming={isStreaming}
            startStream={startStream}
            stopStream={stopStream}
          />
        ) : activeTab === "favorites" ? (
          <FavoritesTab
            favorites={favorites}
            favoriteIds={favoriteIds}
            addFavorite={addFavorite}
            removeFavorite={removeFavorite}
            isFavorite={isFavorite}
            toggleFavorite={toggleFavorite}
            livePrices={livePrices}
            onNavigateToMarket={(marketId) => {
              console.log("Navigate to market:", marketId);
              setActiveTab("terminal");
            }}
          />
        ) : null}
      </main>
      {/* Toast Notification */}
      {lastMessage && (
        <div
          className={clsx(
            "fixed bottom-6 left-1/2 -translate-x-1/2 z-[10000] px-4 py-3 rounded-xl border shadow-2xl flex items-center gap-3 animate-in fade-in slide-in-from-bottom-4 duration-300",
            isError
              ? "bg-rose-950 border-rose-500 text-rose-200"
              : "bg-emerald-950 border-emerald-500 text-emerald-200",
          )}
        >
          {isError ? <AlertCircle size={18} /> : <ShieldCheck size={18} />}
          <span className="text-sm font-medium">{lastMessage}</span>
          <button
            onClick={() => setLastMessage(null)}
            className="ml-2 hover:opacity-70"
          >
            <ShieldCheck size={14} className="rotate-45" />{" "}
            {/* Primitive close icon fallback */}
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
