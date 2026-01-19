/**
 * @module components/dashboard/DashboardOverview
 * @description Master layout for the landing dashboard, aggregating profile, stats, and trending markets.
 */
import { UserProfileWidget } from "./UserProfileWidget";
import { MarketStatsWidget } from "./MarketStatsWidget";
import { TrendingMarketsWidget } from "./TrendingMarketsWidget";
import { GlobalActivityWidget } from "./GlobalActivityWidget";
import { FavoriteMarketsWidget } from "./FavoriteMarketsWidget";
import { FavoriteMarket } from "../../hooks/useFavorites";
import { RiskDashboardWidget } from "./RiskDashboardWidget";

interface DashboardOverviewProps {
  onNavigateToTerminal: (marketId: string) => void;
  livePrices?: Record<string, number>;
  favorites?: FavoriteMarket[];
  onNavigateToFavorites?: () => void;
  riskMetrics?: {
    riskScore: number;
    drawdown: number;
    varValue: number;
  };
}

export function DashboardOverview({
  onNavigateToTerminal,
  livePrices,
  favorites = [],
  onNavigateToFavorites,
  riskMetrics,
}: DashboardOverviewProps) {
  return (
    <div className="flex h-full w-full bg-slate-950 overflow-hidden">
      {/* Left Column: User Profile & Risk (Fixed width) */}
      <div className="w-80 border-r border-slate-800 p-4 overflow-y-auto custom-scrollbar hidden xl:block space-y-6">
        <UserProfileWidget />
        {riskMetrics && (
          <RiskDashboardWidget
            riskScore={riskMetrics.riskScore}
            drawdown={riskMetrics.drawdown}
            varValue={riskMetrics.varValue}
          />
        )}
      </div>

      {/* Center Column: Main Content (Flexible) */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-slate-800 overflow-y-auto custom-scrollbar">
        {/* Top Section: Market Overview Stats */}
        <div className="p-6 pb-4">
          <MarketStatsWidget />
        </div>

        {/* Favorites Section (if any) */}
        {favorites.length > 0 && (
          <div className="px-6 pb-4">
            <FavoriteMarketsWidget
              favorites={favorites}
              livePrices={livePrices}
              onSelectMarket={onNavigateToTerminal}
              onViewAll={onNavigateToFavorites}
            />
          </div>
        )}

        {/* Bottom Section: Trending Markets */}
        <div className="flex-1 px-6 pb-6 min-h-0">
          <TrendingMarketsWidget
            onSelectMarket={onNavigateToTerminal}
            livePrices={livePrices}
          />
        </div>
      </div>

      {/* Right Column: Global Activity (Fixed width) */}
      <div className="w-80 hidden lg:block">
        <GlobalActivityWidget />
      </div>
    </div>
  );
}
