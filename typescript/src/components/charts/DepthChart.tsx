
import { useMemo } from "react";

interface DepthLevel {
    price: number;
    total_quantity: number;
}

interface DepthChartProps {
    bids: DepthLevel[];
    asks: DepthLevel[];
    maxVolume?: number;
    height?: number;
}

export function DepthChart({ bids, asks, maxVolume, height = 100 }: DepthChartProps) {
    // We need to construct the points for the SVG areas
    // Bids: Sorted Descending Price (High to Low). Chart usually goes Left (Low) to Right (High).
    // Standard Depth Chart: X-axis is Price, Y-axis is Cumulative Volume.
    //   Left side: Bids (Green), Right side: Asks (Red).
    //   Center is the Mid Price.

    const data = useMemo(() => {
        // Cumulative Sums
        const bidCurves: { price: number; vol: number }[] = [];
        let bidAcc = 0;
        // Bids come sorted High to Low (Best Bid first).
        // For depth chart (Left side is Bid), we want prices increasing left to right?
        // Actually, usually Depth Chart is:
        //   X: Price. Y: Volume.
        //   Bids on Left (Lower prices), Asks on Right (Higher prices).
        //   But wait, Best Bid is the HIGHEST bid price. Best Ask is LOWEST ask price.
        //   So they meet in the middle.
        //   Bids curve starts from left (Low Price) and goes up to Best Bid (High Price)?
        //   NO, traditionally:
        //     Bids: Area on the left. X axis from MinBid to MaxBid (Best). Y is cumulative from Best down to Min?
        //     Let's follow standard CEX style:
        //       Center: Mid Price.
        //       Left of Center: Bids. increasing volume as we move LEFT (away from mid price).
        //       Right of Center: Asks. increasing volume as we move RIGHT (away from mid price).

        // Let's normalize to a 0-100 coordinate space or SVG pixels.

        if (bids.length === 0 && asks.length === 0) return { bidPoints: "", askPoints: "" };

        const sortedBids = [...bids].sort((a, b) => b.price - a.price); // Best bid first (High)
        const sortedAsks = [...asks].sort((a, b) => a.price - b.price); // Best ask first (Low)

        // Calculate accumulation
        const bidData: { price: number; accVol: number }[] = [];
        let bVol = 0;
        sortedBids.forEach(b => {
            bVol += b.total_quantity;
            bidData.push({ price: b.price, accVol: bVol });
        });

        const askData: { price: number; accVol: number }[] = [];
        let aVol = 0;
        sortedAsks.forEach(a => {
            aVol += a.total_quantity;
            askData.push({ price: a.price, accVol: aVol });
        });

        const maxCumVol = Math.max(bVol, aVol) * 1.1; // Add padding

        if (bidData.length === 0 && askData.length === 0) return { bidPoints: "", askPoints: "" };

        // Determine Ends of Price Scale
        const bestBid = sortedBids[0]?.price || 0;
        const worstBid = sortedBids[sortedBids.length - 1]?.price || bestBid * 0.9;
        const bestAsk = sortedAsks[0]?.price || bestBid;
        const worstAsk = sortedAsks[sortedAsks.length - 1]?.price || bestAsk * 1.1;

        const minPrice = Math.min(worstBid, worstAsk * 0.9); // roughly
        const maxPrice = Math.max(worstAsk, bestBid * 1.1);

        // Coordinate Mapping
        // X: Price (Min -> Max)
        // Y: Volume (0 -> MaxCumVol)
        const getX = (p: number) => ((p - minPrice) / (maxPrice - minPrice)) * 100;
        const getY = (v: number) => 100 - (v / maxCumVol) * 100; // SVG Y is top-down

        // Build Bid Polyline (Area)
        // It should start at bottom-left (worst price, 0 vol? No, worst price, max vol? No).
        // Bids accumulate starting from Best Bid (at Mid).
        // At Best Bid, Vol = Size[0]. At Worst Bid, Vol = Total Bid Vol.
        // So curve goes from (BestBid, Size0) to (WorstBid, Total).
        // And fill should go down to Y=100.

        let bidPoints = "";
        // Start at bottom right of bid side (Best Bid, 0 vol - effectively mid) -> actually Best Bid, vol=0 is not quite right,
        // usually it starts at (BestBid, 0) and goes up? No, first bar has volume.
        // Let's start path at (BestBid, 100) (Bottom)
        if (bidData.length > 0) {
            bidPoints += `${getX(bidData[0].price)},100 `; // Start bottom at best bid
            bidData.forEach((pt) => {
                bidPoints += `${getX(pt.price)},${getY(pt.accVol)} `;
            });
            bidPoints += `${getX(bidData[bidData.length - 1].price)},100`; // Close to bottom at worst bid
        }

        // Ask Polyline
        let askPoints = "";
        if (askData.length > 0) {
            askPoints += `${getX(askData[0].price)},100 `; // Start bottom at best ask
            askData.forEach((pt) => {
                askPoints += `${getX(pt.price)},${getY(pt.accVol)} `;
            });
            askPoints += `${getX(askData[askData.length - 1].price)},100`; // Close to bottom at worst ask
        }

        return { bidPoints, askPoints };
    }, [bids, asks]);

    return (
        <div className="w-full relative select-none" style={{ height }}>
            <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="bidGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="#10b981" stopOpacity={0.1} />
                    </linearGradient>
                    <linearGradient id="askGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.1} />
                    </linearGradient>
                </defs>

                {/* Grid Lines (Optional) */}
                <line x1="0" y1="50" x2="100" y2="50" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="2" />

                {/* Bids */}
                <polygon points={data.bidPoints} fill="url(#bidGradient)" stroke="#10b981" strokeWidth="0.5" />

                {/* Asks */}
                <polygon points={data.askPoints} fill="url(#askGradient)" stroke="#f43f5e" strokeWidth="0.5" />
            </svg>
        </div>
    );
}
