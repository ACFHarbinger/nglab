import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core"; // Updated for Tauri v2
import { save } from "@tauri-apps/plugin-dialog";
import { Save, Activity } from "lucide-react";

interface OutcomeInfo {
    id: string;
    name: string;
}

interface MarketMetadata {
    title: string;
    outcomes: OutcomeInfo[];
}

export interface ScraperTabProps {
    livePrices: Record<string, number>;
    isStreaming: boolean;
    startStream: (marketSource: string, metadata: MarketMetadata) => Promise<void>;
    stopStream: () => Promise<void>;
    activeMarket: MarketMetadata | null;
    setActiveMarket: (metadata: MarketMetadata | null) => void;
}

/**
 * Component for scraping market data from Polymarket.
 *
 * Interacts with the Rust backend's 'web' module to resolve
 * Polymarket IDs/URLs and download historical price data
 * for binary prediction markets.
 */
export default function ScraperTab({
    livePrices,
    isStreaming,
    startStream,
    stopStream,
    activeMarket
}: ScraperTabProps) {
    const [input, setInput] = useState('');
    // Local metadata track current search result
    const [localMetadata, setLocalMetadata] = useState<MarketMetadata | null>(null);

    // Display metadata is determined by active stream OR local search
    const displayMetadata = localMetadata || activeMarket;

    const [selectedOutcomes, setSelectedOutcomes] = useState<Set<string>>(new Set());

    // Config
    const [frequency, setFrequency] = useState('Daily');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

    const [status, setStatus] = useState('');
    const [loading, setLoading] = useState(false);

    // Sync active market to local view on mount/update
    useEffect(() => {
        if (activeMarket) {
            setLocalMetadata(activeMarket);
            setInput(activeMarket.title);
            const allIds = new Set(activeMarket.outcomes.map(o => o.id));
            setSelectedOutcomes(allIds);
        }
    }, [activeMarket]);

    /**
     * Resolves a Polymarket URL or slug into market metadata (title and outcomes).
     * Interacts with the 'resolve_polymarket_id' Tauri command.
     */
    const handleFindMarket = async () => {
        if (!input) {
            setStatus('Please enter a URL or Slug.');
            return;
        }
        setLoading(true);
        setStatus('Resolving market...');
        try {
            const metadata = await invoke<MarketMetadata>('resolve_polymarket_id', { input: input.trim() });
            setLocalMetadata(metadata);
            // Select all by default
            const allIds = new Set(metadata.outcomes.map(o => o.id));
            setSelectedOutcomes(allIds);
            setStatus('Market found. Select outcomes.');
        } catch (error: any) {
            console.error(error);
            setStatus(`Error: ${error}`);
        } finally {
            setLoading(false);
        }
    };

    const toggleOutcome = (id: string) => {
        const newSet = new Set(selectedOutcomes);
        if (newSet.has(id)) {
            newSet.delete(id);
        } else {
            newSet.add(id);
        }
        setSelectedOutcomes(newSet);
    };

    /**
     * Triggers the historical data download for the selected outcomes.
     * Prompts the user for a save location and invokes 'scrape_polymarket'.
     */
    const handleScrape = async () => {
        if (selectedOutcomes.size === 0) {
            setStatus('Please select at least one outcome.');
            return;
        }

        setLoading(true);
        setStatus('Scraping...');

        try {
            const outputPath = await save({
                filters: [{
                    name: 'CSV',
                    extensions: ['csv'],
                }],
            });

            if (!outputPath) {
                setLoading(false);
                setStatus('Cancelled.');
                return;
            }

            // Convert set to array
            const tokens = Array.from(selectedOutcomes);

            // Dates: Convert to ISO sting for Rust
            const start = startDate ? new Date(startDate).toISOString() : null;
            const end = endDate ? new Date(endDate).toISOString() : null;

            await invoke('scrape_polymarket', {
                marketSource: input,
                tokenIds: tokens,
                frequency,
                startDate: start,
                endDate: end,
                outputPath
            });

            setStatus('Success! CSV saved.');
        } catch (error: any) {
            console.error(error);
            setStatus(`Error: ${error}`);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Starts streaming live prices for the current market.
     */
    const handleStream = async () => {
        if (!displayMetadata || isStreaming) return;

        setStatus("Starting stream...");

        try {
            await startStream(input || displayMetadata.title, displayMetadata);
            setStatus("Streaming live prices...");
        } catch (e: any) {
            console.error(e);
            setStatus(`Stream Error: ${e}`);
        }
    };

    const reset = () => {
        setLocalMetadata(null);
        // We only clear active market if user explicitly resets while viewing it? 
        // Or keep it simple: reset clears local view. If streaming, it continues but user sees search.
        // Let's assume reset clears local state to allow new search.
        // It DOES NOT stop streaming (unless we add a stop function).
        setSelectedOutcomes(new Set());
        setStatus('');
        setInput('');
    };

    return (
        <div className="flex flex-col gap-6 p-6 h-full overflow-y-auto">
            <div className="flex flex-col gap-2">
                <h2 className="text-2xl font-bold tracking-tight">Polymarket Scraper</h2>
                <p className="text-muted-foreground text-sm">Download historical price data for Polymarket tokens.</p>
            </div>

            <div className="max-w-3xl flex flex-col gap-6">

                {/* Step 1: Search */}
                {!displayMetadata && (
                    <div className="grid gap-4 p-4 border rounded-lg bg-card text-card-foreground shadow-sm">
                        <div className="grid gap-2">
                            <label className="text-sm font-medium">Market URL, Slug, or ID</label>
                            <input
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="https://polymarket.com/event/portugal-presidential-election"
                                className="flex h-10 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                            />
                            <p className="text-xs text-muted-foreground">Paste the full URL or Slug to find the market options.</p>
                        </div>
                        <button
                            onClick={handleFindMarket}
                            disabled={loading}
                            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2 w-fit"
                        >
                            {loading ? "Searching..." : "Find Market"}
                        </button>
                    </div>
                )}

                {/* Step 2: Selection & Config */}
                {displayMetadata && (
                    <div className="flex flex-col gap-6">
                        <div className="border rounded-lg p-4 bg-muted/20">
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h3 className="font-semibold text-lg">{displayMetadata.title}</h3>
                                    <p className="text-sm text-muted-foreground">{selectedOutcomes.size} selected</p>
                                </div>
                                <button onClick={reset} className="text-xs text-muted-foreground hover:text-foreground">
                                    Change Market
                                </button>
                                {isStreaming && <span className="text-xs text-green-500 animate-pulse ml-2">● LIVE</span>}
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[300px] overflow-y-auto p-1">
                                {displayMetadata.outcomes.map(outcome => (
                                    <label key={outcome.id} className="flex items-center justify-between p-2 rounded hover:bg-muted cursor-pointer border border-transparent hover:border-border">
                                        <div className="flex items-center space-x-2">
                                            <input
                                                type="checkbox"
                                                checked={selectedOutcomes.has(outcome.id)}
                                                onChange={() => toggleOutcome(outcome.id)}
                                                className="h-4 w-4 rounded border-gray-300"
                                            />
                                            <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                                                {outcome.name}
                                            </span>
                                        </div>
                                        {livePrices[outcome.id] !== undefined && (
                                            <span className="text-sm font-mono text-green-400">
                                                ${livePrices[outcome.id].toFixed(3)}
                                            </span>
                                        )}
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="grid gap-2">
                                <label className="text-sm font-medium">Frequency</label>
                                <select
                                    value={frequency}
                                    onChange={(e: any) => setFrequency(e.target.value)}
                                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    <option value="Minutely">Minutely</option>
                                    <option value="Hourly">Hourly</option>
                                    <option value="Daily">Daily</option>
                                    <option value="Weekly">Weekly</option>
                                    <option value="Monthly">Monthly</option>
                                </select>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="grid gap-2">
                                <label className="text-sm font-medium">Start Date (Optional)</label>
                                <input
                                    type="date"
                                    value={startDate}
                                    onChange={(e: any) => setStartDate(e.target.value)}
                                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                />
                            </div>
                            <div className="grid gap-2">
                                <label className="text-sm font-medium">End Date (Optional)</label>
                                <input
                                    type="date"
                                    value={endDate}
                                    onChange={(e: any) => setEndDate(e.target.value)}
                                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                />
                            </div>
                        </div>

                        <button
                            onClick={handleScrape}
                            disabled={loading || selectedOutcomes.size === 0}
                            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2 w-fit"
                        >
                            {loading ? (
                                "Scraping..."
                            ) : (
                                <>
                                    <Save className="mr-2 h-4 w-4" />
                                    Download CSV ({selectedOutcomes.size} selected)
                                </>
                            )}
                        </button>

                        <div className="flex gap-2">
                            <button
                                onClick={handleStream}
                                disabled={isStreaming}
                                className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80 h-9 px-4 py-2 w-fit"
                            >
                                <Activity className="mr-2 h-4 w-4" />
                                Stream Live Prices
                            </button>
                            {isStreaming && (
                                <button
                                    onClick={stopStream}
                                    className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring bg-rose-600 hover:bg-rose-500 text-white shadow-sm h-9 px-4 py-2 w-fit"
                                >
                                    Stop Stream
                                </button>
                            )}
                        </div>
                    </div>
                )}

                {status && (
                    <div className={`text-sm ${status.startsWith('Error') ? 'text-red-500' : 'text-green-500'}`}>
                        {status}
                    </div>
                )}
            </div>
        </div>
    );
}
