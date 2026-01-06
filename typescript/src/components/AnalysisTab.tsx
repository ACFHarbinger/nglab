import { useState, useEffect } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { readTextFile } from '@tauri-apps/plugin-fs';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import CustomStock from 'highcharts/modules/stock';
import Heatmap from 'highcharts/modules/heatmap';
import Indicators from 'highcharts/indicators/indicators-all';
import Papa from 'papaparse';
import { FileSpreadsheet, Loader2 } from 'lucide-react';

// Load modules
try {
    if (typeof CustomStock === 'function') (CustomStock as any)(Highcharts);
    if (typeof Heatmap === 'function') (Heatmap as any)(Highcharts);
    if (typeof Indicators === 'function') (Indicators as any)(Highcharts);
} catch (e) {
    console.error('Highcharts module loading failed:', e);
}

interface CsvRow {
    [key: string]: string | number;
}

function AnalysisTab() {
    const [chartOptions, setChartOptions] = useState<Highcharts.Options | null>(null);
    const [fileName, setFileName] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [rawData, setRawData] = useState<CsvRow[]>([]);
    const [chartType, setChartType] = useState<'price' | 'spread' | 'volatility' | 'candlestick' | 'heatmap'>('price');
    const [allSeries, setAllSeries] = useState<string[]>([]);
    const [selectedSeries, setSelectedSeries] = useState<string[]>([]);
    const [spreadPair, setSpreadPair] = useState<[string, string]>(['', '']);
    const [primaryCandidate, setPrimaryCandidate] = useState<string>('');
    const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | 'All'>('All');
    const [indicators, setIndicators] = useState({ sma: false, macd: false, rsi: false });

    // Re-generate chart when type or data changes
    useEffect(() => {
        if (rawData.length > 0 && !loading && !error) {
            createChart(rawData, chartType);
        } else if (rawData.length === 0 && chartOptions) {
            setChartOptions(null);
        }
    }, [rawData, chartType, loading, error, selectedSeries, spreadPair, primaryCandidate, timeframe, indicators]);


    const handleOpenFile = async () => {
        try {
            const selected = await open({
                multiple: false,
                filters: [{
                    name: 'CSV',
                    extensions: ['csv']
                }]
            });

            if (selected) {
                setLoading(true);
                setError('');
                setChartOptions(null); // Clear previous chart
                setRawData([]); // Clear previous raw data
                setAllSeries([]);
                setSelectedSeries([]);
                setSpreadPair(['', '']);
                const path = selected as string;
                // Extract filename for display
                const name = path.split(/[/\\]/).pop() || path;
                setFileName(name);

                await processFile(path);
            }
        } catch (err: any) {
            console.error(err);
            setError('Failed to open file: ' + err.message);
            setLoading(false);
        }
    };

    const processFile = async (path: string) => {
        try {
            const content = await readTextFile(path);

            Papa.parse(content, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.warn("CSV Errors:", results.errors);
                    }
                    // Pre-process timestamps once
                    // Find date key once
                    const raw = results.data as CsvRow[];
                    if (raw.length === 0) { setLoading(false); return; }
                    const firstRowKeys = Object.keys(raw[0]);
                    const dateIdxKey = firstRowKeys.find(k => k.toLowerCase().includes('date') || k.toLowerCase().includes('time'));

                    // Pre-process timestamps once
                    const processed = raw.map(row => {
                        const timestampCol = row['Timestamp (UTC)'];
                        let timestamp: number;
                        if (typeof timestampCol === 'number') {
                            timestamp = timestampCol < 10000000000 ? timestampCol * 1000 : timestampCol;
                        } else if (dateIdxKey && row[dateIdxKey]) {
                            // Fast parse for common format or use Date
                            const dStr = row[dateIdxKey] as string;
                            timestamp = new Date(dStr.replace(/-/g, '/')).getTime();
                        } else {
                            timestamp = NaN;
                        }
                        return { ...row, _ts: timestamp };
                    }).filter(r => !isNaN(r._ts as number)).sort((a: any, b: any) => a._ts - b._ts);

                    // Init filters
                    if (processed.length > 0) {
                        const keys = Object.keys(processed[0]);
                        const dateKey = keys.find(k => k.toLowerCase().includes('date') || k.toLowerCase().includes('time'));
                        const candidates = keys.filter(k =>
                            k !== dateKey &&
                            k !== 'Timestamp (UTC)' &&
                            k !== '_ts' &&
                            !k.startsWith('Price_')
                        );
                        setAllSeries(candidates);

                        const lastRow = processed[processed.length - 1] as any;
                        const sorted = [...candidates].sort((a, b) => {
                            const pa = lastRow[a] as number || 0;
                            const pb = lastRow[b] as number || 0;
                            return pb - pa;
                        });

                        setSelectedSeries(sorted.slice(0, 5));
                        setSpreadPair([sorted[0] || '', sorted[1] || '']);
                        setPrimaryCandidate(sorted[0] || '');
                    }

                    setRawData(processed);
                    setLoading(false);
                },
                error: (err: any) => {
                    setError('CSV Parse Error: ' + err.message);
                    setLoading(false);
                }
            });
        } catch (err: any) {
            setError('Read Error: ' + err.message);
            setLoading(false);
        }
    };

    const handleTypeChange = (type: 'price' | 'spread' | 'volatility' | 'candlestick' | 'heatmap') => {
        setChartOptions(null);
        setChartType(type);
    };

    const createChart = (data: CsvRow[], type: 'price' | 'spread' | 'volatility' | 'candlestick' | 'heatmap' = chartType) => {
        if (data.length === 0) {
            setError('CSV is empty');
            return;
        }

        // Apply manual timeframe filter
        let filteredData = data;
        if (timeframe !== 'All') {
            const lastRow = data[data.length - 1];
            const lastTs = lastRow?._ts as number || Date.now();
            let lookback = 0;
            if (timeframe === '1D') lookback = 24 * 60 * 60 * 1000;
            else if (timeframe === '1W') lookback = 7 * 24 * 60 * 60 * 1000;
            else if (timeframe === '1M') lookback = 30 * 24 * 60 * 60 * 1000;
            filteredData = data.filter(r => (r._ts as number) >= lastTs - lookback);
        }

        if (filteredData.length === 0) {
            setError('No data in selected timeframe.');
            return;
        }

        const keys = Object.keys(filteredData[0]);
        const dateKey = keys.find(k => k.toLowerCase().includes('date') || k.toLowerCase().includes('time'));
        if (!dateKey) {
            setError('Could not find a Date/Timestamp column.');
            return;
        }

        let keysToProcess: string[] = [];
        if (type === 'spread') {
            keysToProcess = [spreadPair[0], spreadPair[1]].filter(Boolean);
        } else if (type === 'candlestick') {
            keysToProcess = [primaryCandidate].filter(Boolean);
        } else {
            keysToProcess = selectedSeries;
        }

        const seriesData: any[] = [];
        const seriesMap = new Map<string, number[][]>();

        keysToProcess.forEach(seriesName => {
            const points: number[][] = [];
            for (let i = 0; i < filteredData.length; i++) {
                const row = filteredData[i] as any;
                const val = row[seriesName];
                if (typeof val === 'number') {
                    points.push([row._ts as number, val]);
                }
            }
            seriesMap.set(seriesName, points);
        });

        const calculateOHLC = (points: number[][]) => {
            const dailyMap = new Map<number, { open: number, high: number, low: number, close: number, time: number }>();
            points.forEach(([ts, price]) => {
                const date = new Date(ts);
                date.setUTCHours(0, 0, 0, 0);
                const dayTs = date.getTime();
                if (!dailyMap.has(dayTs)) {
                    dailyMap.set(dayTs, { open: price, high: price, low: price, close: price, time: dayTs });
                } else {
                    const day = dailyMap.get(dayTs)!;
                    day.high = Math.max(day.high, price);
                    day.low = Math.min(day.low, price);
                    day.close = price;
                }
            });
            return Array.from(dailyMap.values()).sort((a, b) => a.time - b.time).map(d => [d.time, d.open, d.high, d.low, d.close]);
        };

        let chartTitle = '';

        if (type === 'price') {
            chartTitle = 'Price History';
            seriesMap.forEach((points, name) => {
                seriesData.push({ name, data: points, tooltip: { valueDecimals: 4 } });
            });
        }
        else if (type === 'candlestick') {
            chartTitle = `Daily Candles: ${primaryCandidate}`;
            if (primaryCandidate && seriesMap.has(primaryCandidate)) {
                const points = seriesMap.get(primaryCandidate)!;
                seriesData.push({
                    type: 'candlestick',
                    name: primaryCandidate,
                    data: calculateOHLC(points),
                    color: '#f43f5e',
                    upColor: '#10b981',
                    lineColor: '#f43f5e',
                    upLineColor: '#10b981',
                    tooltip: { valueDecimals: 4 }
                });
            } else {
                setError('Please select a candidate for OHLC analysis.');
            }
        }
        else if (type === 'spread') {
            chartTitle = `Spread: ${spreadPair[0]} vs ${spreadPair[1]}`;
            if (spreadPair[0] && spreadPair[1] && spreadPair[0] !== spreadPair[1]) {
                const s1 = seriesMap.get(spreadPair[0]);
                const s2 = seriesMap.get(spreadPair[1]);
                if (s1 && s2) {
                    const spreadPoints = s1.map((p1, i) => {
                        const p2 = s2[i];
                        if (!p2) return null;
                        return [p1[0], Math.abs(p1[1] - p2[1])];
                    }).filter(p => p);
                    seriesData.push({
                        name: `Spread (${spreadPair[0]} vs ${spreadPair[1]})`,
                        data: spreadPoints,
                        color: '#f59e0b',
                        tooltip: { valueDecimals: 4 }
                    });
                }
            }
        }
        else if (type === 'volatility') {
            chartTitle = 'Volatility (20-period)';
            const window = 20;
            seriesMap.forEach((points, name) => {
                const volPoints = [];
                if (points.length < window) return;
                for (let i = window; i < points.length; i++) {
                    const slice = points.slice(i - window, i).map(p => p[1]);
                    const mean = slice.reduce((a, b) => a + b, 0) / window;
                    const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / window;
                    const stdDev = Math.sqrt(variance);
                    volPoints.push([points[i][0], stdDev]);
                }
                seriesData.push({ name: `${name} Volatility`, data: volPoints, tooltip: { valueDecimals: 5 } });
            });
        }
        else if (type === 'heatmap') {
            chartTitle = 'Returns Heatmap (%)';
            const numBuckets = 20;
            const validSeries = Array.from(seriesMap.values()).filter(pts => pts.length > 0);
            if (validSeries.length === 0) return;

            const startTime = Math.min(...validSeries.map(pts => pts[0][0]));
            const endTime = Math.max(...validSeries.map(pts => pts[pts.length - 1][0]));
            if (!isFinite(startTime) || !isFinite(endTime)) return;

            const bucketSize = (endTime - startTime) / numBuckets;

            const heatmapData: any[] = [];
            const yCategories: string[] = [];

            let currentY = 0;
            selectedSeries.forEach((name) => {
                const points = seriesMap.get(name);
                if (!points) return;
                yCategories.push(name);

                const buckets = new Array(numBuckets).fill(null).map(() => [] as number[]);
                points.forEach(([ts, val]) => {
                    const idx = Math.floor((ts - startTime) / (bucketSize || 1));
                    if (idx >= 0 && idx < numBuckets) {
                        buckets[idx].push(val);
                    } else if (idx === numBuckets) {
                        buckets[numBuckets - 1].push(val);
                    }
                });

                buckets.forEach((bucketPrices, xIdx) => {
                    if (bucketPrices.length >= 2) {
                        const startPrice = bucketPrices[0];
                        const endPrice = bucketPrices[bucketPrices.length - 1];
                        const change = ((endPrice / startPrice) - 1) * 100;
                        const bStart = startTime + xIdx * bucketSize;
                        heatmapData.push([bStart, currentY, parseFloat(change.toFixed(2))]);
                    } else {
                        const bStart = startTime + xIdx * bucketSize;
                        heatmapData.push([bStart, currentY, 0]);
                    }
                });
                currentY++;
            });

            setChartOptions({
                chart: {
                    type: 'heatmap',
                    backgroundColor: '#0f172a',
                    style: { fontFamily: 'Inter, sans-serif' },
                    marginTop: 60,
                    marginBottom: 100,
                    events: {
                        load: function () {
                            const chart = this;
                            setTimeout(() => {
                                chart.reflow();
                            }, 100);
                        }
                    }
                },
                title: { text: chartTitle, style: { color: '#e2e8f0', fontSize: '18px' }, align: 'center' },
                xAxis: {
                    type: 'datetime',
                    labels: { style: { color: '#94a3b8', fontSize: '11px' }, rotation: -45 },
                    gridLineWidth: 0,
                    lineWidth: 0,
                    tickWidth: 0,
                    ordinal: false
                },
                yAxis: {
                    categories: yCategories,
                    title: { text: undefined },
                    labels: {
                        style: { color: '#94a3b8', fontSize: '11px', whiteSpace: 'nowrap' },
                        align: 'right',
                        reserveSpace: true
                    },
                    gridLineWidth: 0,
                    lineWidth: 0,
                    reversed: true,
                    opposite: false
                },
                colorAxis: {
                    stops: [[0, '#f43f5e'], [0.5, '#1e293b'], [1, '#10b981']],
                    min: -5,
                    max: 5,
                    labels: { style: { color: '#94a3b8' } }
                } as any,
                legend: {
                    align: 'right',
                    layout: 'vertical',
                    verticalAlign: 'middle',
                    symbolHeight: 250,
                    itemStyle: { color: '#94a3b8' }
                },
                tooltip: {
                    formatter: function (this: any) {
                        const p = this.point;
                        return `<b>${yCategories[p.y]}</b><br/>Time: ${Highcharts.dateFormat('%b %d, %Y', p.x)}<br/>Change: <b>${p.value}%</b>`;
                    }
                },
                series: [{
                    type: 'heatmap',
                    name: 'Returns',
                    data: heatmapData,
                    colsize: bucketSize,
                    borderWidth: 1,
                    borderColor: '#0f172a',
                    dataLabels: {
                        enabled: true,
                        color: '#ffffff',
                        style: { textOutline: 'none', fontSize: '9px' },
                        formatter: function (this: any) {
                            return this.point.value !== 0 ? this.point.value + '%' : '';
                        }
                    }
                } as any],
                navigator: {
                    enabled: true,
                    maskFill: 'rgba(99, 102, 241, 0.2)',
                    series: { color: '#6366f1', lineColor: '#6366f1' },
                    xAxis: { gridLineColor: '#334155', labels: { style: { color: '#94a3b8' } } }
                },
                rangeSelector: {
                    enabled: true,
                    inputEnabled: false,
                    selected: 4,
                    buttons: [],
                    buttonTheme: { visibility: 'hidden' },
                    labelStyle: { visibility: 'hidden' }
                },
                scrollbar: { enabled: true, barBackgroundColor: '#334155', trackBackgroundColor: '#1e293b' },
                credits: { enabled: false }
            });
            return;
        }

        const yAxisConfig: any[] = [];
        if (type === 'price' || type === 'candlestick') {
            let mainHeight = 100;
            const panes: any[] = [];
            if (indicators.macd) panes.push('macd');
            if (indicators.rsi) panes.push('rsi');

            if (panes.length > 0) {
                mainHeight = 60;
                if (panes.length === 2) mainHeight = 50;
            }

            yAxisConfig.push({
                height: `${mainHeight}%`,
                gridLineColor: '#334155',
                labels: { style: { color: '#94a3b8' } },
                plotLines: [{ value: 0, width: 2, color: '#334155' }],
                resize: { enabled: true }
            });

            let currentTop = mainHeight + 5;
            const paneHeight = (100 - currentTop) / panes.length;

            panes.forEach((p) => {
                yAxisConfig.push({
                    top: `${currentTop}%`,
                    height: `${paneHeight}%`,
                    offset: 0,
                    gridLineColor: '#334155',
                    labels: { style: { color: '#94a3b8' } },
                    title: { text: p.toUpperCase(), style: { color: '#94a3b8' } }
                });
                currentTop += paneHeight;
            });

            if (seriesData.length > 0) {
                seriesData[0].id = 'main-series';

                if (indicators.sma) {
                    seriesData.push({
                        type: 'sma',
                        linkedTo: 'main-series',
                        zIndex: 1,
                        marker: { enabled: false },
                        params: { period: 14 }
                    });
                }
                if (indicators.macd) {
                    seriesData.push({
                        type: 'macd',
                        linkedTo: 'main-series',
                        yAxis: 1,
                        zIndex: 0,
                        params: { shortPeriod: 12, longPeriod: 26, signalPeriod: 9, period: 26 }
                    });
                }
                if (indicators.rsi) {
                    seriesData.push({
                        type: 'rsi',
                        linkedTo: 'main-series',
                        yAxis: indicators.macd ? 2 : 1,
                        zIndex: 0,
                        params: { period: 14 }
                    });
                }
            }
        } else {
            yAxisConfig.push({ gridLineColor: '#334155', labels: { style: { color: '#94a3b8' } }, plotLines: [{ value: 0, width: 2, color: '#334155' }] });
        }

        setChartOptions({
            chart: { backgroundColor: '#0f172a', style: { fontFamily: 'Inter, sans-serif' } },
            colors: ['#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#8b5cf6', '#ec4899'],
            title: { text: chartTitle, style: { color: '#e2e8f0', fontSize: '16px' }, align: 'left' },
            navigator: { enabled: true, maskFill: 'rgba(99, 102, 241, 0.2)', series: { color: '#6366f1', lineColor: '#6366f1' }, xAxis: { gridLineColor: '#334155' } },
            scrollbar: { enabled: false },
            rangeSelector: {
                enabled: true,
                buttonTheme: {
                    fill: '#1e293b', stroke: '#334155', 'stroke-width': 1,
                    style: { color: '#94a3b8', fontWeight: 'bold' },
                    states: { hover: { fill: '#334155', style: { color: '#f8fafc' } }, select: { fill: '#4f46e5', style: { color: 'white' } } }
                },
                inputStyle: { color: '#94a3b8', backgroundColor: '#1e293b' },
                labelStyle: { color: '#94a3b8' },
                selected: 4,
                buttons: [{ type: 'day', count: 1, text: '1D' }, { type: 'week', count: 1, text: '1W' }, { type: 'month', count: 1, text: '1M' }, { type: 'all', text: 'All' }]
            },
            xAxis: { type: 'datetime', gridLineColor: '#334155', labels: { style: { color: '#94a3b8' } }, lineColor: '#334155', tickColor: '#334155' },
            yAxis: yAxisConfig,
            plotOptions: { series: { showInNavigator: true } },
            tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.9)', style: { color: '#f8fafc' }, borderWidth: 0, shadow: false },
            credits: { enabled: false },
            series: seriesData
        });
    };

    return (
        <div className="flex flex-col h-full bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30">
            <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/20">
                <div className="flex items-center gap-6">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">Data Analysis</h2>
                        <p className="text-slate-400 text-sm">Visualize historical price data.</p>
                    </div>
                    <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
                        {(['price', 'candlestick', 'heatmap', 'spread', 'volatility'] as const).map(t => (
                            <button
                                key={t}
                                onClick={() => handleTypeChange(t)}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${chartType === t ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                            >
                                {t.charAt(0).toUpperCase() + t.slice(1)}
                            </button>
                        ))}
                    </div>
                    <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800 ml-4">
                        {(['1D', '1W', '1M', 'All'] as const).map(tf => (
                            <button
                                key={tf}
                                onClick={() => setTimeframe(tf)}
                                className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${timeframe === tf ? 'bg-indigo-600 text-white shadow' : 'text-slate-500 hover:text-slate-200'}`}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>
                </div>
                <button
                    onClick={handleOpenFile}
                    disabled={loading}
                    className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
                >
                    {loading ? <Loader2 className="animate-spin" size={16} /> : <FileSpreadsheet size={16} />}
                    {fileName ? 'Open Another File' : 'Open CSV File'}
                </button>
            </div>

            {error && (
                <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-lg text-sm m-4">
                    {error}
                </div>
            )}

            <div className="flex-1 flex overflow-hidden">
                <div className="flex-1 bg-slate-950 relative border-r border-slate-800">
                    {!chartOptions && !loading && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-4">
                            <FileSpreadsheet size={48} className="opacity-20" />
                            <p>No data loaded. Open a CSV file to begin analysis.</p>
                        </div>
                    )}

                    {loading && (
                        <div className="absolute inset-0 flex items-center justify-center text-indigo-400 gap-2 bg-slate-950/50 z-10">
                            <Loader2 className="animate-spin" />
                            <span>Processing data...</span>
                        </div>
                    )}

                    {chartOptions && (
                        <div className="h-full w-full">
                            <HighchartsReact
                                key={`${chartType}-${fileName}`}
                                highcharts={Highcharts}
                                constructorType={'stockChart'} // Always use stockChart to support navigator
                                options={chartOptions}
                                containerProps={{ style: { height: '100%', width: '100%' } }}
                            />
                        </div>
                    )}
                </div>

                {allSeries.length > 0 && (
                    <div className="w-64 bg-slate-900/50 flex flex-col border-l border-slate-800">
                        <div className="p-4 border-b border-slate-800 bg-slate-900/80">
                            <h3 className="font-semibold text-sm">Configuration</h3>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 space-y-4">
                            {chartType === 'candlestick' ? (
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400">Select Candidate</label>
                                        <select
                                            value={primaryCandidate}
                                            onChange={(e) => setPrimaryCandidate(e.target.value)}
                                            className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                        >
                                            {allSeries.map(s => (
                                                <option key={s} value={s}>{s}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="pt-4 border-t border-slate-800">
                                        <h4 className="text-xs font-medium text-slate-400 mb-2">Indicators</h4>
                                        <div className="space-y-2">
                                            {(['sma', 'macd', 'rsi'] as const).map(ind => (
                                                <label key={ind} className="flex items-center gap-2 cursor-pointer group">
                                                    <input
                                                        type="checkbox"
                                                        checked={indicators[ind]}
                                                        onChange={(e) => setIndicators(prev => ({ ...prev, [ind]: e.target.checked }))}
                                                        className="w-3.5 h-3.5 rounded border-slate-700 bg-slate-800 text-indigo-600 focus:ring-indigo-500"
                                                    />
                                                    <span className="text-xs font-medium text-slate-300 group-hover:text-white uppercase">{ind}</span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ) : chartType === 'spread' ? (
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400">Primary (Asset 1)</label>
                                        <select
                                            value={spreadPair[0]}
                                            onChange={(e) => setSpreadPair([e.target.value, spreadPair[1]])}
                                            className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                        >
                                            <option value="">Select Asset</option>
                                            {allSeries.map(s => (
                                                <option key={s} value={s}>{s}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400">Secondary (Asset 2)</label>
                                        <select
                                            value={spreadPair[1]}
                                            onChange={(e) => setSpreadPair([spreadPair[0], e.target.value])}
                                            className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                        >
                                            <option value="">Select Asset</option>
                                            {allSeries.map(s => (
                                                <option key={s} value={s}>{s}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <p className="text-[10px] text-slate-500 leading-tight">
                                        Calculates the absolute difference between the two selected candidates.
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <label className="text-xs font-medium text-slate-400">Visible Series</label>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={() => setSelectedSeries(allSeries.slice(0, 5))}
                                                className="text-[10px] text-indigo-400 hover:text-indigo-300 font-semibold uppercase tracking-wider"
                                            >Top 5</button>
                                            <button
                                                onClick={() => setSelectedSeries(allSeries)}
                                                className="text-[10px] text-emerald-400 hover:text-emerald-300 font-semibold uppercase tracking-wider"
                                            >All</button>
                                            <button
                                                onClick={() => setSelectedSeries([])}
                                                className="text-[10px] text-rose-400 hover:text-rose-300 font-semibold uppercase tracking-wider"
                                            >None</button>
                                        </div>
                                    </div>
                                    <div className="space-y-1">
                                        {allSeries.map(series => (
                                            <label key={series} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-slate-800 cursor-pointer transition-colors group">
                                                <input
                                                    type="checkbox"
                                                    checked={selectedSeries.includes(series)}
                                                    onChange={(e) => {
                                                        if (e.target.checked) setSelectedSeries([...selectedSeries, series]);
                                                        else setSelectedSeries(selectedSeries.filter(s => s !== series));
                                                    }}
                                                    className="w-3.5 h-3.5 rounded border-slate-700 bg-slate-800 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-slate-900"
                                                />
                                                <span className="text-xs font-medium text-slate-300 group-hover:text-white truncate">{series}</span>
                                            </label>
                                        ))}
                                    </div>
                                    {chartType === 'price' && (
                                        <div className="pt-4 border-t border-slate-800 mt-4">
                                            <h4 className="text-xs font-medium text-slate-400 mb-2">Indicators</h4>
                                            <div className="space-y-2">
                                                {(['sma', 'macd', 'rsi'] as const).map(ind => (
                                                    <label key={ind} className="flex items-center gap-2 cursor-pointer group">
                                                        <input
                                                            type="checkbox"
                                                            checked={indicators[ind]}
                                                            onChange={(e) => setIndicators(prev => ({ ...prev, [ind]: e.target.checked }))}
                                                            className="w-3.5 h-3.5 rounded border-slate-700 bg-slate-800 text-indigo-600 focus:ring-indigo-500"
                                                        />
                                                        <span className="text-xs font-medium text-slate-300 group-hover:text-white uppercase">{ind}</span>
                                                    </label>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default AnalysisTab;
