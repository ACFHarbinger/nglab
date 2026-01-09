import { useState, useEffect, useRef } from 'react';
import { Brain, Loader2, LineChart as LineChartIcon, Activity, FileSpreadsheet } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { createChart, ColorType, IChartApi, ISeriesApi, LineSeries } from 'lightweight-charts';
import { open } from '@tauri-apps/plugin-dialog';
import { readTextFile } from '@tauri-apps/plugin-fs';
import Papa from 'papaparse';

type ArimaResult = {
    path: number[];
    used_seed?: number;
};

type GarchResult = {
    returns: number[];
    volatility: number[];
};

export default function PredictionTab() {
    const [activeModel, setActiveModel] = useState<'arima' | 'garch'>('arima');

    // Data State
    const [rawData, setRawData] = useState<any[]>([]);
    const [fileName, setFileName] = useState('');
    const [columns, setColumns] = useState<string[]>([]);
    const [selectedColumn, setSelectedColumn] = useState('');

    // ARIMA State
    const [ar, setAr] = useState('0.5, -0.2');
    const [ma, setMa] = useState('0.3');
    const [d, setD] = useState(1);
    const [arimaSteps, setArimaSteps] = useState(200);
    const [arimaSigma, setArimaSigma] = useState(0.01);
    const [isArimaLoading, setIsArimaLoading] = useState(false);

    // GARCH State
    const [omega, setOmega] = useState(0.00001);
    const [alpha, setAlpha] = useState('0.1');
    const [beta, setBeta] = useState('0.8');
    const [garchSteps, setGarchSteps] = useState(200);
    const [isGarchLoading, setIsGarchLoading] = useState(false);

    // Common State
    const [seed, setSeed] = useState<number | ''>('');

    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const pastSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
    const predictionSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#0f172a' },
                textColor: '#94a3b8',
            },
            grid: {
                vertLines: { color: '#1e293b' },
                horzLines: { color: '#1e293b' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 400,
        });

        const pastSeries = chart.addSeries(LineSeries, {
            color: '#94a3b8',
            lineWidth: 2,
            title: 'Historical',
        });

        const predictionSeries = chart.addSeries(LineSeries, {
            color: '#6366f1',
            lineWidth: 2,
            title: 'Prediction',
        });

        chartRef.current = chart;
        pastSeriesRef.current = pastSeries;
        predictionSeriesRef.current = predictionSeries;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update chart when selected column or raw data changes
    useEffect(() => {
        if (rawData.length > 0 && selectedColumn && pastSeriesRef.current) {
            try {
                // Determine mode based on first element (rawData is homogeneous from handleOpenFile)
                const hasDates = !isNaN(rawData[0]._ts);

                let data = rawData.map((row, i) => {
                    let time: any;
                    if (hasDates) {
                        // Strict Date Mode
                        time = row._ts / 1000;
                    } else {
                        // Strict Index Mode
                        time = i;
                    }

                    return {
                        time: time,
                        value: parseFloat(row[selectedColumn])
                    };
                })
                    .filter(d => !isNaN(d.value) && d.value !== null && !isNaN(d.time))
                    .sort((a, b) => (a.time as number) - (b.time as number));

                // Deduplicate timestamps (LWC requires strictly ascending unique times)
                const uniqueData = [];
                if (data.length > 0) {
                    uniqueData.push(data[0]);
                    for (let i = 1; i < data.length; i++) {
                        if (data[i].time > data[i - 1].time) {
                            uniqueData.push(data[i]);
                        }
                    }
                }

                if (uniqueData.length > 0) {
                    pastSeriesRef.current.setData(uniqueData);
                    chartRef.current?.timeScale().fitContent();
                }
            } catch (err) {
                console.error("Failed to render chart data:", err);
            }
        }
    }, [rawData, selectedColumn]);

    const handleOpenFile = async () => {
        try {
            const selected = await open({
                multiple: false,
                filters: [{ name: 'CSV', extensions: ['csv'] }]
            });

            if (selected) {
                const path = selected as string;
                setFileName(path.split(/[/\\]/).pop() || path);
                const content = await readTextFile(path);
                Papa.parse(content, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (results) => {
                        const raw = results.data as any[];
                        if (raw.length > 0) {
                            const firstRowKeys = Object.keys(raw[0]);
                            const dateKey = firstRowKeys.find(k => k.toLowerCase().includes('date') || k.toLowerCase().includes('time') || k.toLowerCase() === 'timestamp');

                            // Heuristic Detection of Date Format
                            let isEU = false; // DD/MM/YYYY
                            let isUS = false; // MM/DD/YYYY

                            if (dateKey) {
                                for (const row of raw) {
                                    const val = row[dateKey];
                                    if (typeof val === 'string') {
                                        // Normalize and split date/time
                                        const cleanVal = val.trim().replace('T', ' ');
                                        const datePart = cleanVal.split(' ')[0];

                                        // Check for / or - separators
                                        const parts = datePart.split(/[/\-]/);
                                        if (parts.length === 3) {
                                            const p0 = parseInt(parts[0]);
                                            const p1 = parseInt(parts[1]);

                                            // Ensure p0/p1 are numbers
                                            if (!isNaN(p0) && !isNaN(p1)) {
                                                if (p0 > 12) isEU = true;
                                                if (p1 > 12) isUS = true;
                                            }
                                        }
                                    }
                                    if (isEU || isUS) break; // Found a decisive row
                                }
                            }

                            // Start processing with detected format preference
                            let processed = raw.map(row => {
                                let timestamp: number = NaN;

                                if (dateKey && row[dateKey]) {
                                    const val = row[dateKey];
                                    if (typeof val === 'number') {
                                        timestamp = val < 10000000000 ? val * 1000 : val;
                                    } else {
                                        const dStr = String(val).trim().replace('T', ' ');
                                        const [datePart, ...timeParts] = dStr.split(' ');
                                        const timePart = timeParts.join(' ');

                                        // Try manual parse if format detected
                                        const parts = datePart.split(/[/\-]/);
                                        if (parts.length === 3) {
                                            // Check YYYY-MM-DD (ISO) first - often parts[0] is year
                                            if (parts[0].length === 4) {
                                                timestamp = new Date(dStr).getTime();
                                            }
                                            else {
                                                let dateString = datePart; // Start with just the date part
                                                if (isEU) {
                                                    // Force DD/MM/YYYY -> YYYY/MM/DD or MM/DD/YYYY
                                                    // Construct YYYY/MM/DD which is robust
                                                    dateString = `${parts[2]}/${parts[1]}/${parts[0]}`;
                                                }
                                                else if (isUS) {
                                                    // Force MM/DD/YYYY
                                                    dateString = `${parts[2]}/${parts[0]}/${parts[1]}`;
                                                }

                                                if (timePart) {
                                                    dateString += ' ' + timePart;
                                                }
                                                timestamp = new Date(dateString).getTime();
                                            }
                                        } else {
                                            // Fallback
                                            timestamp = new Date(dStr).getTime();
                                        }

                                        // Final fallback if manual failed
                                        if (isNaN(timestamp)) {
                                            timestamp = new Date(dStr).getTime();
                                        }
                                    }
                                }
                                return { ...row, _ts: timestamp };
                            });

                            // Check if we found valid dates
                            const validDateCount = processed.filter(r => !isNaN(r._ts)).length;
                            const hasValidDates = validDateCount > 0;

                            if (hasValidDates) {
                                // DATE MODE: Filter out rows with invalid dates to prevent 1970 (0) artifacts
                                processed = processed.filter(r => !isNaN(r._ts));
                                processed.sort((a, b) => a._ts - b._ts);
                            } else {
                                // INDEX MODE: Ensure _ts is NaN so we fall back continuously
                                processed = processed.map(r => ({ ...r, _ts: NaN }));
                            }

                            setRawData(processed);
                            const cols = firstRowKeys.filter(k => k !== dateKey && k !== '_ts');
                            setColumns(cols);
                            const defaultCol = cols.find(c => c.toLowerCase().includes('price') || c.toLowerCase().includes('close')) || cols[0];
                            setSelectedColumn(defaultCol);
                        }
                    }
                });
            }
        } catch (err) {
            console.error(err);
        }
    };

    const runArima = async () => {
        setIsArimaLoading(true);
        try {
            const arCoeffs = ar.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            const maCoeffs = ma.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));

            let dataInput = null;
            let lastTimeIdx = -1;

            if (rawData.length > 0 && selectedColumn) {
                const validPoints = rawData.map((row, i) => ({
                    idx: i,
                    val: parseFloat(row[selectedColumn])
                })).filter(d => !isNaN(d.val));

                if (validPoints.length > 0) {
                    dataInput = validPoints.map(d => d.val);
                    lastTimeIdx = validPoints[validPoints.length - 1].idx;
                }
            }

            const result = await invoke<ArimaResult>('predict_arima', {
                params: {
                    ar: arCoeffs,
                    ma: maCoeffs,
                    d,
                    steps: arimaSteps,
                    sigma: arimaSigma,
                    seed: seed === '' ? null : seed,
                    data: dataInput
                }
            });

            if (predictionSeriesRef.current) {
                console.log("ARIMA used seed:", result.used_seed);

                // Calculate time interval
                let interval = 86400; // Default 1 day in seconds
                let lastTime = lastTimeIdx >= 0 && rawData[lastTimeIdx]._ts
                    ? rawData[lastTimeIdx]._ts / 1000
                    : lastTimeIdx; // Fallback to index if no date

                // If we have at least 2 points, try to infer interval
                if (dataInput && dataInput.length >= 2 && !isNaN(rawData[0]._ts)) {
                    // Find the typical difference between the last few points
                    // or just take the last diff
                    // But we used filtered data for input, need filtered timestamps too
                    // Let's reconstruct valid timestamps
                    const validTimes = rawData
                        .filter(row => !isNaN(parseFloat(row[selectedColumn])))
                        .map(row => row._ts / 1000)
                        .filter(t => !isNaN(t));

                    if (validTimes.length >= 2) {
                        const last = validTimes[validTimes.length - 1];
                        const prev = validTimes[validTimes.length - 2];
                        interval = last - prev;
                        lastTime = last;
                    }
                }

                // If using indices (lastTime < 1e9), interval should be 1
                if (lastTime < 100000000) {
                    interval = 1;
                }

                let pathData = result.path.map((v, i) => ({
                    time: (lastTime + (interval * (i + 1))) as any,
                    value: v
                }));

                if (dataInput && dataInput.length > 0) {
                    const lastHistory = dataInput[dataInput.length - 1];
                    // Stitch: Add the point at lastTime to connect the lines
                    pathData = [{ time: lastTime as any, value: lastHistory }, ...pathData];
                }

                predictionSeriesRef.current.setData(pathData);
                chartRef.current?.timeScale().fitContent();
            }
        } catch (err) {
            console.error(err);
        } finally {
            setIsArimaLoading(false);
        }
    };

    const runGarch = async () => {
        setIsGarchLoading(true);
        try {
            const alphaCoeffs = alpha.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            const betaCoeffs = beta.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));

            let dataInput = null;
            let lastTimeIdx = -1;

            if (rawData.length > 0 && selectedColumn) {
                // For GARCH we usually need returns
                // We calculate returns based on ADJACENT valid price points in rawData
                // This assumes rawData is sorted by time but might have NaNs
                // We'll trust that filtered validPoints are temporally ordered
                const validPoints = rawData.map((row, i) => ({
                    idx: i,
                    val: parseFloat(row[selectedColumn])
                })).filter(d => !isNaN(d.val));

                if (validPoints.length > 1) {
                    dataInput = [];
                    // Returns consume 1 point, so the "time" of the return is usually the time of the *current* price (idx)
                    // or previous? Usually r_t = ln(p_t / p_{t-1}). Time t.
                    // So we track the index of the second price.

                    for (let i = 1; i < validPoints.length; i++) {
                        const p_t = validPoints[i].val;
                        const p_prev = validPoints[i - 1].val;
                        dataInput.push(Math.log(p_t / p_prev));
                    }
                    lastTimeIdx = validPoints[validPoints.length - 1].idx;
                }
            }

            const result = await invoke<GarchResult>('predict_garch', {
                params: {
                    omega,
                    alpha: alphaCoeffs,
                    beta: betaCoeffs,
                    steps: garchSteps,
                    seed: seed === '' ? null : seed,
                    data: dataInput
                }
            });

            if (predictionSeriesRef.current) {
                // Calculate time interval
                let interval = 86400; // Default 1 day
                let lastTime = lastTimeIdx >= 0 && rawData[lastTimeIdx]._ts
                    ? rawData[lastTimeIdx]._ts / 1000
                    : lastTimeIdx;

                if (dataInput && dataInput.length >= 2 && !isNaN(rawData[0]._ts)) {
                    const validTimes = rawData
                        .filter(row => !isNaN(parseFloat(row[selectedColumn])))
                        .map(row => row._ts / 1000)
                        .filter(t => !isNaN(t));

                    if (validTimes.length >= 2) {
                        const last = validTimes[validTimes.length - 1];
                        const prev = validTimes[validTimes.length - 2];
                        interval = last - prev;
                        lastTime = last;
                    }
                }

                if (lastTime < 100000000) {
                    interval = 1;
                }

                const pathData = result.volatility.map((v, i) => ({
                    time: (lastTime + (interval * (i + 1))) as any,
                    value: v
                }));

                predictionSeriesRef.current.setData(pathData);
                chartRef.current?.timeScale().fitContent();
            }
        } catch (err) {
            console.error(err);
        } finally {
            setIsGarchLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-950 text-slate-50 font-sans">
            <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/20">
                <div className="flex items-center gap-6">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">Prediction Lab</h2>
                        <p className="text-slate-400 text-sm">AI, Econometrics & Hybrid Time Series Forecasting.</p>
                    </div>
                    <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
                        <button
                            onClick={() => setActiveModel('arima')}
                            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeModel === 'arima' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                            ARIMA
                        </button>
                        <button
                            onClick={() => setActiveModel('garch')}
                            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeModel === 'garch' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                            GARCH
                        </button>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-indigo-300">
                    <Brain size={18} />
                    <span>Econometric Engine</span>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden">
                <div className="w-[360px] bg-slate-900/40 border-r border-slate-800 p-4 overflow-y-auto space-y-6">
                    {/* Data Loading Section */}
                    <div className="space-y-4">
                        <h3 className="text-xs font-semibold uppercase text-slate-400 flex items-center gap-2">
                            <FileSpreadsheet size={14} /> Historical Data
                        </h3>
                        <div className="space-y-3">
                            <button
                                onClick={handleOpenFile}
                                className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 px-3 py-2 rounded-lg transition-colors border border-slate-700 text-sm"
                            >
                                <FileSpreadsheet size={16} /> {fileName ? fileName : 'Load CSV'}
                            </button>
                            {columns.length > 0 && (
                                <label className="block text-xs text-slate-400">
                                    Target Column
                                    <select
                                        value={selectedColumn}
                                        onChange={e => setSelectedColumn(e.target.value)}
                                        className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                    >
                                        {columns.map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </label>
                            )}
                        </div>
                    </div>

                    <div className="border-t border-slate-800 pt-6">
                        {activeModel === 'arima' ? (
                            <div className="space-y-4">
                                <h3 className="text-xs font-semibold uppercase text-slate-400 flex items-center gap-2">
                                    <LineChartIcon size={14} /> ARIMA Configuration
                                </h3>
                                <div className="space-y-3">
                                    <label className="block text-xs text-slate-400">
                                        AR Coefficients (phi)
                                        <input
                                            type="text"
                                            value={ar}
                                            onChange={e => setAr(e.target.value)}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                            placeholder="0.5, -0.2"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        MA Coefficients (theta)
                                        <input
                                            type="text"
                                            value={ma}
                                            onChange={e => setMa(e.target.value)}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                            placeholder="0.3"
                                        />
                                    </label>
                                    <div className="grid grid-cols-2 gap-3">
                                        <label className="text-xs text-slate-400">
                                            d (Integration)
                                            <input
                                                type="number"
                                                value={d}
                                                onChange={e => setD(parseInt(e.target.value))}
                                                className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                            />
                                        </label>
                                        <label className="text-xs text-slate-400">
                                            Noise Sigma
                                            <input
                                                type="number"
                                                value={arimaSigma}
                                                onChange={e => setArimaSigma(parseFloat(e.target.value))}
                                                step="0.001"
                                                className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                            />
                                        </label>
                                    </div>
                                    <label className="block text-xs text-slate-400">
                                        Steps
                                        <input
                                            type="number"
                                            value={arimaSteps}
                                            onChange={e => setArimaSteps(parseInt(e.target.value))}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Random Seed (Optional)
                                        <input
                                            type="number"
                                            value={seed}
                                            onChange={e => setSeed(e.target.value === '' ? '' : parseInt(e.target.value))}
                                            placeholder="Random"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <button
                                        onClick={runArima}
                                        disabled={isArimaLoading}
                                        className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
                                    >
                                        {isArimaLoading ? <Loader2 className="animate-spin" size={16} /> : 'Generate Path'}
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <h3 className="text-xs font-semibold uppercase text-slate-400 flex items-center gap-2">
                                    <Activity size={14} /> GARCH Configuration
                                </h3>
                                <div className="space-y-3">
                                    <label className="block text-xs text-slate-400">
                                        Omega (Constant)
                                        <input
                                            type="number"
                                            value={omega}
                                            onChange={e => setOmega(parseFloat(e.target.value))}
                                            step="0.000001"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Alpha (ARCH coeffs)
                                        <input
                                            type="text"
                                            value={alpha}
                                            onChange={e => setAlpha(e.target.value)}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Beta (GARCH coeffs)
                                        <input
                                            type="text"
                                            value={beta}
                                            onChange={e => setBeta(e.target.value)}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Steps
                                        <input
                                            type="number"
                                            value={garchSteps}
                                            onChange={e => setGarchSteps(parseInt(e.target.value))}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Random Seed (Optional)
                                        <input
                                            type="number"
                                            value={seed}
                                            onChange={e => setSeed(e.target.value === '' ? '' : parseInt(e.target.value))}
                                            placeholder="Random"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <button
                                        onClick={runGarch}
                                        disabled={isGarchLoading}
                                        className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
                                    >
                                        {isGarchLoading ? <Loader2 className="animate-spin" size={16} /> : 'Generate Volatility'}
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                <div className="flex-1 p-6 flex flex-col gap-6 overflow-hidden">
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col relative">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-semibold text-slate-300">
                                {activeModel === 'arima' ? 'Simulated Price Path (Integrated)' : 'Conditional Volatility Series'}
                            </h3>
                            <div className="flex gap-2">
                                {rawData.length > 0 && (
                                    <div className="flex items-center gap-1.5 text-[10px] text-slate-500 bg-slate-950 px-2 py-1 rounded">
                                        <div className="w-2 h-2 rounded-full bg-slate-500"></div> Historical
                                    </div>
                                )}
                                <div className="flex items-center gap-1.5 text-[10px] text-slate-500 bg-slate-950 px-2 py-1 rounded">
                                    <div className="w-2 h-2 rounded-full bg-indigo-500"></div> Forecast
                                </div>
                            </div>
                        </div>
                        <div ref={chartContainerRef} className="flex-1 w-full" />
                    </div>

                    <div className="h-32 bg-slate-900/30 border border-slate-800 rounded-xl p-4 text-sm text-slate-400 overflow-y-auto">
                        {activeModel === 'arima' ? (
                            <p>
                                ARIMA (AutoRegressive Integrated Moving Average) models can now use <b>historical data</b>.
                                The model initializes its state using your CSV data, ensuring the prediction starts exactly where your data ends.
                            </p>
                        ) : (
                            <p>
                                GARCH models use historical returns to estimate initial conditional variance.
                                This allows for more realistic volatility forecasting based on recent market regimes found in your data.
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
