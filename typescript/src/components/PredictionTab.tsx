import { useState, useEffect, useRef } from 'react';
import { Brain, Loader2, LineChart as LineChartIcon, Activity, FileSpreadsheet } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { createChart, ColorType, IChartApi, ISeriesApi, LineSeries } from 'lightweight-charts';
import { open } from '@tauri-apps/plugin-dialog';
import { readTextFile } from '@tauri-apps/plugin-fs';
import Papa from 'papaparse';
import { prepareChartData } from '../utils/dataHelpers';

type ArimaResult = {
    path: number[];
    used_seed?: number;
};

type GarchResult = {
    returns: number[];
    volatility: number[];
};

type HoltWintersResult = {
    path: number[];
    used_seed?: number;
};

type ProphetResult = {
    times: number[];
    values: number[];
    trend: number[];
    seasonal: number[];
};

export default function PredictionTab() {
    const [activeModel, setActiveModel] = useState<'arima' | 'garch' | 'holt_winters' | 'prophet'>('arima');

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

    // Holt-Winters State
    const [hwAlpha, setHwAlpha] = useState(0.5);
    const [hwBeta, setHwBeta] = useState(0.1);
    const [hwGamma, setHwGamma] = useState(0.1);
    const [hwPeriod, setHwPeriod] = useState(12);
    const [hwSeasonalType, setHwSeasonalType] = useState<'Additive' | 'Multiplicative'>('Additive');
    const [hwSteps, setHwSteps] = useState(50);
    const [hwSigma, setHwSigma] = useState(0.0);
    const [isHwLoading, setIsHwLoading] = useState(false);

    // Prophet State
    const [prophetGrowth, setProphetGrowth] = useState('linear');
    const [prophetYearly, setProphetYearly] = useState(false);
    const [prophetWeekly, setProphetWeekly] = useState(false);
    const [prophetDaily, setProphetDaily] = useState(false);
    const [prophetMode, setProphetMode] = useState('additive');
    const [prophetHorizon, setProphetHorizon] = useState(30);
    const [isProphetLoading, setIsProphetLoading] = useState(false);

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
                const { data } = prepareChartData(rawData, selectedColumn);
                if (data.length > 0) {
                    const chartData = data.map(d => ({
                        time: d.time as any,
                        value: d.value
                    }));
                    pastSeriesRef.current.setData(chartData);
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

            let dataInput: number[] | null = null;

            // Use helper to get clean data
            const { values, lastPoint, interval } = prepareChartData(rawData, selectedColumn);
            if (values.length > 0) {
                dataInput = values;
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

            if (predictionSeriesRef.current && lastPoint) {
                console.log("ARIMA used seed:", result.used_seed);

                let pathData = result.path.map((v, i) => ({
                    time: (lastPoint.time + (interval * (i + 1))) as any,
                    value: v
                }));

                // Stitch: Add the point at lastTime to connect the lines
                pathData = [{ time: lastPoint.time as any, value: lastPoint.value }, ...pathData];

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

            let dataInput: number[] | null = null;
            const { values, lastPoint, interval } = prepareChartData(rawData, selectedColumn);

            if (values.length > 1) {
                // Calculate Log Returns
                dataInput = [];
                for (let i = 1; i < values.length; i++) {
                    const p_t = values[i];
                    const p_prev = values[i - 1];
                    dataInput.push(Math.log(p_t / p_prev));
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

            if (predictionSeriesRef.current && lastPoint) {
                const pathData = result.volatility.map((v, i) => ({
                    time: (lastPoint.time + (interval * (i + 1))) as any,
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

    const runHoltWinters = async () => {
        setIsHwLoading(true);
        try {
            let dataInput: number[] | null = null;
            const { values, lastPoint, interval } = prepareChartData(rawData, selectedColumn);
            if (values.length > 0) {
                dataInput = values;
            }

            const result = await invoke<HoltWintersResult>('predict_holt_winters', {
                params: {
                    alpha: hwAlpha,
                    beta: hwBeta,
                    gamma: hwGamma,
                    period: hwPeriod,
                    seasonal_type: hwSeasonalType,
                    steps: hwSteps,
                    sigma: hwSigma,
                    seed: seed === '' ? null : seed,
                    data: dataInput
                }
            });

            if (predictionSeriesRef.current && lastPoint) {
                let pathData = result.path.map((v, i) => ({
                    time: (lastPoint.time + (interval * (i + 1))) as any,
                    value: v
                }));

                if (dataInput && dataInput.length > 0) {
                    pathData = [{ time: lastPoint.time as any, value: lastPoint.value }, ...pathData];
                }

                predictionSeriesRef.current.setData(pathData);
                chartRef.current?.timeScale().fitContent();
            }

        } catch (error) {
            console.error('Holt-Winters Error:', error);
            // alert('Holt-Winters simulation failed: ' + error);
        } finally {
            setIsHwLoading(false);
        }
    };

    const runProphet = async () => {
        if (!chartRef.current || !predictionSeriesRef.current || !pastSeriesRef.current) return;
        setIsProphetLoading(true);

        try {
            // Prepare Data
            const values = rawData.map(d => parseFloat(d[selectedColumn])).filter(v => !isNaN(v));
            // Create synthetic timestamps if rawData doesn't have good Time column or we just want indices
            // But Prophet needs real time for seasonality.
            // Try to find a date column.
            let times: number[] = [];

            // Simple heuristic to find date col: looks for 'date' or 'time' in columns
            const dateCol = columns.find(c => c.toLowerCase().includes('date') || c.toLowerCase().includes('time'));

            if (dateCol) {
                times = rawData.map(d => {
                    const val = d[dateCol];
                    // Try parse
                    const dt = new Date(val);
                    if (!isNaN(dt.getTime())) {
                        return Math.floor(dt.getTime() / 1000); // Unix seconds
                    }
                    return 0;
                }).filter(t => t > 0);
            }

            // Fallback if no valid times found: generate daily sequence from now backwards? 
            // Or just 0, 86400, ... params depend on real time for seasonality logic!
            if (times.length !== values.length) {
                // Synthesize
                const now = Math.floor(Date.now() / 1000);
                times = Array.from({ length: values.length }, (_, i) => now - (values.length - 1 - i) * 86400);
            }

            const params = {
                growth: prophetGrowth,
                changepoints: null,
                seasonality_mode: prophetMode,
                yearly_seasonality: prophetYearly,
                weekly_seasonality: prophetWeekly,
                daily_seasonality: prophetDaily,
                seasonality_prior_scale: 10.0,
                changepoint_prior_scale: 0.05,
                forecast_horizon: prophetHorizon,
                times: times,
                values: values
            };

            const result = await invoke<ProphetResult>('predict_prophet', { params });

            // Visualize
            const lastTime = times[times.length - 1];
            const predData = result.values.map((v, i) => ({
                time: result.times[i] as any,
                value: v
            }));

            // Connect last past point to first pred point for visual continuity
            const bridge = { time: lastTime as any, value: values[values.length - 1] };

            predictionSeriesRef.current.setData([bridge, ...predData]);

            // Re-render past to match scale if needed (times usually match)
            // But if we synthesized times, we might need to update past series too?
            const pastData = values.map((v, i) => ({
                time: times[i] as any,
                value: v
            }));
            pastSeriesRef.current.setData(pastData);
        } catch (error) {
            console.error('Prophet Error:', error);
        } finally {
            setIsProphetLoading(false);
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
                        <button
                            onClick={() => setActiveModel('holt_winters')}
                            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeModel === 'holt_winters' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                            Holt-Winters
                        </button>
                        <button
                            onClick={() => setActiveModel('prophet')}
                            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeModel === 'prophet' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                            Prophet
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
                        ) : activeModel === 'garch' ? (
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
                        ) : activeModel === 'holt_winters' ? (
                            <div className="space-y-4">
                                <h3 className="text-xs font-semibold uppercase text-slate-400 flex items-center gap-2">
                                    <Activity size={14} /> HW Configuration
                                </h3>
                                <div className="space-y-3">
                                    <label className="block text-xs text-slate-400">
                                        Alpha (Level)
                                        <input
                                            type="number"
                                            value={hwAlpha}
                                            onChange={e => setHwAlpha(parseFloat(e.target.value))}
                                            step="0.05"
                                            max="1"
                                            min="0"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Beta (Trend)
                                        <input
                                            type="number"
                                            value={hwBeta}
                                            onChange={e => setHwBeta(parseFloat(e.target.value))}
                                            step="0.05"
                                            max="1"
                                            min="0"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Gamma (Seasonal)
                                        <input
                                            type="number"
                                            value={hwGamma}
                                            onChange={e => setHwGamma(parseFloat(e.target.value))}
                                            step="0.05"
                                            max="1"
                                            min="0"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <div className="grid grid-cols-2 gap-3">
                                        <label className="text-xs text-slate-400">
                                            Period
                                            <input
                                                type="number"
                                                value={hwPeriod}
                                                onChange={e => setHwPeriod(parseInt(e.target.value))}
                                                min="1"
                                                className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                            />
                                        </label>
                                        <label className="text-xs text-slate-400">
                                            Type
                                            <select
                                                value={hwSeasonalType}
                                                onChange={e => setHwSeasonalType(e.target.value as any)}
                                                className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                            >
                                                <option value="Additive">Additive</option>
                                                <option value="Multiplicative">Multiplicative</option>
                                            </select>
                                        </label>
                                    </div>
                                    <label className="block text-xs text-slate-400">
                                        Steps
                                        <input
                                            type="number"
                                            value={hwSteps}
                                            onChange={e => setHwSteps(parseInt(e.target.value))}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Noise Sigma
                                        <input
                                            type="number"
                                            value={hwSigma}
                                            onChange={e => setHwSigma(parseFloat(e.target.value))}
                                            step="0.1"
                                            min="0"
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Random Seed
                                        <input
                                            type="number"
                                            value={seed}
                                            onChange={e => setSeed(e.target.value === '' ? '' : parseInt(e.target.value))}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>
                                    <button
                                        onClick={runHoltWinters}
                                        disabled={isHwLoading}
                                        className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
                                    >
                                        {isHwLoading ? <Loader2 className="animate-spin" size={16} /> : 'Generate Path'}
                                    </button>
                                </div>
                            </div>
                        ) : activeModel === 'prophet' ? (
                            <div className="space-y-4">
                                <h3 className="text-xs font-semibold uppercase text-slate-400 flex items-center gap-2">
                                    <Activity size={14} /> Prophet Configuration
                                </h3>
                                <div className="space-y-3">
                                    <label className="block text-xs text-slate-400">
                                        Growth Trend
                                        <select
                                            value={prophetGrowth}
                                            onChange={e => setProphetGrowth(e.target.value)}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        >
                                            <option value="linear">Linear</option>
                                            <option value="flat">Flat</option>
                                        </select>
                                    </label>
                                    <div className="flex gap-4">
                                        <label className="flex items-center gap-2 text-xs text-slate-400">
                                            <input
                                                type="checkbox"
                                                checked={prophetYearly}
                                                onChange={e => setProphetYearly(e.target.checked)}
                                                className="rounded bg-slate-800 border-slate-700"
                                            />
                                            Yearly
                                        </label>
                                        <label className="flex items-center gap-2 text-xs text-slate-400">
                                            <input
                                                type="checkbox"
                                                checked={prophetWeekly}
                                                onChange={e => setProphetWeekly(e.target.checked)}
                                                className="rounded bg-slate-800 border-slate-700"
                                            />
                                            Weekly
                                        </label>
                                        <label className="flex items-center gap-2 text-xs text-slate-400">
                                            <input
                                                type="checkbox"
                                                checked={prophetDaily}
                                                onChange={e => setProphetDaily(e.target.checked)}
                                                className="rounded bg-slate-800 border-slate-700"
                                            />
                                            Daily
                                        </label>
                                    </div>

                                    <label className="block text-xs text-slate-400">
                                        Seasonality Mode
                                        <select
                                            value={prophetMode}
                                            onChange={e => setProphetMode(e.target.value)}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        >
                                            <option value="additive">Additive</option>
                                            <option value="multiplicative">Multiplicative</option>
                                        </select>
                                    </label>

                                    <label className="block text-xs text-slate-400">
                                        Forecast Horizon (Steps)
                                        <input
                                            type="number"
                                            value={prophetHorizon}
                                            onChange={e => setProphetHorizon(parseInt(e.target.value))}
                                            className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                                        />
                                    </label>

                                    <button
                                        onClick={runProphet}
                                        disabled={isProphetLoading}
                                        className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
                                    >
                                        {isProphetLoading ? <Loader2 className="animate-spin" size={16} /> : 'Generate Forecast'}
                                    </button>
                                </div>
                            </div>
                        ) : null}
                    </div>
                </div>

                <div className="flex-1 p-6 flex flex-col gap-6 overflow-hidden">
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col relative">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-semibold text-slate-300">
                                {activeModel === 'arima' ? 'Simulated Price Path (Integrated)' :
                                    activeModel === 'garch' ? 'Conditional Volatility Series' :
                                        activeModel === 'holt_winters' ? 'Holt-Winters Forecast' :
                                            'Prophet Forecast'}
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
                        ) : activeModel === 'garch' ? (
                            <p>
                                GARCH models use historical returns to estimate initial conditional variance.
                                This allows for more realistic volatility forecasting based on recent market regimes found in your data.
                            </p>
                        ) : activeModel === 'holt_winters' ? (
                            <p>
                                Holt-Winters (Triple Exponential Smoothing) captures level, trend, and seasonality.
                                Adjust Alpha, Beta, and Gamma to control how much weight is given to recent vs. old data for each component.
                            </p>
                        ) : (
                            <p>
                                Prophet uses a decomposable time series model with three main components: trend, seasonality, and holidays.
                                It is robust to missing data and shifts in the trend, and specifically designed for business forecasting.
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div >
    );
}
