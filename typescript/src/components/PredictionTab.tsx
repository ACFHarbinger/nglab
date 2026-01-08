import { useState, useEffect, useRef } from 'react';
import { Brain, Loader2, LineChart as LineChartIcon, Activity, FileSpreadsheet } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { createChart, ColorType, IChartApi, ISeriesApi, LineSeries } from 'lightweight-charts';
import { open } from '@tauri-apps/plugin-dialog';
import { readTextFile } from '@tauri-apps/plugin-fs';
import Papa from 'papaparse';

type ArimaResult = {
    path: number[];
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
            const data = rawData.map((row, i) => ({
                time: i as any,
                value: parseFloat(row[selectedColumn])
            })).filter(d => !isNaN(d.value));
            pastSeriesRef.current.setData(data);
            chartRef.current?.timeScale().fitContent();
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
                        const data = results.data as any[];
                        if (data.length > 0) {
                            setRawData(data);
                            const cols = Object.keys(data[0]);
                            setColumns(cols);
                            // Try to find a good default column
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
            if (rawData.length > 0 && selectedColumn) {
                dataInput = rawData.map(row => parseFloat(row[selectedColumn])).filter(n => !isNaN(n));
            }

            const result = await invoke<ArimaResult>('predict_arima', {
                params: {
                    ar: arCoeffs,
                    ma: maCoeffs,
                    d,
                    steps: arimaSteps,
                    sigma: arimaSigma,
                    seed: null,
                    data: dataInput
                }
            });
            
            if (predictionSeriesRef.current) {
                const startIdx = dataInput ? dataInput.length : 0;
                predictionSeriesRef.current.setData(result.path.map((v, i) => ({ time: (startIdx + i) as any, value: v })));
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
            if (rawData.length > 0 && selectedColumn) {
                // For GARCH we usually need returns
                const prices = rawData.map(row => parseFloat(row[selectedColumn])).filter(n => !isNaN(n));
                if (prices.length > 1) {
                    dataInput = [];
                    for(let i=1; i<prices.length; i++) {
                        dataInput.push(Math.log(prices[i] / prices[i-1]));
                    }
                }
            }

            const result = await invoke<GarchResult>('predict_garch', {
                params: {
                    omega,
                    alpha: alphaCoeffs,
                    beta: betaCoeffs,
                    steps: garchSteps,
                    seed: null,
                    data: dataInput
                }
            });
            
            if (predictionSeriesRef.current) {
                const startIdx = dataInput ? dataInput.length : 0;
                // Visualize volatility for GARCH
                predictionSeriesRef.current.setData(result.volatility.map((v, i) => ({ time: (startIdx + i) as any, value: v })));
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
