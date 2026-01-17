import { createChart, ColorType, IChartApi, ISeriesApi, AreaSeries } from 'lightweight-charts';
import { useEffect, useRef } from 'react';

interface ChartDataPoint {
    time: number;
    value: number;
}

interface TerminalChartProps {
    data: ChartDataPoint[];
    color?: string;
    height?: number;
}

export function TerminalChart({ data, color = '#22c55e', height }: TerminalChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#94a3b8',
            },
            grid: {
                vertLines: { color: '#1e293b', style: 1 },
                horzLines: { color: '#1e293b', style: 1 },
            },
            width: chartContainerRef.current.clientWidth,
            height: height || chartContainerRef.current.clientHeight || 400,
            timeScale: {
                timeVisible: true,
                secondsVisible: true,
                borderColor: '#1e293b',
            },
            rightPriceScale: {
                borderColor: '#1e293b',
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.1,
                },
            },
            crosshair: {
                vertLine: {
                    color: '#64748b',
                    labelBackgroundColor: '#64748b',
                },
                horzLine: {
                    color: '#64748b',
                    labelBackgroundColor: '#64748b',
                },
            },
        });

        // Use Area series for that "modern crypto" look
        const areaSeries = chart.addSeries(AreaSeries, {
            lineColor: color,
            topColor: color + '40', // 25% opacity
            bottomColor: color + '00', // 0% opacity
            lineWidth: 2,
            priceFormat: {
                type: 'price',
                precision: 3,
                minMove: 0.001,
            },
        });

        chartRef.current = chart;
        seriesRef.current = areaSeries;

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
    }, [color, height]);

    // Update data
    useEffect(() => {
        if (!seriesRef.current) return;
        // Sort data just in case, LWC requires sorted time
        const sortedData = [...data].sort((a, b) => a.time - b.time);
        seriesRef.current.setData(sortedData as any);

        // Auto-fit if we have data
        if (sortedData.length > 0 && chartRef.current) {
            chartRef.current.timeScale().fitContent();
        }
    }, [data.length]);

    return <div ref={chartContainerRef} className="w-full h-full min-h-[300px]" />;
}
