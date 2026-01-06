import { createChart, ColorType, IChartApi, ISeriesApi, LineSeries } from 'lightweight-charts';
import { useEffect, useRef } from 'react';
import { ArenaUpdate } from '../../hooks/useArena';

interface ChartProps {
    data: ArenaUpdate[];
}

export function PriceChart({ data }: ChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const priceSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
    const portfolioSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#1a1a1a' },
                textColor: '#d1d5db',
            },
            grid: {
                vertLines: { color: '#333' },
                horzLines: { color: '#333' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 400,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            }
        });

        const priceSeries = chart.addSeries(LineSeries, {
            color: '#22c55e',
            lineWidth: 2,
            title: 'Price',
        });

        const portfolioSeries = chart.addSeries(LineSeries, {
            color: '#3b82f6',
            lineWidth: 2,
            title: 'Portfolio',
            priceScaleId: 'left', // Use separate scale
        });

        chartRef.current = chart;
        priceSeriesRef.current = priceSeries;
        portfolioSeriesRef.current = portfolioSeries;

        return () => {
            chart.remove();
        };
    }, []);

    // Update data
    useEffect(() => {
        if (!priceSeriesRef.current || !portfolioSeriesRef.current) return;

        // Map data to chart format
        // lightweight-charts expects sorted time.
        // We assume data is appended in order.
        // If we reload generic chart with full history:
        const priceData = data.map(d => ({ time: d.step as any, value: d.price }));
        const portfolioData = data.map(d => ({ time: d.step as any, value: d.portfolio_value }));

        priceSeriesRef.current.setData(priceData);
        portfolioSeriesRef.current.setData(portfolioData);

        // Auto scale if needed
        // chartRef.current?.timeScale().fitContent(); 
    }, [data.length]); // Optimize: only update when length changes?

    return <div ref={chartContainerRef} className="w-full h-[400px]" />;
}
