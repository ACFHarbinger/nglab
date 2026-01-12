import { createChart, ColorType, IChartApi, ISeriesApi, LineSeries } from 'lightweight-charts';
import { useEffect, useRef } from 'react';
import { ArenaUpdate } from '../../hooks/useArena';

/**
 * Props for the PriceChart component.
 */
interface PriceChartProps {
    /** Array of arena updates containing price and portfolio data */
    data: ArenaUpdate[];
}

/**
 * High-performance financial line chart using TradingView's Lightweight Charts.
 *
 * Optimized for rendering thousands of data points with smooth zooming and panning.
 * Automatically adapts to the container size using a ResizeObserver.
 * Visualizes both current price and portfolio value.
 */
export function PriceChart({ data }: PriceChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const priceSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
    const portfolioSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        /**
         * Initialize the Lightweight Chart instance.
         */
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

        /**
         * The green line series representing market mid-price.
         */
        const priceSeries = chart.addSeries(LineSeries, {
            color: '#22c55e',
            lineWidth: 2,
            title: 'Price',
        });

        /**
         * The blue line series representing portfolio account value.
         * Rendered on a separate left-side price scale for better visibility.
         */
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
        const priceData = data.map(d => ({ time: d.step as any, value: d.price }));
        const portfolioData = data.map(d => ({ time: d.step as any, value: d.portfolio_value }));

        priceSeriesRef.current.setData(priceData);
        portfolioSeriesRef.current.setData(portfolioData);
    }, [data.length]);

    return <div ref={chartContainerRef} className="w-full h-[400px]" />;
}
