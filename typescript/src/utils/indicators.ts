/**
 * @module utils/indicators
 * @description Technical indicator calculation utilities.
 */

/**
 * Calculates the Simple Moving Average (SMA).
 * @param data Array of prices.
 * @param period Lookback period.
 * @returns Array of SMA values aligned with the input data (first period-1 are NaN or null).
 */
export function calculateSMA(data: number[], period: number): (number | null)[] {
    const sma = new Array(data.length).fill(null);
    if (data.length < period) return sma;

    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += data[i];
    }
    sma[period - 1] = sum / period;

    for (let i = period; i < data.length; i++) {
        sum = sum - data[i - period] + data[i];
        sma[i] = sum / period;
    }

    return sma;
}

/**
 * Calculates the Exponential Moving Average (EMA).
 * @param data Array of prices.
 * @param period Lookback period.
 * @returns Array of EMA values.
 */
export function calculateEMA(data: number[], period: number): (number | null)[] {
    const ema = new Array(data.length).fill(null);
    if (data.length < period) return ema;

    const k = 2 / (period + 1);

    // Initialize with SMA
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += data[i];
    }
    let prevEma = sum / period;
    ema[period - 1] = prevEma;

    for (let i = period; i < data.length; i++) {
        const currentPrice = data[i];
        const currentEma = (currentPrice - prevEma) * k + prevEma;
        ema[i] = currentEma;
        prevEma = currentEma;
    }

    return ema;
}

/**
 * Calculates Bollinger Bands.
 * @param data Array of prices.
 * @param period MA period.
 * @param stdDev Multiplier for standard deviation.
 * @returns Object containing arrays for upper, middle, and lower bands.
 */
export function calculateBollingerBands(
    data: number[],
    period: number = 20,
    stdDev: number = 2
): { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] } {
    const middle = calculateSMA(data, period);
    const upper = new Array(data.length).fill(null);
    const lower = new Array(data.length).fill(null);

    for (let i = period - 1; i < data.length; i++) {
        const slice = data.slice(i - period + 1, i + 1);
        const mean = middle[i]!;

        const squaredDiffs = slice.map(val => Math.pow(val - mean, 2));
        const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
        const sd = Math.sqrt(variance);

        upper[i] = mean + sd * stdDev;
        lower[i] = mean - sd * stdDev;
    }

    return { upper, middle, lower };
}

/**
 * Calculates the Relative Strength Index (RSI).
 * @param data Array of prices.
 * @param period RSI period (typically 14).
 * @returns Array of RSI values.
 */
export function calculateRSI(data: number[], period: number = 14): (number | null)[] {
    const rsi = new Array(data.length).fill(null);
    if (data.length < period + 1) return rsi;

    let gains = 0;
    let losses = 0;

    for (let i = 1; i <= period; i++) {
        const diff = data[i] - data[i - 1];
        if (diff >= 0) {
            gains += diff;
        } else {
            losses += Math.abs(diff);
        }
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    rsi[period] = 100 - (100 / (1 + avgGain / (avgLoss === 0 ? 1 : avgLoss)));

    for (let i = period + 1; i < data.length; i++) {
        const diff = data[i] - data[i - 1];
        const currentGain = diff > 0 ? diff : 0;
        const currentLoss = diff < 0 ? Math.abs(diff) : 0;

        avgGain = ((avgGain * (period - 1)) + currentGain) / period;
        avgLoss = ((avgLoss * (period - 1)) + currentLoss) / period;

        if (avgLoss === 0) {
            rsi[i] = 100;
        } else {
            const rs = avgGain / avgLoss;
            rsi[i] = 100 - (100 / (1 + rs));
        }
    }

    return rsi;
}
