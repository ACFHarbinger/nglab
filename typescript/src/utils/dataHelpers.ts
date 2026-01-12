
/**
 * Prepares raw CSV/JSON data for use in financial charts (Lightweight Charts).
 *
 * Handles timestamp normalization, deduplication, and sorting.
 * Also extracts a raw values array for backend simulation/forecasting.
 *
 * @param rawData - Array of objects from parsed CSV/JSON.
 * @param selectedColumn - The column name to extract values from.
 * @returns Formatted chart data, the last point, interval estimate, and raw values.
 */
export const prepareChartData = (rawData: any[], selectedColumn: string) => {
    if (!rawData || rawData.length === 0 || !selectedColumn) {
        return { data: [], lastPoint: null, interval: 86400, values: [] };
    }

    const hasDates = !isNaN(rawData[0]._ts);

    // Map and Filter
    const mapped = rawData.map((row, i) => ({
        time: hasDates ? row._ts / 1000 : i,
        value: parseFloat(row[selectedColumn]),
        originalRow: row
    })).filter(d => !isNaN(d.value) && d.value !== null && !isNaN(d.time));

    // Sort
    mapped.sort((a, b) => a.time - b.time);

    // Deduplicate
    const unique = [];
    if (mapped.length > 0) {
        unique.push(mapped[0]);
        for (let i = 1; i < mapped.length; i++) {
            if (mapped[i].time > mapped[i - 1].time) {
                unique.push(mapped[i]);
            }
        }
    }

    // Interval Estimate
    let interval = 86400;
    if (!hasDates) {
        interval = 1;
    } else if (unique.length >= 2) {
        // Use the average interval of the last few points for stability?
        // Or just the last diff? Last diff is simplest and connects to the immediate trend.
        interval = unique[unique.length - 1].time - unique[unique.length - 2].time;
    }

    // Extract values array for backend
    const values = unique.map(d => d.value);

    return {
        data: unique,
        lastPoint: unique.length > 0 ? unique[unique.length - 1] : null,
        interval,
        values
    };
};
