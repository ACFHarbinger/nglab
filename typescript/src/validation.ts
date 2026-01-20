import { z } from 'zod';

/**
 * Validation schema for Order parameters.
 */
export const OrderSchema = z.object({
    price: z.number().positive('Price must be positive'),
    quantity: z.number().positive('Quantity must be positive'),
    side: z.enum(['Bid', 'Ask']),
});

/**
 * Validation schema for ARIMA parameters.
 */
export const ArimaParamsSchema = z.object({
    ar: z.array(z.number()),
    ma: z.array(z.number()),
    d: z.number().int().min(0),
    steps: z.number().int().positive(),
    sigma: z.number().nonnegative(),
    seed: z.number().optional(),
    data: z.array(z.number()).optional(),
});

/**
 * Validation schema for GARCH parameters.
 */
export const GarchParamsSchema = z.object({
    omega: z.number().nonnegative(),
    alpha: z.array(z.number()),
    beta: z.array(z.number()),
    steps: z.number().int().positive(),
    seed: z.number().optional(),
    data: z.array(z.number()).optional(),
});

/**
 * Validation schema for Holt-Winters parameters.
 */
export const HoltWintersParamsSchema = z.object({
    alpha: z.number().min(0).max(1),
    beta: z.number().min(0).max(1),
    gamma: z.number().min(0).max(1),
    period: z.number().int().positive(),
    seasonal_type: z.enum(['Additive', 'Multiplicative']),
    steps: z.number().int().positive(),
    sigma: z.number().nonnegative(),
    seed: z.number().optional(),
    data: z.array(z.number()).optional(),
});

/**
 * Validation schema for Prophet parameters.
 */
export const ProphetParamsSchema = z.object({
    growth: z.enum(['linear', 'flat']),
    changepoints: z.array(z.number().int()).optional(),
    seasonality_mode: z.enum(['additive', 'multiplicative']),
    yearly_seasonality: z.boolean(),
    weekly_seasonality: z.boolean(),
    daily_seasonality: z.boolean(),
    seasonality_prior_scale: z.number().positive(),
    changepoint_prior_scale: z.number().positive(),
    forecast_horizon: z.number().int().positive(),
    times: z.array(z.number().int()),
    values: z.array(z.number()),
});
