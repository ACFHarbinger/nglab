import { useState } from 'react';
import { Calculator, Loader2 } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';

const clampNumber = (value: number, min?: number, max?: number) => {
  if (!Number.isFinite(value)) return Number.isFinite(min) ? (min as number) : 0;
  if (Number.isFinite(min) && value < (min as number)) return min as number;
  if (Number.isFinite(max) && value > (max as number)) return max as number;
  return value;
};

type BlackScholesResult = {
  call: number;
  put: number;
  delta: number;
  gamma: number;
  vega: number;
  d1: number;
  d2: number;
};

type CreditRiskResult = {
  base: number;
  adjusted: number;
  survival: number;
  cva: number;
};

type RBergomiResult = {
  price: number;
  std_error: number;
  mean_terminal: number;
};

type RoughHestonResult = {
  price: number;
  std_error: number;
  mean_terminal: number;
  p05: number;
  p95: number;
  paths: number;
  steps: number;
};

function PricingTab() {
  const [model, setModel] = useState<'bsm' | 'rbergomi' | 'rough-heston' | 'credit'>('bsm');

  const [spot, setSpot] = useState(100);
  const [strike, setStrike] = useState(100);
  const [rate, setRate] = useState(0.00); // Risk-free rate often 0 in simple rBergomi implementations unless adjusted
  const [volatility, setVolatility] = useState(0.2); // For BSM
  const [maturity, setMaturity] = useState(1);
  const [optionType, setOptionType] = useState<'call' | 'put'>('call');

  // rBergomi params
  const [xi, setXi] = useState(0.04); // Initial variance
  const [eta, setEta] = useState(1.9); // Vol of Vol
  const [hurst, setHurst] = useState(0.07); // Roughness
  const [rho, setRho] = useState(-0.9); // Correlation

  const [paths, setPaths] = useState(10000);
  const [steps, setSteps] = useState(100);

  const [hestonV0, setHestonV0] = useState(0.04);
  const [hestonTheta, setHestonTheta] = useState(0.04);
  const [hestonKappa, setHestonKappa] = useState(1.5);
  const [hestonNu, setHestonNu] = useState(0.6);
  const [hestonRho, setHestonRho] = useState(-0.4);
  const [hestonHurst, setHestonHurst] = useState(0.1);
  const [hestonPaths, setHestonPaths] = useState(2000);
  const [hestonSteps, setHestonSteps] = useState(80);

  const [hazardRate, setHazardRate] = useState(0.02);
  const [recovery, setRecovery] = useState(0.4);

  const [rbergomiResult, setRbergomiResult] = useState<RBergomiResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationError, setSimulationError] = useState('');
  const [roughHestonResult, setRoughHestonResult] = useState<RoughHestonResult | null>(null);
  const [isHestonSimulating, setIsHestonSimulating] = useState(false);
  const [roughHestonError, setRoughHestonError] = useState('');

  const [bsmResult, setBsmResult] = useState<BlackScholesResult | null>(null);
  const [bsmError, setBsmError] = useState('');
  const [creditResult, setCreditResult] = useState<CreditRiskResult | null>(null);
  const [creditError, setCreditError] = useState('');

  const [isBsmSimulating, setIsBsmSimulating] = useState(false);
  const [isCreditSimulating, setIsCreditSimulating] = useState(false);

  const runBsm = async () => {
    setBsmError('');
    setIsBsmSimulating(true);
    try {
      const result = await invoke<BlackScholesResult>('pricing_black_scholes', {
        params: { spot, strike, rate, volatility, maturity }
      });
      setBsmResult(result);
    } catch (err: any) {
      console.error(err);
      setBsmError('Pricing failed: ' + err.toString());
    } finally {
      setIsBsmSimulating(false);
    }
  };

  const runCredit = async () => {
    setCreditError('');
    setIsCreditSimulating(true);
    try {
      const result = await invoke<CreditRiskResult>('pricing_credit_risk', {
        params: {
          spot,
          strike,
          rate,
          volatility,
          maturity,
          option_type: optionType,
          hazard_rate: hazardRate,
          recovery
        }
      });
      setCreditResult(result);
    } catch (err: any) {
      console.error(err);
      setCreditError('Credit pricing failed: ' + err.toString());
    } finally {
      setIsCreditSimulating(false);
    }
  };

  const runRBergomi = async () => {
    setSimulationError('');
    setIsSimulating(true);

    try {
      // Invoke Rust backend
      const result = await invoke<RBergomiResult>('pricing_rbergomi', {
        params: {
          spot,
          strike,
          t: maturity,
          steps: clampNumber(Math.round(steps), 10, 1000),
          paths: clampNumber(Math.round(paths), 100, 100000),
          h: clampNumber(hurst, 0.01, 0.49),
          eta,
          xi,
          rho: clampNumber(rho, -0.99, 0.99)
        }
      });
      setRbergomiResult(result);
    } catch (err: any) {
      console.error(err);
      setSimulationError('Simulation failed: ' + err.toString());
    } finally {
      setIsSimulating(false);
    }
  };

  const runRoughHeston = async () => {
    setRoughHestonError('');
    setIsHestonSimulating(true);

    try {
      const result = await invoke<RoughHestonResult>('pricing_rough_heston', {
        params: {
          spot,
          strike,
          rate,
          maturity,
          v0: hestonV0,
          theta: hestonTheta,
          kappa: hestonKappa,
          nu: hestonNu,
          rho: clampNumber(hestonRho, -0.99, 0.99),
          hurst: clampNumber(hestonHurst, 0.01, 0.49),
          steps: clampNumber(Math.round(hestonSteps), 16, 512),
          paths: clampNumber(Math.round(hestonPaths), 100, 10000),
          option_type: optionType
        }
      });
      setRoughHestonResult(result);
    } catch (err: any) {
      console.error(err);
      setRoughHestonError('Simulation failed: ' + err.toString());
    } finally {
      setIsHestonSimulating(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30">
      <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/20">
        <div className="flex items-center gap-6">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Pricing Lab</h2>
            <p className="text-slate-400 text-sm">Traditional stochastic pricing and risk overlays.</p>
          </div>
          <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
            {([
              { id: 'bsm', label: 'Black-Scholes-Merton' },
              { id: 'rbergomi', label: 'Rough Bergomi' },
              { id: 'rough-heston', label: 'Rough Heston' },
              { id: 'credit', label: 'Counterparty Risk' }
            ] as const).map((item) => (
              <button
                key={item.id}
                onClick={() => setModel(item.id)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${model === item.id ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm text-indigo-300">
          <Calculator size={18} />
          <span>Traditional Stochastics</span>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="w-[360px] bg-slate-900/40 border-r border-slate-800 p-4 overflow-y-auto space-y-6">
          <div className="space-y-2">
            <h3 className="text-xs font-semibold uppercase text-slate-400">Core Inputs</h3>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Spot
                <input
                  type="number"
                  value={spot}
                  onChange={(e) => setSpot(Number(e.target.value))}
                  step="0.1"
                  className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-xs text-slate-400">Strike
                <input
                  type="number"
                  value={strike}
                  onChange={(e) => setStrike(Number(e.target.value))}
                  step="0.1"
                  className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-xs text-slate-400">Rate
                <input
                  type="number"
                  value={rate}
                  onChange={(e) => setRate(Number(e.target.value))}
                  step="0.001"
                  className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-xs text-slate-400">Volatility (BSM)
                <input
                  type="number"
                  value={volatility}
                  onChange={(e) => setVolatility(Number(e.target.value))}
                  step="0.01"
                  className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-xs text-slate-400">Maturity (Yrs)
                <input
                  type="number"
                  value={maturity}
                  onChange={(e) => setMaturity(Number(e.target.value))}
                  step="0.1"
                  className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-xs text-slate-400">Option
                <select
                  value={optionType}
                  onChange={(e) => setOptionType(e.target.value as 'call' | 'put')}
                  className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                >
                  <option value="call">Call</option>
                  <option value="put">Put</option>
                </select>
              </label>
            </div>
          </div>

          {model === 'bsm' && (
            <div className="space-y-3 mt-4">
              <button
                onClick={runBsm}
                disabled={isBsmSimulating}
                className="w-full mt-2 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
              >
                {isBsmSimulating ? <Loader2 className="animate-spin" size={16} /> : null}
                Run BSM
              </button>
            </div>
          )}

          {model === 'rbergomi' && (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold uppercase text-slate-400">Rough Bergomi Inputs</h3>
              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs text-slate-400">Initial Var (xi)
                  <input
                    type="number"
                    value={xi}
                    onChange={(e) => setXi(Number(e.target.value))}
                    step="0.01"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Vol of Vol (eta)
                  <input
                    type="number"
                    value={eta}
                    onChange={(e) => setEta(Number(e.target.value))}
                    step="0.1"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Hurst (H)
                  <input
                    type="number"
                    value={hurst}
                    onChange={(e) => setHurst(Number(e.target.value))}
                    step="0.01"
                    min="0.01"
                    max="0.49"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Rho (Corr)
                  <input
                    type="number"
                    value={rho}
                    onChange={(e) => setRho(Number(e.target.value))}
                    step="0.1"
                    min="-0.99"
                    max="0.99"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Paths
                  <input
                    type="number"
                    value={paths}
                    onChange={(e) => setPaths(Number(e.target.value))}
                    step="1000"
                    min="100"
                    max="100000"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Steps
                  <input
                    type="number"
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    step="10"
                    min="10"
                    max="1000"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
              </div>
              <button
                onClick={runRBergomi}
                disabled={isSimulating}
                className="w-full mt-2 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
              >
                {isSimulating ? <Loader2 className="animate-spin" size={16} /> : null}
                Run Simulation
              </button>
              <p className="text-[11px] text-slate-500 leading-snug">
                Backend-accelerated Rough Bergomi simulation using fractional Brownian motion Cholesky decomposition.
              </p>
            </div>
          )}

          {model === 'rough-heston' && (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold uppercase text-slate-400">Rough Heston Inputs</h3>
              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs text-slate-400">v0
                  <input
                    type="number"
                    value={hestonV0}
                    onChange={(e) => setHestonV0(Number(e.target.value))}
                    step="0.01"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">theta
                  <input
                    type="number"
                    value={hestonTheta}
                    onChange={(e) => setHestonTheta(Number(e.target.value))}
                    step="0.01"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">kappa
                  <input
                    type="number"
                    value={hestonKappa}
                    onChange={(e) => setHestonKappa(Number(e.target.value))}
                    step="0.1"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">nu
                  <input
                    type="number"
                    value={hestonNu}
                    onChange={(e) => setHestonNu(Number(e.target.value))}
                    step="0.05"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">rho
                  <input
                    type="number"
                    value={hestonRho}
                    onChange={(e) => setHestonRho(Number(e.target.value))}
                    step="0.05"
                    min="-0.99"
                    max="0.99"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Hurst (H)
                  <input
                    type="number"
                    value={hestonHurst}
                    onChange={(e) => setHestonHurst(Number(e.target.value))}
                    step="0.01"
                    min="0.01"
                    max="0.49"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Paths
                  <input
                    type="number"
                    value={hestonPaths}
                    onChange={(e) => setHestonPaths(Number(e.target.value))}
                    step="100"
                    min="100"
                    max="10000"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Steps
                  <input
                    type="number"
                    value={hestonSteps}
                    onChange={(e) => setHestonSteps(Number(e.target.value))}
                    step="8"
                    min="16"
                    max="512"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
              </div>
              <button
                onClick={runRoughHeston}
                disabled={isHestonSimulating}
                className="w-full mt-2 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
              >
                {isHestonSimulating ? <Loader2 className="animate-spin" size={16} /> : null}
                Run Rough Heston
              </button>
              <p className="text-[11px] text-slate-500 leading-snug">
                Rough Heston applies a fractional variance driver with correlated spot noise for a rough volatility surface.
              </p>
            </div>
          )}

          {model === 'credit' && (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold uppercase text-slate-400">Credit Overlay</h3>
              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs text-slate-400">Hazard Rate
                  <input
                    type="number"
                    value={hazardRate}
                    onChange={(e) => setHazardRate(Number(e.target.value))}
                    step="0.005"
                    min="0"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs text-slate-400">Recovery
                  <input
                    type="number"
                    value={recovery}
                    onChange={(e) => setRecovery(Number(e.target.value))}
                    step="0.05"
                    min="0"
                    max="1"
                    className="mt-1 w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm"
                  />
                </label>
              </div>
              <button
                onClick={runCredit}
                disabled={isCreditSimulating}
                className="w-full mt-2 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-2 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
              >
                {isCreditSimulating ? <Loader2 className="animate-spin" size={16} /> : null}
                Run Credit Model
              </button>
              <p className="text-[11px] text-slate-500 leading-snug">
                Applies a constant-intensity survival discount to the base option value.
              </p>
            </div>
          )}
        </div>

        <div className="flex-1 p-6 overflow-y-auto">
          {model === 'bsm' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">Price Outputs</h3>
                <div className="space-y-2 text-sm">
                  {bsmError && <div className="text-xs text-rose-400">{bsmError}</div>}
                  <div className="flex justify-between"><span className="text-slate-400">Call</span><span>{(bsmResult?.call ?? 0).toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Put</span><span>{(bsmResult?.put ?? 0).toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">d1</span><span>{(bsmResult?.d1 ?? 0).toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">d2</span><span>{(bsmResult?.d2 ?? 0).toFixed(4)}</span></div>
                </div>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">Greeks</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-slate-400">Delta</span><span>{(bsmResult?.delta ?? 0).toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Gamma</span><span>{(bsmResult?.gamma ?? 0).toFixed(6)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Vega</span><span>{(bsmResult?.vega ?? 0).toFixed(4)}</span></div>
                </div>
              </div>
            </div>
          )}

          {model === 'rbergomi' && (
            <div className="space-y-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">Results</h3>
                {simulationError && (
                  <div className="text-sm text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-md p-3 mb-3">
                    {simulationError}
                  </div>
                )}
                {!rbergomiResult ? (
                  <p className="text-sm text-slate-500">Run the simulation to estimate the option value.</p>
                ) : (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="space-y-2">
                      <div className="flex justify-between"><span className="text-slate-400">Price</span><span>{rbergomiResult.price.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Std. Error</span><span>{rbergomiResult.std_error.toFixed(4)}</span></div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between"><span className="text-slate-400">Mean Terminal</span><span>{rbergomiResult.mean_terminal.toFixed(2)}</span></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="bg-slate-900/30 border border-slate-800 rounded-xl p-4 text-sm text-slate-400">
                <p>
                  Rough Bergomi (rBergomi) uses Fractional Brownian Motion (H &lt; 0.5) to simultaneously capture the power-law skew and the roughness of volatility time series.
                  This simulation runs completely in Rust for high performance, utilizing parallel execution or optimized Cholesky decomposition.
                </p>
              </div>
            </div>
          )}

          {model === 'rough-heston' && (
            <div className="space-y-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">Rough Heston Results</h3>
                {roughHestonError && (
                  <div className="text-sm text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-md p-3 mb-3">
                    {roughHestonError}
                  </div>
                )}
                {!roughHestonResult ? (
                  <p className="text-sm text-slate-500">Run the simulation to estimate the option value.</p>
                ) : (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="space-y-2">
                      <div className="flex justify-between"><span className="text-slate-400">Price</span><span>{roughHestonResult.price.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Std. Error</span><span>{roughHestonResult.std_error.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Paths</span><span>{roughHestonResult.paths}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Steps</span><span>{roughHestonResult.steps}</span></div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between"><span className="text-slate-400">Mean Terminal</span><span>{roughHestonResult.mean_terminal.toFixed(2)}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">5% Quantile</span><span>{roughHestonResult.p05.toFixed(2)}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">95% Quantile</span><span>{roughHestonResult.p95.toFixed(2)}</span></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="bg-slate-900/30 border border-slate-800 rounded-xl p-4 text-sm text-slate-400">
                <p>
                  Rough Heston couples fractional volatility dynamics with correlated spot shocks to capture the steep short-dated smile and volatility memory.
                </p>
              </div>
            </div>
          )}

          {model === 'credit' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">Base Price</h3>
                <div className="space-y-2 text-sm">
                  {creditError && <div className="text-xs text-rose-400">{creditError}</div>}
                  <div className="flex justify-between"><span className="text-slate-400">Unadjusted</span><span>{(creditResult?.base ?? 0).toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Survival</span><span>{(creditResult?.survival ?? 0).toFixed(4)}</span></div>
                </div>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">Credit Adjusted</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-slate-400">Adjusted Price</span><span>{(creditResult?.adjusted ?? 0).toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">CVA</span><span>{(creditResult?.cva ?? 0).toFixed(4)}</span></div>
                </div>
              </div>
              <div className="col-span-2 bg-slate-900/30 border border-slate-800 rounded-xl p-4 text-sm text-slate-400">
                <p>
                  Counterparty credit risk discounts the option value by survival probability, mirroring vulnerable option pricing
                  with a constant hazard rate and recovery assumption.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default PricingTab;
