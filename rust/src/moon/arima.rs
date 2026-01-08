use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ArimaParams {
    pub ar: Vec<f64>, // AR coefficients (phi)
    pub ma: Vec<f64>, // MA coefficients (theta)
    pub d: usize,     // Integration order
    pub steps: usize,
    pub sigma: f64, // Noise standard deviation
    pub seed: Option<u64>,
    pub data: Option<Vec<f64>>, // Past data
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ArimaResult {
    pub path: Vec<f64>,
}

pub fn simulate(params: ArimaParams) -> Result<ArimaResult, String> {
    let mut rng = if let Some(s) = params.seed {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(s)
    } else {
        use rand::SeedableRng;
        rand::rngs::StdRng::from_entropy()
    };

    let p = params.ar.len();
    let q = params.ma.len();

    // If data is provided, we use it to initialize.
    // Otherwise, we start from zero with a warmup.
    let initial_series = if let Some(ref data) = params.data {
        if data.len() <= params.d {
            return Err("Data length must be greater than integration order d".to_string());
        }
        let mut current = data.clone();
        for _ in 0..params.d {
            let mut diff = Vec::new();
            for i in 1..current.len() {
                diff.push(current[i] - current[i - 1]);
            }
            current = diff;
        }
        current
    } else {
        vec![0.0; p.max(q).max(params.d) + 100] // Warmup
    };

    let n = params.steps + initial_series.len();

    // Generate white noise (epsilon)
    let mut eps = vec![0.0; n];
    for t in 0..n {
        let z: f64 = StandardNormal.sample(&mut rng);
        eps[t] = z * params.sigma;
    }

    // Generate ARMA process
    let mut x = vec![0.0; n];
    // Copy initial series
    for t in 0..initial_series.len() {
        x[t] = initial_series[t];
    }

    let start_idx = initial_series.len();

    for t in start_idx..n {
        let mut val = eps[t];

        // AR part
        for i in 0..p {
            if t > i {
                val += params.ar[i] * x[t - 1 - i];
            }
        }

        // MA part
        for j in 0..q {
            if t > j {
                val += params.ma[j] * eps[t - 1 - j];
            }
        }

        x[t] = val;
    }

    // Integrate d times
    let result_x = x;
    if let Some(ref data) = params.data {
        let mut current_predictions = result_x[start_idx..].to_vec();

        for d_iter in 0..params.d {
            // Get the series at one level less of differencing
            let mut level_data = data.clone();
            for _ in 0..(params.d - 1 - d_iter) {
                let mut diff = Vec::new();
                for i in 1..level_data.len() {
                    diff.push(level_data[i] - level_data[i - 1]);
                }
                level_data = diff;
            }

            let mut last_val = *level_data.last().unwrap();
            let mut next_preds = Vec::new();
            for p_val in current_predictions {
                last_val += p_val;
                next_preds.push(last_val);
            }
            current_predictions = next_preds;
        }
        Ok(ArimaResult {
            path: current_predictions,
        })
    } else {
        let mut integrated_x = result_x;
        for _ in 0..params.d {
            let mut integrated = vec![0.0; n];
            let mut cumsum = 0.0;
            for t in 0..n {
                cumsum += integrated_x[t];
                integrated[t] = cumsum;
            }
            integrated_x = integrated;
        }
        let result_path = integrated_x.into_iter().skip(n - params.steps).collect();
        Ok(ArimaResult { path: result_path })
    }
}
