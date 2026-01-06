use nglab::gym::TradingEnv;
use nglab::orderbook::OrderBook;
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};
use tokio::time::{sleep, Duration};

struct ArenaState {
    env: Mutex<TradingEnv>,
    running: Mutex<bool>,
}

#[derive(serde::Serialize, Clone)]
struct ArenaUpdate {
    step: u64,
    price: f64,
    portfolio_value: f64,
    orderbook: OrderBook,
}

#[tauri::command]
fn start_simulation(state: State<ArenaState>, app: tauri::AppHandle) {
    let mut running = state.running.lock().unwrap();
    if *running {
        return;
    }
    *running = true;

    tauri::async_runtime::spawn(async move {
        let state = app.state::<ArenaState>();
        loop {
            // Check if we should keep running
            {
                let running = state.running.lock().unwrap();
                if !*running {
                    break;
                }
            }

            // Perform simulation step
            let update = {
                let mut env = state.env.lock().unwrap();
                // 0 = Hold action
                let (_, _, _, _, step_info) = env.step_rs(0);

                let orderbook = env.orderbook().clone();
                let price = orderbook.mid_price().unwrap_or(0.0);

                ArenaUpdate {
                    step: step_info.total_steps,
                    price,
                    portfolio_value: step_info.portfolio_value,
                    orderbook,
                }
            };

            // Emit event to frontend
            // We ignore errors if frontend is closed/not listening
            let _ = app.emit("arena-update", &update);

            sleep(Duration::from_millis(100)).await;
        }
    });
}

#[tauri::command]
fn stop_simulation(state: State<ArenaState>) {
    let mut running = state.running.lock().unwrap();
    *running = false;
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize TradingEnv with default parameters
    let env = TradingEnv::new(10000.0, 0.001, 30, 1000);

    let state = ArenaState {
        env: Mutex::new(env),
        running: Mutex::new(false),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![start_simulation, stop_simulation])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
