/*!
 * Simulation control commands.
 */

use crate::state::{ArenaState, ArenaUpdate};
use tauri::{Emitter, Manager, State};
use tokio::time::{sleep, Duration};

/**
 * Starts the simulation loop in a background task.
 *
 * This command initializes a loop that calls the environment's step function
 * and emits `arena-update` events to the frontend at a regular interval.
 *
 * @param state Shared application state.
 * @param app Tauri application handle for emitting events.
 */
#[tauri::command]
pub fn start_simulation(state: State<ArenaState>, app: tauri::AppHandle) {
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
            let _ = app.emit("arena-update", &update);

            sleep(Duration::from_millis(100)).await;
        }
    });
}

/**
 * Stops the active simulation loop.
 *
 * Sets the `running` flag to false, which will cause the background
 * simulation task to exit on its next iteration.
 *
 * @param state Shared application state.
 */
#[tauri::command]
pub fn stop_simulation(state: State<ArenaState>) {
    let mut running = state.running.lock().unwrap();
    *running = false;
}
