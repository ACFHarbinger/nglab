#!/bin/bash
# Script to apply all remaining clippy fixes

set -e

cd "$(dirname "$0")/rust"

echo "Applying clippy fixes..."

# Fix models/rough_heston.rs - manual_clamp
sed -i 's/params\.steps\.max(16)\.min(512)/params.steps.clamp(16, 512)/g' src/models/rough_heston.rs
sed -i 's/params\.paths\.max(100)\.min(10_000)/params.paths.clamp(100, 10_000)/g' src/models/rough_heston.rs

# Fix moon/arima.rs - needless_range_loop and manual_memcpy
# These require more complex changes, will do manually

# Fix moon/es.rs - needless_borrow
sed -i 's/initialize_from_data(&data,/initialize_from_data(data,/g' src/moon/es.rs

# Fix moon/garch.rs - manual_memcpy (will do manually)

# Fix secret/vault.rs - needless_borrows_for_generic_args
sed -i 's/pragma_update(None, "key", &key)/pragma_update(None, "key", key)/g' src/secret/vault.rs

# Fix simulation/risk.rs - manual_clamp
sed -i 's/(daily_ratio \* 30\.0)\.min(30\.0)\.max(0\.0)/(daily_ratio * 30.0).clamp(0.0, 30.0)/g' src/simulation/risk.rs

# Fix utils/memory.rs - manual_map (will do manually)

# Fix web/polymarket.rs - wrong_self_convention (will do manually)

# Fix web/streaming.rs - redundant_pattern_matching
sed -i 's/if let Err(_) = ping_tx\.send(Message::Text("PING"\.into()))\.await/if (ping_tx.send(Message::Text("PING".into())).await).is_err()/g' src/web/streaming.rs

echo "Simple fixes applied. Manual fixes still needed for complex cases."
