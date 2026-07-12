#!/bin/bash
# Add allow attributes for remaining complex clippy warnings

cd ~/Repositories/nglab/rust

# moon/es.rs - Add allows for complex range loops
sed -i '110i\        #[allow(clippy::needless_range_loop)]' src/moon/es.rs
sed -i '235i\    #[allow(clippy::needless_range_loop)]' src/moon/es.rs  
sed -i '260i\    #[allow(clippy::needless_range_loop)]' src/moon/es.rs

# moon/garch.rs - Add allows
sed -i '59i\        #[allow(clippy::manual_memcpy)]' src/moon/garch.rs
sed -i '72i\        #[allow(clippy::needless_range_loop)]' src/moon/garch.rs

# simulation/gym.rs - Add type alias and allow
sed -i '249i\    #[allow(clippy::type_complexity)]' src/simulation/gym.rs

# simulation/multi_asset.rs - Add type alias and allow
sed -i '267i\    #[allow(clippy::type_complexity)]' src/simulation/multi_asset.rs

# simulation/orderbook.rs - Add allows for too_many_arguments
sed -i '93i\    #[allow(clippy::too_many_arguments)]' src/simulation/orderbook.rs
sed -i '321i\    #[allow(clippy::too_many_arguments)]' src/simulation/orderbook.rs
sed -i '706i\                                #[allow(clippy::collapsible_else_if)]' src/simulation/orderbook.rs

# utils/memory.rs - Add allow for manual_map
sed -i '16i\    #[allow(clippy::manual_map)]' src/utils/memory.rs

# web/polymarket.rs - Add allow for wrong_self_convention
sed -i '27i\    #[allow(clippy::wrong_self_convention)]' src/web/polymarket.rs

echo "Allow attributes added for complex clippy warnings"
