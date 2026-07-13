// NGLab Crypto Daemon (Tier 2 / Warm Path).
//
// All crypto market-data, exchange-connection, and concurrent trading logic
// lives here (migrated out of Rust). Kept stdlib-only for the scaffold; add
// exchange SDKs / protobuf runtime as connectors land.
module github.com/ACFHarbinger/nglab/go

go 1.26
