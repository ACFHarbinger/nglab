// Package rpc holds JSON-RPC node connections used for crypto balances, nonces,
// and chain state (migrated out of Rust). Concurrent, reconnecting clients land
// here; kept as a typed stub until the connectors are ported.
//
// See moon/roadmaps/crypto_go.md §1.
package rpc

// NodeEndpoint identifies a JSON-RPC node connection.
type NodeEndpoint struct {
	Name string
	URL  string // ws:// or https:// JSON-RPC endpoint
}
