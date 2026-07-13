// Command crypto-daemon is the headless NGLab Crypto Daemon (Tier 2 / Warm Path).
//
// It handles I/O-bound work and massive concurrency — thousands of exchange
// WebSocket feeds and JSON-RPC node connections — and streams fast market ticks
// and account/balance state to the Rust hub over a local loopback bridge.
//
// The Rust backend supplies a dynamic port at startup so the daemon never binds
// a privileged/fixed port:
//
//	crypto-daemon --port=54321
//
// The server binds 127.0.0.1:<port> only (never 0.0.0.0).
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/ACFHarbinger/nglab/go/internal/loopback"
)

func main() {
	port := flag.Int("port", 0, "loopback TCP port supplied by the Rust backend (required)")
	flag.Parse()

	if *port <= 0 {
		log.Fatal("crypto-daemon: --port is required (the Rust backend supplies a dynamic port)")
	}

	// Graceful shutdown on SIGINT/SIGTERM so the Rust lifecycle manager can
	// restart the daemon cleanly.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	srv := loopback.NewServer(*port)
	log.Printf("crypto-daemon: starting loopback bridge on 127.0.0.1:%d", *port)

	if err := srv.Run(ctx); err != nil {
		log.Printf("crypto-daemon: exited with error: %v", err)
		os.Exit(1)
	}
	log.Print("crypto-daemon: shut down cleanly")
}
