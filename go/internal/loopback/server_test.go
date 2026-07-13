package loopback

import (
	"bytes"
	"context"
	"net"
	"testing"
	"time"
)

func TestFrameRoundTrip(t *testing.T) {
	cases := [][]byte{
		{},
		[]byte("tick"),
		bytes.Repeat([]byte{0xAB}, 1024),
	}
	for _, payload := range cases {
		var buf bytes.Buffer
		if err := WriteFrame(&buf, payload); err != nil {
			t.Fatalf("WriteFrame: %v", err)
		}
		got, err := ReadFrame(&buf)
		if err != nil {
			t.Fatalf("ReadFrame: %v", err)
		}
		if !bytes.Equal(got, payload) {
			t.Fatalf("round-trip mismatch: got %v want %v", got, payload)
		}
	}
}

func TestServerAddrIsLoopbackOnly(t *testing.T) {
	if got := NewServer(54321).Addr(); got != "127.0.0.1:54321" {
		t.Fatalf("Addr() = %q, want loopback 127.0.0.1:54321", got)
	}
}

func TestServerBindsAndShutsDown(t *testing.T) {
	// Bind an ephemeral port, confirm it accepts a loopback dial, then cancel.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("probe listen: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	srv := NewServer(port)
	errc := make(chan error, 1)
	go func() { errc <- srv.Run(ctx) }()

	// Give the listener a moment, then dial the loopback bridge.
	var conn net.Conn
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if conn, err = net.Dial("tcp", srv.Addr()); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if conn == nil {
		t.Fatalf("could not dial loopback bridge: %v", err)
	}
	_ = conn.Close()

	cancel()
	if err := <-errc; err != nil {
		t.Fatalf("Run returned error on clean shutdown: %v", err)
	}
}
