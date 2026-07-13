// Package loopback implements the local IPC bridge from the Go Crypto Daemon to
// the Rust backend.
//
// Transport rule (see moon/roadmaps/crypto_go.md §2): a loopback TCP server on
// 127.0.0.1:<dynamic port>. Frames are length-prefixed (4-byte big-endian
// length + payload). The payload will carry serialized Protobuf Tick / balance
// messages once the schema lands; for now it is an opaque byte slice so the
// framing can be tested independently.
package loopback

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
)

// Server is the loopback bridge. It binds a single 127.0.0.1 port and streams
// length-prefixed frames to the connected Rust reader.
type Server struct {
	port int
}

// NewServer returns a loopback Server that will bind 127.0.0.1:port.
func NewServer(port int) *Server {
	return &Server{port: port}
}

// Addr is the loopback address the server binds. Never 0.0.0.0.
func (s *Server) Addr() string {
	return fmt.Sprintf("127.0.0.1:%d", s.port)
}

// Run binds the loopback listener and serves until ctx is cancelled.
func (s *Server) Run(ctx context.Context) error {
	var lc net.ListenConfig
	ln, err := lc.Listen(ctx, "tcp", s.Addr())
	if err != nil {
		return fmt.Errorf("loopback: bind %s: %w", s.Addr(), err)
	}
	defer ln.Close()

	// Unblock Accept on shutdown.
	go func() {
		<-ctx.Done()
		_ = ln.Close()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil // clean shutdown
			}
			return fmt.Errorf("loopback: accept: %w", err)
		}
		go s.handle(ctx, conn)
	}
}

// handle streams frames to a single Rust reader. Real feeds fan in here.
func (s *Server) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	<-ctx.Done()
}

// WriteFrame writes a single length-prefixed frame (4-byte big-endian length +
// payload) to w. This is the wire format the Rust reader consumes.
func WriteFrame(w io.Writer, payload []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// ReadFrame reads a single length-prefixed frame written by WriteFrame.
func ReadFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}
