// Package feeds fans in market-data streams from many exchange connections.
//
// Each exchange connector runs in its own goroutine (one per WebSocket stream)
// and pushes normalized ticks into a bounded channel. The dispatcher drains
// them toward the loopback bridge, dropping the oldest frame on overflow (with a
// counter) so a slow reader never blocks the hot path. Concrete connectors land
// under internal/feeds/<exchange>/ — see moon/roadmaps/crypto_go.md and
// .agent/skills/add-crypto-feed.md.
package feeds

// Tick is a placeholder for the Protobuf-generated market tick (schema in
// proto/tick.proto). Connectors normalize raw exchange messages into this type.
type Tick struct {
	Symbol  string
	Price   float64
	Size    float64
	TsNanos int64
}

// Dispatcher fans ticks from N feeds into a single bounded output channel.
type Dispatcher struct {
	out     chan Tick
	dropped uint64
}

// NewDispatcher creates a dispatcher with the given output buffer capacity.
func NewDispatcher(capacity int) *Dispatcher {
	return &Dispatcher{out: make(chan Tick, capacity)}
}

// Publish enqueues a tick without ever blocking: if the buffer is full the
// oldest frame is dropped and the dropped counter is incremented.
func (d *Dispatcher) Publish(t Tick) {
	select {
	case d.out <- t:
	default:
		select {
		case <-d.out: // drop oldest
			d.dropped++
		default:
		}
		select {
		case d.out <- t:
		default:
			d.dropped++
		}
	}
}

// Out is the drain side consumed by the loopback bridge.
func (d *Dispatcher) Out() <-chan Tick { return d.out }

// Dropped reports how many ticks were dropped due to back-pressure.
func (d *Dispatcher) Dropped() uint64 { return d.dropped }
