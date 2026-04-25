package tunnel

import (
	"sync"

	"github.com/awnumar/memguard"
)

const (
	// PacketBufferSize is sized for standard Ethernet MTU (1500) + WireGuard/QUIC overhead
	PacketBufferSize = 2048
)

// EnclavePool manages a set of hardware-locked buffers to prevent heap leakage.
type EnclavePool struct {
	pool sync.Pool
}

// NewEnclavePool initializes a pool of locked memory buffers.
func NewEnclavePool() *EnclavePool {
	return &EnclavePool{
		pool: sync.Pool{
			New: func() interface{} {
				// Allocate a new locked buffer from memguard
				return memguard.NewBuffer(PacketBufferSize)
			},
		},
	}
}

// Get retrieves a locked buffer from the pool.
func (p *EnclavePool) Get() *memguard.LockedBuffer {
	return p.pool.Get().(*memguard.LockedBuffer)
}

// Put wipes the buffer and returns it to the pool.
func (p *EnclavePool) Put(b *memguard.LockedBuffer) {
	if b == nil {
		return
	}
	// Deterministically zero out the memory before reuse
	b.Wipe()
	p.pool.Put(b)
}

// GlobalPool is the shared memory arena for all tunnel operations.
var GlobalPool = NewEnclavePool()
