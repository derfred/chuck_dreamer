// Package snapshot holds the latest Monitor freeze-frame JPEG in memory
// (Monitor UX: a recent still shown before Stream On, so the operator never
// pays WebRTC bandwidth just to see the scene). The Pi pushes a low-rate JPEG
// to the tailnet-only ingest listener (PUT /snapshot); the public listener
// serves it back (GET /api/snapshot, GET /api/snapshot/meta).
//
// Deliberately NOT persisted to the storage.Backend (disk/MinIO): this is a
// single ephemeral slot, not an artifact — on a webui restart it is simply
// empty until the Pi's next push (at most one push interval later), which is
// a fine trade for skipping disk/MinIO plumbing entirely.
package snapshot

import (
	"sync"
	"time"
)

// Holder mirrors health.Monitor's mutex-guarded-struct idiom: Store (the
// ingest PUT) and Get (the public GET) both take the same lock.
type Holder struct {
	mu         sync.Mutex
	data       []byte
	receivedAt time.Time
}

// Store caches the latest JPEG bytes + arrival time. Called from the ingest
// PUT handler.
func (h *Holder) Store(jpeg []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.data = jpeg
	h.receivedAt = time.Now()
}

// Get returns the cached JPEG bytes, its arrival time, and whether a snapshot
// has arrived yet (false before the first Pi push).
func (h *Holder) Get() (data []byte, receivedAt time.Time, ok bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.data == nil {
		return nil, time.Time{}, false
	}
	return h.data, h.receivedAt, true
}
