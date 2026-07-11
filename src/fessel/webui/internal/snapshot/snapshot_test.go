package snapshot

import "testing"

func TestHolderGetBeforeStoreIsNotOK(t *testing.T) {
	h := &Holder{}
	data, _, ok := h.Get()
	if ok || data != nil {
		t.Fatalf("want not-ok/nil before first Store, got ok=%v data=%q", ok, data)
	}
}

func TestHolderStoreThenGet(t *testing.T) {
	h := &Holder{}
	h.Store([]byte("jpeg-bytes"))
	data, receivedAt, ok := h.Get()
	if !ok || string(data) != "jpeg-bytes" {
		t.Fatalf("want ok/jpeg-bytes, got ok=%v data=%q", ok, data)
	}
	if receivedAt.IsZero() {
		t.Fatalf("want a non-zero receivedAt")
	}
}

func TestHolderStoreOverwrites(t *testing.T) {
	h := &Holder{}
	h.Store([]byte("one"))
	h.Store([]byte("two"))
	data, _, ok := h.Get()
	if !ok || string(data) != "two" {
		t.Fatalf("want ok/two, got ok=%v data=%q", ok, data)
	}
}
