package storage

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newDisk(t *testing.T) *DiskBackend {
	t.Helper()
	d, err := NewDiskBackend(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestDiskStoreAndReadBack(t *testing.T) {
	d := newDisk(t)
	if err := d.Store("rec1", "seg-00001.ts", strings.NewReader("segment-bytes")); err != nil {
		t.Fatal(err)
	}
	res, err := d.Read("rec1", "seg-00001.ts", "")
	if err != nil || res == nil {
		t.Fatalf("read: %v %v", res, err)
	}
	defer res.Body.Close()
	got, _ := io.ReadAll(res.Body)
	if string(got) != "segment-bytes" {
		t.Fatalf("got %q", got)
	}
	if res.ContentType != "video/mp2t" {
		t.Fatalf("content type %q", res.ContentType)
	}
	if res.ContentLength != int64(len("segment-bytes")) {
		t.Fatalf("length %d", res.ContentLength)
	}
}

func TestDiskStoreIsIdempotent(t *testing.T) {
	d := newDisk(t)
	for _, content := range []string{"first", "second-longer"} {
		if err := d.Store("rec1", "index.m3u8", strings.NewReader(content)); err != nil {
			t.Fatal(err)
		}
	}
	res, _ := d.Read("rec1", "index.m3u8", "")
	got, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if string(got) != "second-longer" {
		t.Fatalf("got %q", got)
	}
}

func TestDiskRejectsTraversal(t *testing.T) {
	d := newDisk(t)
	bad := [][2]string{
		{"../evil", "index.m3u8"},
		{"rec1", "../../etc/passwd"},
		{".", "x"},
		{"rec1", ".."},
		{"a/b", "x"},
		{"rec1", "x\\y"},
		{"", "x"},
	}
	for _, pair := range bad {
		if err := d.Store(pair[0], pair[1], strings.NewReader("x")); err == nil {
			t.Fatalf("store accepted %q/%q", pair[0], pair[1])
		}
		if _, err := d.PlaybackURL(pair[0], pair[1]); err == nil {
			t.Fatalf("playback accepted %q/%q", pair[0], pair[1])
		}
	}
}

func TestDiskMetadataTypeRouting(t *testing.T) {
	d := newDisk(t)
	// Segment arrives before metadata -> lands under explicit/.
	if err := d.Store("recA", "seg-00001.ts", strings.NewReader("s")); err != nil {
		t.Fatal(err)
	}
	// metadata.json declaring anomaly routes ITSELF to anomaly/ (the recording
	// dir split across types is tolerated; readers search both).
	meta := `{"type": "anomaly", "started_at": "2026-07-01T00:00:00Z"}`
	if err := d.Store("recA", "metadata.json", strings.NewReader(meta)); err != nil {
		t.Fatal(err)
	}
	views := d.List()
	found := false
	for _, v := range views {
		if v.RecordingID == "recA" && v.RecType == Anomaly && v.Metadata != nil {
			found = true
		}
	}
	if !found {
		t.Fatalf("anomaly view not found: %+v", views)
	}
	// Subsequent files follow the stored metadata's type.
	if err := d.Store("recA", "seg-00002.ts", strings.NewReader("s2")); err != nil {
		t.Fatal(err)
	}
	if d.findExisting("recA", "seg-00002.ts") == "" {
		t.Fatal("seg-00002 not found")
	}
	if !strings.Contains(d.findExisting("recA", "seg-00002.ts"), string(filepath.Separator)+Anomaly+string(filepath.Separator)) {
		t.Fatalf("seg-00002 not under anomaly/: %s", d.findExisting("recA", "seg-00002.ts"))
	}
}

func TestDiskSnapshotStoreAndReadBack(t *testing.T) {
	d := newDisk(t)
	if _, _, ok := d.ReadSnapshot(); ok {
		t.Fatalf("want no snapshot before first store")
	}
	// The capture time is carried by the file's own mtime, so it survives a
	// webui restart with no side file: a frame captured well before it was
	// written must read back with the CAPTURE time, not the write time.
	capturedAt := time.Now().Add(-90 * time.Second).Truncate(time.Second)
	if err := d.StoreSnapshot([]byte("jpeg-one"), capturedAt); err != nil {
		t.Fatal(err)
	}
	data, gotAt, ok := d.ReadSnapshot()
	if !ok || string(data) != "jpeg-one" {
		t.Fatalf("stored: %q %v", data, ok)
	}
	if !gotAt.Equal(capturedAt) {
		t.Fatalf("capturedAt %v, want %v", gotAt, capturedAt)
	}
}

// A zero capture time (a backend called without one) leaves the write time in
// place rather than stamping the epoch onto the file.
func TestDiskSnapshotZeroCapturedAtFallsBackToWriteTime(t *testing.T) {
	d := newDisk(t)
	before := time.Now().Add(-time.Second)
	if err := d.StoreSnapshot([]byte("jpeg"), time.Time{}); err != nil {
		t.Fatal(err)
	}
	_, gotAt, ok := d.ReadSnapshot()
	if !ok || gotAt.Before(before) {
		t.Fatalf("capturedAt %v, want >= %v", gotAt, before)
	}
}

func TestDiskSnapshotOverwrites(t *testing.T) {
	d := newDisk(t)
	_ = d.StoreSnapshot([]byte("one"), time.Now())
	_ = d.StoreSnapshot([]byte("two"), time.Now())
	data, _, ok := d.ReadSnapshot()
	if !ok || string(data) != "two" {
		t.Fatalf("stored: %q %v", data, ok)
	}
}

func TestDiskSnapshotDoesNotAppearInRecordingsList(t *testing.T) {
	d := newDisk(t)
	_ = d.Store("rec1", "seg-00001.ts", strings.NewReader("x"))
	_ = d.StoreSnapshot([]byte("jpeg"), time.Now())
	views := d.List()
	for _, v := range views {
		if v.RecordingID == "monitor" || v.RecordingID == "snapshot.jpg" {
			t.Fatalf("snapshot leaked into recordings list: %v", views)
		}
	}
	if len(views) != 1 || views[0].RecordingID != "rec1" {
		t.Fatalf("want only rec1, got %v", views)
	}
}

func TestDiskListNewestFirst(t *testing.T) {
	d := newDisk(t)
	_ = d.Store("old", "metadata.json", strings.NewReader(`{"started_at":"2026-01-01T00:00:00Z"}`))
	_ = d.Store("new", "metadata.json", strings.NewReader(`{"started_at":"2026-06-01T00:00:00Z"}`))
	_ = d.Store("nometa", "seg-00001.ts", strings.NewReader("x"))
	views := d.List()
	if len(views) != 3 {
		t.Fatalf("want 3, got %d", len(views))
	}
	if views[0].RecordingID != "new" || views[1].RecordingID != "old" || views[2].RecordingID != "nometa" {
		t.Fatalf("order: %s %s %s", views[0].RecordingID, views[1].RecordingID, views[2].RecordingID)
	}
}

func TestDiskRangeReads(t *testing.T) {
	d := newDisk(t)
	content := "0123456789"
	_ = d.Store("rec1", "seg-00001.ts", strings.NewReader(content))

	res, err := d.Read("rec1", "seg-00001.ts", "bytes=2-5")
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if string(got) != "2345" || res.ByteRange == nil || res.ByteRange.Start != 2 || res.ByteRange.End != 5 {
		t.Fatalf("range read: %q %+v", got, res.ByteRange)
	}

	// Suffix range.
	res, err = d.Read("rec1", "seg-00001.ts", "bytes=-3")
	if err != nil {
		t.Fatal(err)
	}
	got, _ = io.ReadAll(res.Body)
	res.Body.Close()
	if string(got) != "789" {
		t.Fatalf("suffix range: %q", got)
	}

	// Open-ended range clamps to EOF.
	res, err = d.Read("rec1", "seg-00001.ts", "bytes=8-")
	if err != nil {
		t.Fatal(err)
	}
	got, _ = io.ReadAll(res.Body)
	res.Body.Close()
	if string(got) != "89" {
		t.Fatalf("open range: %q", got)
	}

	// Unsatisfiable / malformed -> ErrRangeNotSatisfiable.
	for _, hdr := range []string{"bytes=10-12", "bytes=5-2", "items=0-1", "bytes=0-1,3-4", "bytes=-0"} {
		if _, err := d.Read("rec1", "seg-00001.ts", hdr); err == nil {
			t.Fatalf("accepted %q", hdr)
		}
	}

	// Absent file -> nil, nil.
	res, err = d.Read("rec1", "nope.ts", "")
	if res != nil || err != nil {
		t.Fatalf("absent: %v %v", res, err)
	}
}

func TestDiskAtomicNoTempLeftover(t *testing.T) {
	dir := t.TempDir()
	d, _ := NewDiskBackend(dir)
	_ = d.Store("rec1", "index.m3u8", strings.NewReader("x"))
	var leftovers []string
	_ = filepath.WalkDir(dir, func(path string, e os.DirEntry, err error) error {
		if err == nil && !e.IsDir() && strings.HasSuffix(e.Name(), ".tmp") {
			leftovers = append(leftovers, path)
		}
		return nil
	})
	if len(leftovers) != 0 {
		t.Fatalf("temp files left: %v", leftovers)
	}
}

func TestDiskExists(t *testing.T) {
	d := newDisk(t)
	if d.Exists("rec1") {
		t.Fatal("exists before store")
	}
	_ = d.Store("rec1", "seg-00001.ts", bytes.NewReader([]byte("x")))
	if !d.Exists("rec1") {
		t.Fatal("missing after store")
	}
	if d.Exists("../evil") {
		t.Fatal("traversal in exists")
	}
}

func TestDiskPlaybackTargets(t *testing.T) {
	d := newDisk(t)
	_ = d.Store("rec1", "index.m3u8", strings.NewReader("x"))
	target, err := d.PlaybackURL("rec1", "index.m3u8")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := target.(ServeLocally); !ok {
		t.Fatalf("want ServeLocally, got %T", target)
	}
	target, err = d.PlaybackURL("rec1", "missing.ts")
	if err != nil || target != nil {
		t.Fatalf("missing file: %v %v", target, err)
	}
}
