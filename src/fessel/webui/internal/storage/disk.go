// Disk storage backend (B5.5.3).
//
// Writes recordings to a mounted directory (a PVC in the cluster); serves
// playback itself as byte-range GETs. The simpler backend operationally, but
// more code, because the bytes flow through the webui on the way out too.
//
// PATH SAFETY is load-bearing: every path is built from a caller-supplied id
// and name, and the playback endpoint is browser-facing (through oauth2-proxy).
// A directory traversal there would read arbitrary pod files. safePath rejects
// any id/name that isn't a single plain path component and confirms the
// resolved path stays strictly under <root>/recordings/.
package storage

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const recordingsDirname = "recordings"

// Monitor freeze-frame: a sibling root to recordings/, a single fixed file —
// deliberately outside the recordings tree so it never surfaces in List().
const (
	monitorDirname   = "monitor"
	snapshotBaseName = "snapshot.jpg"
)

// isPlainComponent reports whether part is a single, safe path component:
// non-empty, not `.`/`..`, no separators, no NUL. The defence against
// id/name traversal.
func isPlainComponent(part string) bool {
	if part == "" || part == "." || part == ".." {
		return false
	}
	return !strings.ContainsAny(part, "/\\\x00")
}

type DiskBackend struct {
	recordingsRoot string
}

func NewDiskBackend(root string) (*DiskBackend, error) {
	abs, err := filepath.Abs(filepath.Join(root, recordingsDirname))
	if err != nil {
		return nil, fmt.Errorf("resolve recordings root: %w", err)
	}
	return &DiskBackend{recordingsRoot: abs}, nil
}

// --- path safety -----------------------------------------------------------

func (d *DiskBackend) typeDir(recType string) string {
	// Only the two known types map to a subdir; anything else falls back to
	// explicit (a stored metadata.json with a junk type can't escape the tree).
	sub := Explicit
	if recType == Anomaly {
		sub = Anomaly
	}
	return filepath.Join(d.recordingsRoot, sub)
}

// safePath builds <root>/recordings/<type>/<id>/<name>, rejecting traversal.
func (d *DiskBackend) safePath(recordingID, fileName, recType string) (string, error) {
	if !isPlainComponent(recordingID) || !isPlainComponent(fileName) {
		return "", ErrInvalidPath
	}
	candidate := filepath.Join(d.typeDir(recType), recordingID, fileName)
	rel, err := filepath.Rel(d.recordingsRoot, candidate)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("%w: path escapes recordings root", ErrInvalidPath)
	}
	return candidate, nil
}

// findExisting locates an already-stored file, searching both type subdirs (a
// reader doesn't know the type up front). Returns "" when absent.
func (d *DiskBackend) findExisting(recordingID, fileName string) string {
	for _, recType := range []string{Explicit, Anomaly} {
		p, err := d.safePath(recordingID, fileName, recType)
		if err != nil {
			return ""
		}
		if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
			return p
		}
	}
	return ""
}

// recTypeOf reports which subdir holds this recording (explicit unless an
// anomaly dir exists), so a file arriving before metadata.json still lands in
// the right tree once the dir is known.
func (d *DiskBackend) recTypeOf(recordingID string) string {
	for _, recType := range []string{Explicit, Anomaly} {
		if st, err := os.Stat(filepath.Join(d.typeDir(recType), recordingID)); err == nil && st.IsDir() {
			return recType
		}
	}
	return DefaultType
}

// --- store -------------------------------------------------------------------

func (d *DiskBackend) Store(recordingID, fileName string, body io.Reader) error {
	// Determine the type subdir. metadata.json is the authoritative source of
	// `type`, so when THIS file is the metadata, buffer it (it's tiny — KB) and
	// read the type out of it. For any other file, use the type from an
	// already-stored metadata.json, else where the recording dir already lives,
	// else explicit/ (the uploader sends metadata.json last, so segments arrive
	// before the type is known).
	var buffered []byte
	var recType string
	if fileName == MetadataFilename {
		raw, err := io.ReadAll(body)
		if err != nil {
			return fmt.Errorf("read metadata body: %w", err)
		}
		buffered = raw
		recType = typeFromMetadataBytes(raw)
		if recType == "" {
			recType = d.recTypeOf(recordingID)
		}
	} else {
		recType = d.readType(recordingID)
	}

	dest, err := d.safePath(recordingID, fileName, recType)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	// Write to a temp sibling then rename -> per-file atomic + idempotent: a
	// retried PUT overwrites cleanly, and a reader never sees a half file.
	tmp := filepath.Join(filepath.Dir(dest), "."+filepath.Base(dest)+".tmp")
	defer os.Remove(tmp)
	fh, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if buffered != nil {
		_, err = fh.Write(buffered)
	} else {
		_, err = io.Copy(fh, body)
	}
	if cerr := fh.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp, dest)
}

// readType is the type for a *store* call: prefer the type recorded in an
// already-stored metadata.json, else where the recording dir already lives,
// else default.
func (d *DiskBackend) readType(recordingID string) string {
	if metaPath := d.findExisting(recordingID, MetadataFilename); metaPath != "" {
		if meta := loadJSONMap(metaPath); meta != nil {
			if t, ok := meta["type"].(string); ok && (t == Explicit || t == Anomaly) {
				return t
			}
		}
	}
	return d.recTypeOf(recordingID)
}

// --- exists / list -----------------------------------------------------------

func (d *DiskBackend) Exists(recordingID string) bool {
	if !isPlainComponent(recordingID) {
		return false
	}
	for _, recType := range []string{Explicit, Anomaly} {
		entries, err := os.ReadDir(filepath.Join(d.typeDir(recType), recordingID))
		if err == nil && len(entries) > 0 {
			return true
		}
	}
	return false
}

func (d *DiskBackend) List() []RecordingView {
	views := []RecordingView{}
	for _, recType := range []string{Explicit, Anomaly} {
		base := d.typeDir(recType)
		children, err := os.ReadDir(base)
		if err != nil {
			continue
		}
		for _, child := range children {
			if !child.IsDir() {
				continue
			}
			dir := filepath.Join(base, child.Name())
			entries, err := os.ReadDir(dir)
			if err != nil {
				continue
			}
			var files []string
			for _, e := range entries {
				if e.Type().IsRegular() {
					files = append(files, e.Name())
				}
			}
			sort.Strings(files)
			views = append(views, RecordingView{
				RecordingID: child.Name(),
				RecType:     recType,
				Files:       files,
				Metadata:    loadJSONMap(filepath.Join(dir, MetadataFilename)),
			})
		}
	}
	// Newest first; recordings without metadata.started_at sort last.
	sort.SliceStable(views, func(i, j int) bool { return views[i].StartedAt() > views[j].StartedAt() })
	return views
}

// --- playback (served by the webui) -------------------------------------------

func (d *DiskBackend) PlaybackURL(recordingID, fileName string) (PlaybackTarget, error) {
	if !isPlainComponent(recordingID) || !isPlainComponent(fileName) {
		return nil, ErrInvalidPath
	}
	if d.findExisting(recordingID, fileName) == "" {
		return nil, nil
	}
	return ServeLocally{RecordingID: recordingID, FileName: fileName}, nil
}

func (d *DiskBackend) Read(recordingID, fileName, httpRange string) (*ReadResult, error) {
	path := d.findExisting(recordingID, fileName)
	if path == "" {
		return nil, nil
	}
	st, err := os.Stat(path)
	if err != nil {
		return nil, nil
	}
	total := st.Size()
	byteRange, err := ParseByteRange(httpRange, total)
	if err != nil {
		return nil, err // ErrRangeNotSatisfiable -> 416
	}
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if byteRange == nil {
		return &ReadResult{
			Body:          fh,
			ContentType:   ContentTypeFor(fileName),
			ContentLength: total,
		}, nil
	}
	if _, err := fh.Seek(byteRange.Start, io.SeekStart); err != nil {
		fh.Close()
		return nil, err
	}
	return &ReadResult{
		Body:          limitReadCloser{Reader: io.LimitReader(fh, byteRange.Length()), Closer: fh},
		ContentType:   ContentTypeFor(fileName),
		ContentLength: byteRange.Length(),
		ByteRange:     byteRange,
	}, nil
}

type limitReadCloser struct {
	io.Reader
	io.Closer
}

func (d *DiskBackend) snapshotPath() string {
	return filepath.Join(filepath.Dir(d.recordingsRoot), monitorDirname, snapshotBaseName)
}

func (d *DiskBackend) StoreSnapshot(jpeg []byte, capturedAt time.Time) error {
	dest := d.snapshotPath()
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	// Write to a temp sibling then rename, same idempotent/atomic pattern as
	// Store: a reader never sees a half-written frame.
	tmp := dest + ".tmp"
	if err := os.WriteFile(tmp, jpeg, 0o644); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dest); err != nil {
		return err
	}
	// The file's mtime IS the capture time (not the write time): that keeps the
	// timestamp intrinsic to the stored object — no side file to lose, and a
	// snapshot from before this was tracked still reads back sensibly. A
	// failure here is not fatal; the frame is stored, only its age degrades to
	// "when it was written".
	if !capturedAt.IsZero() {
		_ = os.Chtimes(dest, capturedAt, capturedAt)
	}
	return nil
}

func (d *DiskBackend) ReadSnapshot() ([]byte, time.Time, bool) {
	path := d.snapshotPath()
	st, err := os.Stat(path)
	if err != nil {
		return nil, time.Time{}, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, time.Time{}, false
	}
	return data, st.ModTime(), true
}

// typeFromMetadataBytes parses the `type` out of a metadata.json byte buffer,
// or "" if it is absent / unparseable / not one of the known types.
func typeFromMetadataBytes(raw []byte) string {
	var meta map[string]any
	if err := json.Unmarshal(raw, &meta); err != nil {
		return ""
	}
	if t, ok := meta["type"].(string); ok && (t == Explicit || t == Anomaly) {
		return t
	}
	return ""
}

func loadJSONMap(path string) map[string]any {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var meta map[string]any
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil
	}
	return meta
}

// --- delete ------------------------------------------------------------------

// Delete removes the recording's directory from this store (B5.5.x operator
// cleanup). Path safety matters as much as it does for playback: the id is
// browser-supplied and this call REMOVES a tree, so the id must be a single
// plain component and the resolved dir must sit strictly under the recordings
// root — the same guard safePath applies, reused here via a per-type dir join.
func (d *DiskBackend) Delete(recordingID string) (bool, error) {
	if !isPlainComponent(recordingID) {
		return false, ErrInvalidPath
	}
	found := false
	for _, recType := range []string{Explicit, Anomaly} {
		dir := filepath.Join(d.typeDir(recType), recordingID)
		// Re-check containment explicitly rather than trusting the join.
		rel, err := filepath.Rel(d.recordingsRoot, dir)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return false, fmt.Errorf("%w: path escapes recordings root", ErrInvalidPath)
		}
		st, err := os.Stat(dir)
		if err != nil || !st.IsDir() {
			continue
		}
		if err := os.RemoveAll(dir); err != nil {
			return false, fmt.Errorf("delete recording %s: %w", recordingID, err)
		}
		found = true
	}
	return found, nil
}
