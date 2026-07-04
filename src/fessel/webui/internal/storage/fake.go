// FakeBackend is the in-memory test double, mirroring the Python
// FakeStorageBackend seam: tests exercise handlers without MinIO or a tempdir.
package storage

import (
	"bytes"
	"encoding/json"
	"io"
	"sort"
	"sync"
)

type FakeBackend struct {
	mu sync.Mutex
	// files[recordingID][fileName] = bytes
	files map[string]map[string][]byte
	types map[string]string

	// PresignBase, when set, makes PlaybackURL return PresignedURL like the
	// MinIO backend; empty means ServeLocally like the disk backend.
	PresignBase string
	// StoreErr, when set, makes Store fail (the 502 ingest path).
	StoreErr error
}

func NewFakeBackend() *FakeBackend {
	return &FakeBackend{files: map[string]map[string][]byte{}, types: map[string]string{}}
}

func (f *FakeBackend) Store(recordingID, fileName string, body io.Reader) error {
	if f.StoreErr != nil {
		return f.StoreErr
	}
	if !isPlainComponent(recordingID) || !isPlainComponent(fileName) {
		return ErrInvalidPath
	}
	raw, err := io.ReadAll(body)
	if err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.files[recordingID] == nil {
		f.files[recordingID] = map[string][]byte{}
	}
	f.files[recordingID][fileName] = raw
	if fileName == MetadataFilename {
		if t := typeFromMetadataBytes(raw); t != "" {
			f.types[recordingID] = t
		}
	}
	return nil
}

func (f *FakeBackend) Exists(recordingID string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.files[recordingID]) > 0
}

func (f *FakeBackend) List() []RecordingView {
	f.mu.Lock()
	defer f.mu.Unlock()
	views := []RecordingView{}
	for rid, files := range f.files {
		var names []string
		for name := range files {
			names = append(names, name)
		}
		sort.Strings(names)
		var meta map[string]any
		if raw, ok := files[MetadataFilename]; ok {
			_ = json.Unmarshal(raw, &meta)
		}
		recType := f.types[rid]
		if recType == "" {
			recType = DefaultType
		}
		views = append(views, RecordingView{RecordingID: rid, RecType: recType, Files: names, Metadata: meta})
	}
	sort.SliceStable(views, func(i, j int) bool { return views[i].StartedAt() > views[j].StartedAt() })
	return views
}

func (f *FakeBackend) PlaybackURL(recordingID, fileName string) (PlaybackTarget, error) {
	if !isPlainComponent(recordingID) || !isPlainComponent(fileName) {
		return nil, ErrInvalidPath
	}
	f.mu.Lock()
	_, ok := f.files[recordingID][fileName]
	f.mu.Unlock()
	if !ok {
		return nil, nil
	}
	if f.PresignBase != "" {
		return PresignedURL{URL: f.PresignBase + "/" + recordingID + "/" + fileName}, nil
	}
	return ServeLocally{RecordingID: recordingID, FileName: fileName}, nil
}

func (f *FakeBackend) Read(recordingID, fileName, httpRange string) (*ReadResult, error) {
	f.mu.Lock()
	raw, ok := f.files[recordingID][fileName]
	f.mu.Unlock()
	if !ok {
		return nil, nil
	}
	total := int64(len(raw))
	byteRange, err := ParseByteRange(httpRange, total)
	if err != nil {
		return nil, err
	}
	if byteRange == nil {
		return &ReadResult{
			Body:          io.NopCloser(bytes.NewReader(raw)),
			ContentType:   ContentTypeFor(fileName),
			ContentLength: total,
		}, nil
	}
	return &ReadResult{
		Body:          io.NopCloser(bytes.NewReader(raw[byteRange.Start : byteRange.End+1])),
		ContentType:   ContentTypeFor(fileName),
		ContentLength: byteRange.Length(),
		ByteRange:     byteRange,
	}, nil
}

// Bytes exposes stored content for test assertions.
func (f *FakeBackend) Bytes(recordingID, fileName string) ([]byte, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	raw, ok := f.files[recordingID][fileName]
	return raw, ok
}
