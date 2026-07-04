// Package storage is the recording-store abstraction (B5.5.1).
//
// Two implementations:
//   - MinIO (minio.go): S3-compatible object store. Playback is a
//     presigned-GET redirect — the browser fetches segments straight from
//     MinIO, the webui never touches the bytes.
//   - disk (disk.go): a mounted directory (PVC). Playback is served by the
//     webui itself as byte-range GETs (Read()).
//
// The on-store layout is identical for both backends (architecture §4.3):
//
//	recordings/<type>/<id>/{index.m3u8, seg-NNNNN.ts, metadata.json}
//
// so a switch from one to the other is a copy, not a reformat.
//
// Interface contract (enforced by the implementations + their tests):
//   - Store STREAMS its reader to the store incrementally — a 200 MB segment
//     must not buffer fully in memory.
//   - Store is IDEMPOTENT: a repeated PUT for the same (id, name) overwrites
//     cleanly, which is what makes the Pi-side per-file retry safe.
//   - Store is per-file ATOMIC: a call either completes the whole file or
//     leaves the previous version (or absence) intact.
//   - There are NO cross-recording transactions: a recording is "complete"
//     only when all of its files are present; partial completeness is
//     observable and consumers (list/playback) tolerate it.
package storage

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

const (
	Explicit    = "explicit"
	Anomaly     = "anomaly"
	DefaultType = Explicit

	PlaylistFilename = "index.m3u8"
	MetadataFilename = "metadata.json"
)

// ErrInvalidPath marks a caller-supplied recording id / file name that failed
// path safety. Handlers map it to 400.
var ErrInvalidPath = errors.New("invalid recording id or file name")

// RecordingView is what List() reports per recording: the id, the files
// present, and the parsed metadata.json (nil if absent / unparseable — a
// partial upload).
type RecordingView struct {
	RecordingID string
	RecType     string
	Files       []string
	Metadata    map[string]any // parsed metadata.json; nil until it lands
}

// StartedAt is the sort key for List() (newest first). Missing metadata
// sorts last.
func (v RecordingView) StartedAt() string {
	if v.Metadata != nil {
		if s, ok := v.Metadata["started_at"].(string); ok {
			return s
		}
	}
	return ""
}

// --- playback targets --------------------------------------------------------
// PlaybackURL() returns one of these; the playback handler (B5.5.4) renders a
// 302 redirect for PresignedURL (MinIO) or a 200/206 byte stream for
// ServeLocally (disk). A uniform front, backend-specific behaviour.

type PlaybackTarget interface{ isPlaybackTarget() }

// PresignedURL — MinIO: a short-TTL presigned GET URL; 302-redirect to it.
type PresignedURL struct{ URL string }

// ServeLocally — disk: the file is read from the store and streamed (Range).
type ServeLocally struct {
	RecordingID string
	FileName    string
}

func (PresignedURL) isPlaybackTarget() {}
func (ServeLocally) isPlaybackTarget() {}

// ByteRange is a parsed single-range HTTP Range request resolved against a
// known size. Start/End are inclusive byte offsets; Total is the file size.
type ByteRange struct {
	Start, End, Total int64
}

func (r ByteRange) Length() int64 { return r.End - r.Start + 1 }

// ReadResult is a disk-backend Read(): the byte stream plus what the handler
// needs to build the response — content length, content type, and (for a
// Range request) the resolved range so it can emit 206 + Content-Range.
type ReadResult struct {
	Body          io.ReadCloser
	ContentType   string
	ContentLength int64
	ByteRange     *ByteRange // set only for a Range request (206)
}

// Backend is the abstraction the rest of the webui depends on (B5.5.1).
type Backend interface {
	// Store persists body under (recordingID, fileName), streaming. Idempotent,
	// per-file atomic. The type subdir is read from a previously-stored
	// metadata.json when available, else DefaultType. Returns ErrInvalidPath
	// for a bad id/name.
	Store(recordingID, fileName string, body io.Reader) error
	// Exists reports whether any file for this recording is present.
	Exists(recordingID string) bool
	// List enumerates all recordings, newest first, with parsed metadata.
	List() []RecordingView
	// PlaybackURL returns a PlaybackTarget for one file, nil if it does not
	// exist, or ErrInvalidPath.
	PlaybackURL(recordingID, fileName string) (PlaybackTarget, error)
	// Read is disk-only: open the file and stream it (honouring an HTTP Range
	// header). Returns (nil, nil) if the file is absent, ErrRangeNotSatisfiable
	// on a malformed/unsatisfiable range. The MinIO backend returns an error —
	// its playback is via redirect, never through the webui.
	Read(recordingID, fileName, httpRange string) (*ReadResult, error)
}

// ErrRangeNotSatisfiable marks a malformed or unsatisfiable Range header.
// Handlers map it to 416.
var ErrRangeNotSatisfiable = errors.New("range not satisfiable")

// --- shared helpers used by more than one backend ----------------------------

// ContentTypeFor is the HLS-aware Content-Type. hls.js + Safari need the right
// type on .m3u8 and .ts or playback silently fails (F5.5.1 / B5.5.4).
func ContentTypeFor(name string) string {
	switch {
	case strings.HasSuffix(name, ".m3u8"):
		return "application/vnd.apple.mpegurl"
	case strings.HasSuffix(name, ".ts"):
		return "video/mp2t"
	case strings.HasSuffix(name, ".json"):
		return "application/json"
	default:
		return "application/octet-stream"
	}
}

// ParseByteRange parses a single `bytes=start-end` Range header against a
// known size. Returns (nil, nil) when there is no Range header, and
// ErrRangeNotSatisfiable on a malformed or unsatisfiable range. Only the
// single-range form HLS players use is supported.
func ParseByteRange(header string, total int64) (*ByteRange, error) {
	if header == "" {
		return nil, nil
	}
	spec := strings.TrimSpace(header)
	if !strings.HasPrefix(strings.ToLower(spec), "bytes=") {
		return nil, fmt.Errorf("%w: unsupported range unit %q", ErrRangeNotSatisfiable, header)
	}
	spec = spec[len("bytes="):]
	if strings.Contains(spec, ",") {
		return nil, fmt.Errorf("%w: multi-range requests are not supported", ErrRangeNotSatisfiable)
	}
	startS, endS, found := strings.Cut(spec, "-")
	if !found {
		return nil, fmt.Errorf("%w: invalid range %q", ErrRangeNotSatisfiable, header)
	}
	var start, end int64
	if startS == "" {
		// Suffix range: bytes=-N -> the last N bytes.
		n, err := strconv.ParseInt(endS, 10, 64)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%w: invalid suffix range %q", ErrRangeNotSatisfiable, header)
		}
		start = max(0, total-n)
		end = total - 1
	} else {
		var err error
		start, err = strconv.ParseInt(startS, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid range %q", ErrRangeNotSatisfiable, header)
		}
		if endS == "" {
			end = total - 1
		} else {
			end, err = strconv.ParseInt(endS, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("%w: invalid range %q", ErrRangeNotSatisfiable, header)
			}
		}
	}
	if start < 0 || end < start || start >= total {
		return nil, fmt.Errorf("%w: %q against size %d", ErrRangeNotSatisfiable, header, total)
	}
	end = min(end, total-1)
	return &ByteRange{Start: start, End: end, Total: total}, nil
}
