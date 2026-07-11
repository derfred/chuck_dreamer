// MinIO storage backend (B5.5.2).
//
// The webui holds the S3 credentials and the Pi holds none. Writes to an
// S3-compatible bucket; playback is a presigned-GET redirect so the browser
// fetches segments straight from MinIO and the webui never touches the bytes.
//
// On-store key layout (identical to disk, architecture §4.3):
//
//	recordings/<type>/<id>/<name>
package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

const (
	keyRoot = "recordings"
	// 5 minutes: enough for an HLS player to walk the playlist + segments.
	presignTTL = 5 * time.Minute
	// 10 MiB part size for length-unknown streaming uploads (the SDK's minimum
	// multipart part is 5 MiB; 10 keeps part counts low for multi-MB segments).
	minioPartSize = 10 * 1024 * 1024
	listCacheTTL  = 1500 * time.Millisecond

	// Monitor freeze-frame: a sibling key prefix to recordings/, a single
	// fixed object — deliberately outside the recordings prefix so it never
	// surfaces in List() (which only walks keyRoot+"/").
	snapshotKey = "monitor/snapshot.jpg"
)

func objectKey(recType, recordingID, fileName string) string {
	sub := Explicit
	if recType == Anomaly {
		sub = Anomaly
	}
	return fmt.Sprintf("%s/%s/%s/%s", keyRoot, sub, recordingID, fileName)
}

type MinioBackend struct {
	client *minio.Client
	bucket string

	// Tiny TTL cache for List() to soften the dashboard's poll rate (B5.5.2).
	mu        sync.Mutex
	cached    []RecordingView
	cachedAt  time.Time
	haveCache bool
}

type MinioOptions struct {
	Endpoint  string
	AccessKey string
	SecretKey string
	Bucket    string
	Secure    bool
}

func NewMinioBackend(opts MinioOptions) (*MinioBackend, error) {
	client, err := minio.New(opts.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(opts.AccessKey, opts.SecretKey, ""),
		Secure: opts.Secure,
	})
	if err != nil {
		return nil, fmt.Errorf("minio client: %w", err)
	}
	return &MinioBackend{client: client, bucket: opts.Bucket}, nil
}

func (m *MinioBackend) Store(recordingID, fileName string, body io.Reader) error {
	if !isPlainComponent(recordingID) || !isPlainComponent(fileName) {
		return ErrInvalidPath
	}
	recType := m.recType(recordingID)
	key := objectKey(recType, recordingID, fileName)
	// PutObject with size -1 streams to EOF in part-size chunks (idempotent: a
	// repeated PUT overwrites the object; atomic: the SDK only commits on
	// completion).
	_, err := m.client.PutObject(context.Background(), m.bucket, key, body, -1,
		minio.PutObjectOptions{PartSize: minioPartSize, ContentType: ContentTypeFor(fileName)})
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.haveCache = false // invalidate: a new file changes the listing
	m.mu.Unlock()
	return nil
}

// recType reads the recording's type from a stored metadata.json if present,
// else defaults to explicit. Checked under both type prefixes.
func (m *MinioBackend) recType(recordingID string) string {
	for _, recType := range []string{Explicit, Anomaly} {
		raw := m.getObjectBytes(objectKey(recType, recordingID, MetadataFilename))
		if raw == nil {
			continue
		}
		var meta map[string]any
		if err := json.Unmarshal(raw, &meta); err != nil {
			continue
		}
		if t, ok := meta["type"].(string); ok && (t == Explicit || t == Anomaly) {
			return t
		}
		return recType // the dir exists under this prefix -> use it
	}
	return DefaultType
}

func (m *MinioBackend) Exists(recordingID string) bool {
	if !isPlainComponent(recordingID) {
		return false
	}
	for _, recType := range []string{Explicit, Anomaly} {
		prefix := fmt.Sprintf("%s/%s/%s/", keyRoot, recType, recordingID)
		ctx, cancel := context.WithCancel(context.Background())
		for range m.client.ListObjects(ctx, m.bucket, minio.ListObjectsOptions{Prefix: prefix, Recursive: true}) {
			cancel()
			return true
		}
		cancel()
	}
	return false
}

func (m *MinioBackend) List() []RecordingView {
	m.mu.Lock()
	if m.haveCache && time.Since(m.cachedAt) < listCacheTTL {
		cached := m.cached
		m.mu.Unlock()
		return cached
	}
	m.mu.Unlock()

	type entry struct {
		recType string
		files   []string
	}
	byID := map[string]*entry{}
	var order []string
	for obj := range m.client.ListObjects(context.Background(), m.bucket,
		minio.ListObjectsOptions{Prefix: keyRoot + "/", Recursive: true}) {
		if obj.Err != nil {
			continue
		}
		// key = recordings/<type>/<id>/<file>
		parts := strings.Split(obj.Key, "/")
		if len(parts) < 4 || parts[0] != keyRoot {
			continue
		}
		recType, rid, fname := parts[1], parts[2], strings.Join(parts[3:], "/")
		e, ok := byID[rid]
		if !ok {
			e = &entry{recType: recType}
			byID[rid] = e
			order = append(order, rid)
		}
		e.files = append(e.files, fname)
	}
	views := []RecordingView{}
	for _, rid := range order {
		e := byID[rid]
		var meta map[string]any
		for _, f := range e.files {
			if f == MetadataFilename {
				if raw := m.getObjectBytes(objectKey(e.recType, rid, MetadataFilename)); raw != nil {
					var parsed map[string]any
					if err := json.Unmarshal(raw, &parsed); err == nil {
						meta = parsed
					}
				}
				break
			}
		}
		recType := e.recType
		if recType != Explicit && recType != Anomaly {
			recType = DefaultType
		}
		sort.Strings(e.files)
		views = append(views, RecordingView{RecordingID: rid, RecType: recType, Files: e.files, Metadata: meta})
	}
	sort.SliceStable(views, func(i, j int) bool { return views[i].StartedAt() > views[j].StartedAt() })

	m.mu.Lock()
	m.cached, m.cachedAt, m.haveCache = views, time.Now(), true
	m.mu.Unlock()
	return views
}

func (m *MinioBackend) PlaybackURL(recordingID, fileName string) (PlaybackTarget, error) {
	if !isPlainComponent(recordingID) || !isPlainComponent(fileName) {
		return nil, ErrInvalidPath
	}
	recType := m.recType(recordingID)
	key := objectKey(recType, recordingID, fileName)
	u, err := m.client.PresignedGetObject(context.Background(), m.bucket, key, presignTTL, url.Values{})
	if err != nil {
		// Missing object / transient -> treat as absent.
		return nil, nil
	}
	return PresignedURL{URL: u.String()}, nil
}

func (m *MinioBackend) Read(recordingID, fileName, httpRange string) (*ReadResult, error) {
	// MinIO playback is via presigned redirect, never through the webui.
	return nil, fmt.Errorf("minio backend does not serve playback bytes; use PlaybackURL")
}

func (m *MinioBackend) StoreSnapshot(jpeg []byte) error {
	_, err := m.client.PutObject(context.Background(), m.bucket, snapshotKey,
		bytes.NewReader(jpeg), int64(len(jpeg)),
		minio.PutObjectOptions{ContentType: "image/jpeg"})
	return err
}

func (m *MinioBackend) ReadSnapshot() ([]byte, time.Time, bool) {
	info, err := m.client.StatObject(context.Background(), m.bucket, snapshotKey, minio.StatObjectOptions{})
	if err != nil {
		return nil, time.Time{}, false
	}
	raw := m.getObjectBytes(snapshotKey)
	if raw == nil {
		return nil, time.Time{}, false
	}
	return raw, info.LastModified, true
}

func (m *MinioBackend) getObjectBytes(key string) []byte {
	obj, err := m.client.GetObject(context.Background(), m.bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return nil
	}
	defer obj.Close()
	raw, err := io.ReadAll(obj)
	if err != nil {
		return nil
	}
	return raw
}
