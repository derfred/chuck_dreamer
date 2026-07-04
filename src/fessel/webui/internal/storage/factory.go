// Storage backend factory (B5.5.1). Chooses the implementation from the
// config at startup. The Tanka library wires the env from the
// `recordings_storage` config block — MinIO credentials from a k8s Secret,
// the disk path from the PVC mount.
package storage

import (
	"fmt"

	"github.com/derfred/fessel/webui/internal/config"
)

func Build(cfg config.Config) (Backend, error) {
	backend := cfg.RecordingsBackend
	if backend == "" {
		// Back-compat with the MinIO-only env: if those vars are present,
		// behave as a MinIO backend; otherwise there is nothing to build. The
		// ingest endpoint requires a store, so error rather than silently
		// dropping uploads.
		if cfg.MinioEndpoint != "" {
			backend = "minio"
		} else {
			return nil, fmt.Errorf("no recordings storage configured: set FESSEL_RECORDINGS_BACKEND to 'minio' or 'disk'")
		}
	}
	switch backend {
	case "disk":
		if cfg.DiskPath == "" {
			return nil, fmt.Errorf("FESSEL_RECORDINGS_DISK_PATH must be set for the disk backend")
		}
		return NewDiskBackend(cfg.DiskPath)
	case "minio":
		if cfg.MinioEndpoint == "" || cfg.MinioBucket == "" || cfg.MinioAccessKey == "" || cfg.MinioSecretKey == "" {
			return nil, fmt.Errorf("MinIO backend requires FESSEL_MINIO_ENDPOINT/_BUCKET/_ACCESS_KEY/_SECRET_KEY")
		}
		return NewMinioBackend(MinioOptions{
			Endpoint:  cfg.MinioEndpoint,
			AccessKey: cfg.MinioAccessKey,
			SecretKey: cfg.MinioSecretKey,
			Bucket:    cfg.MinioBucket,
			Secure:    cfg.MinioSecure,
		})
	default:
		return nil, fmt.Errorf("unknown recordings storage backend: %q", backend)
	}
}
