// Package version resolves the release stamp reported by /healthz.
//
// A deployed cluster webui and Pi supervisor should report the same string
// (image tag == dpkg version); a mismatch means the two halves of a release
// drifted. Resolution order mirrors the Python fessel_schemas.version:
//  1. FESSEL_VERSION env var — explicit override for tests/dev, and what the
//     container image sets from the build arg.
//  2. the build-time ldflags stamp.
//  3. "unknown".
package version

import "os"

// Stamp is overridable at build time:
//
//	go build -ldflags "-X .../internal/version.Stamp=1.4.0"
var Stamp = "unknown"

func Version() string {
	if v := os.Getenv("FESSEL_VERSION"); v != "" {
		return v
	}
	return Stamp
}

// Minor returns the MAJOR.MINOR components of a semver-shaped string, or
// ok=false if it isn't parseable (spec §2.1: only the first two dotted
// components are compared; "1.4" with no patch is accepted, "unknown" is not).
func Minor(v string) (major, minor string, ok bool) {
	parts := splitDots(v)
	if len(parts) < 2 || !allDigits(parts[0]) || !allDigits(parts[1]) {
		return "", "", false
	}
	return parts[0], parts[1], true
}

func splitDots(v string) []string {
	var out []string
	cur := ""
	for _, r := range trimSpace(v) {
		if r == '.' {
			out = append(out, cur)
			cur = ""
			continue
		}
		cur += string(r)
	}
	return append(out, cur)
}

func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n') {
		end--
	}
	return s[start:end]
}

func allDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
