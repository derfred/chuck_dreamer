// Package auth reads the operator identity from oauth2-proxy headers (B2.1).
//
// The webui sits behind oauth2-proxy, which runs the GitHub OIDC flow and
// forwards already-authenticated requests carrying identity headers
// (X-Auth-Request-User / -Email). The service does NOT run the OIDC code
// flow; it trusts these headers when present and treats their absence as
// unauthenticated. Header names are config, not hardcoded.
//
// With the mediamtx design gone there are no proxy-bypass endpoints left on
// the public listener (the /jwks reject-identity dance is deleted); the
// ingest listener carries no identity headers at all (tailnet trust).
package auth

import (
	"net/http"
	"strings"
)

type Identity struct {
	User   string
	Email  string
	Groups []string
}

// Headers holds the configured identity-header names (read once at startup).
type Headers struct {
	User   string
	Email  string
	Groups string
}

// Read returns the forwarded operator identity, or nil if unauthenticated.
func (h Headers) Read(r *http.Request) *Identity {
	user := r.Header.Get(h.User)
	if user == "" {
		return nil
	}
	var groups []string
	for _, g := range strings.Split(r.Header.Get(h.Groups), ",") {
		if g = strings.TrimSpace(g); g != "" {
			groups = append(groups, g)
		}
	}
	return &Identity{User: user, Email: r.Header.Get(h.Email), Groups: groups}
}
