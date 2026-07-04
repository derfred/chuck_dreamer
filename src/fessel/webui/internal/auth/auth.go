// Package auth reads the operator identity from oauth2-proxy headers (B2.1).
//
// The webui sits behind oauth2-proxy, which runs the GitHub OIDC flow and
// forwards already-authenticated requests carrying identity headers
// (X-Auth-Request-User / -Email). The service does NOT run the OIDC code
// flow; it trusts these headers when present and treats their absence as
// unauthenticated. Header names are config, not hardcoded.
//
// There are no proxy-bypass endpoints on the public listener; the ingest
// listener carries no identity headers at all (tailnet trust).
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

	// DevIdentity, when non-empty, is the identity assumed for requests that
	// carry NO identity headers — an auth BYPASS for proxy-less throwaway
	// environments (the live-preview harness). NEVER set in production: with
	// oauth2-proxy in front, unauthenticated requests must stay 401.
	DevIdentity string
}

// Read returns the forwarded operator identity, or nil if unauthenticated.
func (h Headers) Read(r *http.Request) *Identity {
	user := r.Header.Get(h.User)
	if user == "" {
		if h.DevIdentity != "" {
			return &Identity{User: h.DevIdentity}
		}
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
