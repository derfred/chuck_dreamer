package auth

import (
	"net/http/httptest"
	"testing"
)

func TestReadRequiresHeaderByDefault(t *testing.T) {
	h := Headers{User: "X-Auth-Request-User", Email: "X-Auth-Request-Email", Groups: "X-Auth-Request-Groups"}
	if id := h.Read(httptest.NewRequest("GET", "/api/me", nil)); id != nil {
		t.Fatalf("header-less request authenticated: %+v", id)
	}
	req := httptest.NewRequest("GET", "/api/me", nil)
	req.Header.Set("X-Auth-Request-User", "alice")
	if id := h.Read(req); id == nil || id.User != "alice" {
		t.Fatalf("id: %+v", id)
	}
}

func TestDevIdentityBypassesOnlyHeaderlessRequests(t *testing.T) {
	h := Headers{User: "X-Auth-Request-User", Email: "X-Auth-Request-Email",
		Groups: "X-Auth-Request-Groups", DevIdentity: "live-preview-operator"}
	// Header-less -> the dev identity (proxy-less preview environments).
	if id := h.Read(httptest.NewRequest("GET", "/whep", nil)); id == nil || id.User != "live-preview-operator" {
		t.Fatalf("id: %+v", id)
	}
	// A real forwarded identity still wins.
	req := httptest.NewRequest("GET", "/whep", nil)
	req.Header.Set("X-Auth-Request-User", "alice")
	if id := h.Read(req); id == nil || id.User != "alice" {
		t.Fatalf("id: %+v", id)
	}
}
