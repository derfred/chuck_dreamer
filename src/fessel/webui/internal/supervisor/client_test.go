package supervisor

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestForwardPassesStatusAndBodyThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") != UserAgent {
			t.Errorf("user agent %q", r.Header.Get("User-Agent"))
		}
		w.WriteHeader(503)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": "plug_unverified"})
	}))
	defer srv.Close()
	c := New(srv.URL, time.Second)
	res := c.Post("/control/shutdown/arm", nil)
	if res.StatusCode != 503 || res.Body.(map[string]any)["error"] != "plug_unverified" {
		t.Fatalf("%+v", res)
	}
}

func TestForwardPreservesArrayBodies(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `[{"recording_id": "r1"}]`)
	}))
	defer srv.Close()
	res := New(srv.URL, time.Second).Get("/recordings")
	arr, ok := res.Body.([]any)
	if !ok || len(arr) != 1 {
		t.Fatalf("%+v", res.Body)
	}
}

func TestForwardWrapsNonJSONAndScalars(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/text":
			_, _ = io.WriteString(w, "plain text")
		case "/scalar":
			_, _ = io.WriteString(w, "42")
		}
	}))
	defer srv.Close()
	c := New(srv.URL, time.Second)
	if body := c.Get("/text").Body.(map[string]any); body["detail"] != "plain text" {
		t.Fatalf("%v", body)
	}
	if body := c.Get("/scalar").Body.(map[string]any); body["value"] != float64(42) {
		t.Fatalf("%v", body)
	}
}

func TestUnreachableSynthesises502(t *testing.T) {
	c := New("http://127.0.0.1:1", 200*time.Millisecond)
	res := c.Get("/state")
	if res.StatusCode != 502 {
		t.Fatalf("%d", res.StatusCode)
	}
	if res.Body.(map[string]any)["error"] != "supervisor_unreachable" {
		t.Fatalf("%v", res.Body)
	}
}

func TestGetBytesForwardsRangeAndFiltersHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "bytes=0-3" {
			t.Errorf("range %q", r.Header.Get("Range"))
		}
		w.Header().Set("Content-Type", "video/mp2t")
		w.Header().Set("Content-Range", "bytes 0-3/100")
		w.Header().Set("X-Internal", "secret")
		w.WriteHeader(206)
		_, _ = w.Write([]byte("part"))
	}))
	defer srv.Close()
	res := New(srv.URL, time.Second).GetBytes("/ring/index.m3u8", map[string]string{"Range": "bytes=0-3"})
	if res.StatusCode != 206 || string(res.Body) != "part" {
		t.Fatalf("%+v", res)
	}
	if res.Headers["Content-Range"] != "bytes 0-3/100" {
		t.Fatalf("%v", res.Headers)
	}
	if _, leaked := res.Headers["X-Internal"]; leaked {
		t.Fatal("internal header leaked")
	}
}

func TestGetBytesUnreachable(t *testing.T) {
	res := New("http://127.0.0.1:1", 200*time.Millisecond).GetBytes("/ring/index.m3u8", nil)
	if res.StatusCode != 502 || res.Err == "" {
		t.Fatalf("%+v", res)
	}
}
