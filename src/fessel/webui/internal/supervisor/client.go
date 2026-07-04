// Package supervisor is the webui -> supervisor client over the cluster->Pi
// Tailscale egress (B3.1/B3.2).
//
// The webui is a *pass-through*: it does NOT add retries or business logic on
// top of supervisor. supervisor's status code and structured body flow through
// to the frontend verbatim — a 5xx (actuator failure) reaches the frontend as
// a 5xx so the operator sees the real diagnostic and decides whether to retry.
package supervisor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// UserAgent identifies the webui in supervisor's logs (the same convention
// supervisor uses for its own outbound HTTP).
const UserAgent = "fessel-webui"

// ForwardResult is the raw outcome of a forwarded call: supervisor's status +
// parsed JSON body (or a synthesised body when supervisor was unreachable).
// Body is a map for object responses and a []any for array responses (e.g.
// /recordings), passed through to the frontend verbatim.
type ForwardResult struct {
	StatusCode int
	Body       any
}

// ProxyResult is a proxied binary response: status + body bytes + the subset
// of headers an HLS player needs (content-type, ranges, caching). Err is set
// only when supervisor was unreachable (status 502).
type ProxyResult struct {
	StatusCode int
	Body       []byte
	Headers    map[string]string
	Err        string
}

type Client struct {
	base string
	http *http.Client
}

func New(base string, timeout time.Duration) *Client {
	return &Client{base: base, http: &http.Client{Timeout: timeout}}
}

// NewWithHTTPClient is the test seam: any RoundTripper-backed client.
func NewWithHTTPClient(base string, c *http.Client) *Client {
	return &Client{base: base, http: c}
}

func (c *Client) forward(method, path string, jsonBody any) ForwardResult {
	var body io.Reader
	if jsonBody != nil {
		raw, err := json.Marshal(jsonBody)
		if err != nil {
			return unreachable(fmt.Errorf("encode body: %w", err))
		}
		body = bytes.NewReader(raw)
	}
	req, err := http.NewRequest(method, c.base+path, body)
	if err != nil {
		return unreachable(err)
	}
	req.Header.Set("User-Agent", UserAgent)
	if jsonBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		// supervisor unreachable (egress down, Pi offline). Surface as 502 with
		// a structured body — the operator should see "couldn't reach the Pi",
		// distinct from supervisor's own 503 actuator failures.
		return unreachable(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return unreachable(err)
	}
	return ForwardResult{StatusCode: resp.StatusCode, Body: jsonOrText(raw)}
}

func unreachable(err error) ForwardResult {
	return ForwardResult{
		StatusCode: 502,
		Body:       map[string]any{"error": "supervisor_unreachable", "message": err.Error()},
	}
}

func (c *Client) Post(path string, jsonBody any) ForwardResult {
	return c.forward(http.MethodPost, path, jsonBody)
}

func (c *Client) Get(path string) ForwardResult {
	return c.forward(http.MethodGet, path, nil)
}

// hlsHeaders is the header subset forwarded to an HLS player for ranged
// playback + caching.
var hlsHeaders = []string{"Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "Cache-Control"}

// GetBytes fetches a binary resource (ring/recording HLS file) from
// supervisor, passing through request headers (e.g. Range). Range support is
// end-to-end: supervisor's file responses honour Range and return 206, which
// flows through here verbatim.
func (c *Client) GetBytes(path string, headers map[string]string) ProxyResult {
	req, err := http.NewRequest(http.MethodGet, c.base+path, nil)
	if err != nil {
		return ProxyResult{StatusCode: 502, Err: err.Error()}
	}
	req.Header.Set("User-Agent", UserAgent)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return ProxyResult{StatusCode: 502, Err: err.Error()}
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ProxyResult{StatusCode: 502, Err: err.Error()}
	}
	passthrough := map[string]string{}
	for _, k := range hlsHeaders {
		if v := resp.Header.Get(k); v != "" {
			passthrough[k] = v
		}
	}
	return ProxyResult{StatusCode: resp.StatusCode, Body: body, Headers: passthrough}
}

// jsonOrText mirrors the FastAPI backend's _json_or_text: preserve object and
// array bodies; wrap non-JSON text as {"detail": ...} and bare scalars as
// {"value": ...} so the result is always a JSON container.
func jsonOrText(raw []byte) any {
	var parsed any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return map[string]any{"detail": string(raw)}
	}
	switch parsed.(type) {
	case map[string]any, []any:
		return parsed
	default:
		return map[string]any{"value": parsed}
	}
}
