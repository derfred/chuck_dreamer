package storage

import "testing"

func TestRewritePlaylistURIs(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"bare segment", "#EXTINF:2.0,\nseg-0.ts\n", "#EXTINF:2.0,\nsegment/seg-0.ts\n"},
		{"tags untouched", "#EXTM3U\n#EXT-X-ENDLIST\n", "#EXTM3U\n#EXT-X-ENDLIST\n"},
		{"blank lines kept", "#EXTM3U\n\nseg-0.ts\n", "#EXTM3U\n\nsegment/seg-0.ts\n"},
		{"absolute url left alone", "http://x/seg-0.ts\n", "http://x/seg-0.ts\n"},
		{"root-relative left alone", "/abs/seg-0.ts\n", "/abs/seg-0.ts\n"},
		{"crlf preserved", "#EXTM3U\r\nseg-0.ts\r\n", "#EXTM3U\r\nsegment/seg-0.ts\r\n"},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := string(RewritePlaylistURIs([]byte(c.in), "segment/"))
			if got != c.want {
				t.Fatalf("got %q want %q", got, c.want)
			}
		})
	}
}
