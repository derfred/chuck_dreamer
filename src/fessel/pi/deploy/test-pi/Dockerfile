# Test-only Pi image (Slice 1.5, T1.1). Installs the REAL dpkg (built in a
# first stage) so the integration test exercises the install asset, then
# adds a process-supervisor entrypoint. Camera is video's built-in
# videotestsrc (no /dev/video0, no v4l2loopback).
#
# Debian trixie is used because it ships Python 3.13 (bookworm ships 3.11,
# but fessel requires >=3.12).
#
# Build context is src/fessel/ (so the build can see schemas/, pi/, etc.).
#   docker build -f pi/deploy/test-pi/Dockerfile -t <tag> .

# ---- stage 1: build the .deb (no venv/pip — just dpkg-deb) ----
FROM debian:trixie AS deb-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      dpkg-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . /src
RUN chmod +x pi/deploy/dpkg/build-dpkg.sh \
    && pi/deploy/dpkg/build-dpkg.sh 0.1.0

# ---- stage 1b: build whipclientsink (gst-plugins-rs net/webrtc) for amd64 ----
# fessel-monitor now Depends on gstreamer1.0-plugins-rs-fessel (the live WHIP
# sender, §2.2), which on the Pi is the cross-built arm64 deb from
# pi/deploy/gst-plugins-rs. Debian trixie does not package the rust net/webrtc
# plugin, so the test image builds the same plugin natively (amd64) and wraps
# it in a minimal local deb that satisfies the Depends. Mirrors
# pi/deploy/gst-plugins-rs/Dockerfile (branch 0.15 targets GStreamer >= 1.22;
# trixie ships 1.26). This stage is fully static, so Docker layer caching
# avoids the rust rebuild on unrelated changes.
FROM debian:trixie AS whip-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      cargo-c binutils curl gcc dpkg-dev \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      libgstreamer-plugins-bad1.0-dev \
      libssl-dev pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# rustup toolchain (trixie's rustc 1.85 is too old for the 0.15 dep tree).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /src
RUN git clone --depth 1 --branch 0.15 \
      https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git .
RUN cargo cbuild -p gst-plugin-webrtc --release
RUN set -e; \
    mkdir /plugins; \
    find target -name 'libgstrswebrtc.so' -exec cp {} /plugins/ \; ; \
    strip --strip-unneeded /plugins/*.so; \
    STAGE=/pkg/stage; \
    PLUGDIR="$STAGE/usr/lib/x86_64-linux-gnu/gstreamer-1.0"; \
    install -d "$PLUGDIR" "$STAGE/DEBIAN"; \
    install -m 644 /plugins/*.so "$PLUGDIR/"; \
    { \
      echo "Package: gstreamer1.0-plugins-rs-fessel"; \
      echo "Version: 0.15+test"; \
      echo "Section: libs"; \
      echo "Priority: optional"; \
      echo "Architecture: amd64"; \
      echo "Depends: gstreamer1.0-plugins-bad, gstreamer1.0-nice"; \
      echo "Maintainer: Fessel <noreply@example.com>"; \
      echo "Description: gst-plugins-rs webrtc (whipclientsink) for the amd64 test-pi image"; \
    } > "$STAGE/DEBIAN/control"; \
    mkdir /out; \
    dpkg-deb --build --root-owner-group "$STAGE" \
      "/out/gstreamer1.0-plugins-rs-fessel_0.15+test_amd64.deb"

# ---- stage 2: the runtime test-Pi image ----
FROM debian:trixie

# The dpkg's Depends pull these, but listing them keeps the image build
# deterministic. The Python libs (pydantic/fastapi/uvicorn[+extras]/paho/yaml)
# are now apt packages, not a venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-gi \
      python3-pydantic python3-paho-mqtt python3-yaml \
      python3-fastapi python3-uvicorn python3-uvloop python3-httptools \
      python3-websockets python3-watchfiles python3-httpx \
      python3-numpy python3-opencv \
      gir1.2-gst-plugins-bad-1.0 \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools \
      gstreamer1.0-nice \
      mosquitto mosquitto-clients v4l-utils \
      curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# whipclientsink (locally built amd64 deb; satisfies fessel-monitor's
# gstreamer1.0-plugins-rs-fessel Depends — video fails loud at startup
# without the element).
COPY --from=whip-builder /out/gstreamer1.0-plugins-rs-fessel_0.15+test_amd64.deb /tmp/
RUN dpkg -i /tmp/gstreamer1.0-plugins-rs-fessel_0.15+test_amd64.deb \
    && rm -f /tmp/gstreamer1.0-plugins-rs-fessel_0.15+test_amd64.deb

# Install the built package (exercises the real install asset).
COPY --from=deb-builder /src/dist/fessel-monitor_0.1.0_all.deb /tmp/
RUN dpkg -i /tmp/fessel-monitor_0.1.0_all.deb || (apt-get update && apt-get -f install -y) \
    && rm -f /tmp/fessel-monitor_0.1.0_all.deb

# Entrypoint (process supervisor shim).
COPY pi/deploy/test-pi/entrypoint.sh /usr/local/bin/fessel-test-entrypoint
RUN chmod +x /usr/local/bin/fessel-test-entrypoint

# Test config: point video at the in-cluster webui WHIP ingest listener and
# turn on the software-encoder seam (no hardware encoder in a container). The
# uploader defaults to the in-memory 'fake' ingest client; the integration env
# flips it to the real http driver against the in-cluster webui ingest listener
# via FESSEL_INGEST_* env. This single fessel.yaml overwrites the dpkg-seeded
# conffile for the test image only.
COPY pi/deploy/test-pi/fessel.test.yaml /etc/fessel/fessel.yaml

EXPOSE 8443
ENTRYPOINT ["/usr/local/bin/fessel-test-entrypoint"]
