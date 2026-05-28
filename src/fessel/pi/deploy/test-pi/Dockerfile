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

# ---- stage 1: build the .deb ----
FROM debian:trixie AS deb-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip dpkg-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . /src
RUN chmod +x pi/deploy/dpkg/build-dpkg.sh \
    && pi/deploy/dpkg/build-dpkg.sh 0.1.0

# ---- stage 2: the runtime test-Pi image ----
FROM debian:trixie

# System deps the dpkg declares + the ones video needs for gi/GStreamer.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-gi \
      gir1.2-gst-plugins-bad-1.0 \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools \
      mosquitto mosquitto-clients v4l-utils \
      curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# Install the built package (exercises the real install asset).
COPY --from=deb-builder /src/dist/fessel-monitor_0.1.0_all.deb /tmp/
RUN dpkg -i /tmp/fessel-monitor_0.1.0_all.deb || (apt-get update && apt-get -f install -y) \
    && rm -f /tmp/fessel-monitor_0.1.0_all.deb

# Entrypoint (process supervisor shim).
COPY pi/deploy/test-pi/entrypoint.sh /usr/local/bin/fessel-test-entrypoint
RUN chmod +x /usr/local/bin/fessel-test-entrypoint

# Test config: point video at the in-cluster mediamtx-srt Service and turn
# on the software-encoder seam (no hardware encoder in a container). These
# overwrite the dpkg-seeded conffiles for the test image only.
COPY pi/deploy/test-pi/video.test.yaml /etc/fessel/video.yaml
COPY pi/deploy/test-pi/supervisor.test.yaml /etc/fessel/supervisor.yaml

EXPOSE 8443
ENTRYPOINT ["/usr/local/bin/fessel-test-entrypoint"]
