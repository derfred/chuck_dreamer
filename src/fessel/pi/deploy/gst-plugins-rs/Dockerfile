# Cross-build gst-plugins-rs WHIP support for the Pi and package it as a .deb.
#
# Ships TWO plugins — both are required for the WHIP uplink:
#   libgstrswebrtc.so  whipclientsink (the WHIP publisher element)
#   libgstrsrtp.so     rtpgccbwe (GCC bandwidth estimation; without it
#                      whipclientsink disables congestion control and the
#                      bitrate bounces instead of adapting — the entire point
#                      on a cellular uplink)
#
# Built inside arm64 Debian trixie (buildx --platform linux/arm64, QEMU on the
# Mac / amd64 CI) so the plugins link against the SAME GStreamer 1.26 C libs
# the Pi runs — no cross-linker / sysroot juggling. Branch 0.13 of
# gst-plugins-rs targets GStreamer 1.26.
#
# The final stage is a scratch image holding just the .deb:
#   docker buildx build --platform linux/arm64 --output type=local,dest=./dist .
#
# Version: <upstream gst-plugins-rs version>+fessel<DEB_REVISION>+gst<minor>,
# e.g. 0.13.6+fessel1+gst1.26. Bump DEB_REVISION for a repackage of the same
# upstream; the gst minor changes only when the Pi's GStreamer does (which
# REQUIRES a rebuild — the plugin ABI tracks the GStreamer minor, enforced by
# the <<-next-minor Depends guard below).
ARG GST_PLUGINS_RS_BRANCH=0.13
ARG DEB_REVISION=1

FROM debian:trixie AS build
ARG GST_PLUGINS_RS_BRANCH

# plugins-bad1.0-dev provides gstreamer-webrtc-1.0.pc (the C webrtc lib the
# rust plugin links against); base-dev alone is not enough.
RUN apt-get update && apt-get install -y --no-install-recommends \
      cargo rustc cargo-c binutils \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      libgstreamer-plugins-bad1.0-dev \
      libssl-dev pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${GST_PLUGINS_RS_BRANCH}" \
      https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git .

# cargo-c produces proper shared objects with GStreamer plugin metadata.
# One cbuild per plugin (cargo-c takes a single -p).
RUN cargo cbuild -p gst-plugin-webrtc --release && \
    cargo cbuild -p gst-plugin-rtp --release

# Collect + strip (unstripped they carry ~hundreds of MB of debuginfo, which
# matters both for the .deb size and the Pi's apt download over cellular).
RUN mkdir /plugins && \
    find target -name 'libgstrswebrtc.so' -exec cp {} /plugins/ \; && \
    find target -name 'libgstrsrtp.so' -exec cp {} /plugins/ \; && \
    strip --strip-unneeded /plugins/*.so && \
    ls -la /plugins

# ---- .deb assembly ------------------------------------------------------------
FROM build AS deb
ARG DEB_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends dpkg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pkg
COPY postinst /pkg/postinst

RUN set -e; \
    # Version: upstream + our revision + the GStreamer minor we built against.
    # Upstream comes from the .pc file cargo-c generated for the artifact we
    # actually built — NOT from net/webrtc/Cargo.toml, which declares
    # `version.workspace = true` (no literal version line to grep).
    PC="$(find /src/target -name 'gstrswebrtc*.pc' | head -n1)"; \
    UPSTREAM="$([ -n "$PC" ] && sed -n 's/^Version: *//p' "$PC" | head -n1)"; \
    # Fallback: the workspace-level version ([workspace.package] in the root
    # Cargo.toml) — first literal version line wins.
    [ -n "$UPSTREAM" ] || UPSTREAM="$(sed -n 's/^version = "\([^"]*\)".*/\1/p' /src/Cargo.toml | head -n1)"; \
    [ -n "$UPSTREAM" ] || { echo "could not determine upstream version (no .pc, no workspace version)" >&2; exit 1; }; \
    GST_VER="$(pkg-config --modversion gstreamer-1.0)"; \
    GST_MINOR="${GST_VER%.*}"; \
    GST_NEXT="1.$(( $(echo "$GST_MINOR" | cut -d. -f2) + 2 ))"; \
    VERSION="${UPSTREAM}+fessel${DEB_REVISION}+gst${GST_MINOR}"; \
    echo "packaging version $VERSION (built against GStreamer $GST_VER)"; \
    \
    STAGE=/pkg/stage; \
    PLUGDIR="$STAGE/usr/lib/aarch64-linux-gnu/gstreamer-1.0"; \
    install -d "$PLUGDIR"; \
    install -m 644 /plugins/*.so "$PLUGDIR/"; \
    \
    # Derive library Depends with dpkg-shlibdeps (needs a debian/control stub
    # to run outside a debhelper build). This runs in the SAME container the
    # plugins were built in, so every NEEDED library resolves to the exact
    # trixie package that provides it.
    mkdir -p /pkg/shlib/debian; \
    printf 'Source: gstreamer1.0-plugins-rs-fessel\n\nPackage: gstreamer1.0-plugins-rs-fessel\nArchitecture: arm64\nDescription: stub\n' \
      > /pkg/shlib/debian/control; \
    ( cd /pkg/shlib && dpkg-shlibdeps -O "$PLUGDIR"/*.so > substvars ); \
    SHLIB_DEPS="$(sed -n 's/^shlibs:Depends=//p' /pkg/shlib/substvars)"; \
    echo "shlibdeps: $SHLIB_DEPS"; \
    [ -n "$SHLIB_DEPS" ]; \
    \
    # Manual additions shlibdeps cannot see:
    #  - libgstreamer1.0-0 << next-minor: the plugin ABI tracks the GStreamer
    #    minor; a Pi OS upgrade to a newer GStreamer must fail loudly at apt
    #    level (rebuild + bump gst suffix) instead of mysteriously at runtime.
    #  - gstreamer1.0-plugins-bad: webrtcbin, which whipclientsink drives.
    #  - gstreamer1.0-nice: libnice ICE elements; without them webrtcbin fails
    #    pad requests at runtime ("Failed to request pad") — invisible to
    #    shlibdeps because elements load dynamically.
    DEPS="$SHLIB_DEPS, libgstreamer1.0-0 (<< ${GST_NEXT}), gstreamer1.0-plugins-bad, gstreamer1.0-nice"; \
    \
    install -d "$STAGE/DEBIAN"; \
    { \
      echo "Package: gstreamer1.0-plugins-rs-fessel"; \
      echo "Version: $VERSION"; \
      echo "Section: libs"; \
      echo "Priority: optional"; \
      echo "Architecture: arm64"; \
      echo "Depends: $DEPS"; \
      echo "Maintainer: Fessel <noreply@example.com>"; \
      echo "Description: gst-plugins-rs WHIP publisher elements for Fessel"; \
      echo " whipclientsink (WHIP WebRTC publisher) and rtpgccbwe (GCC"; \
      echo " congestion control) from gst-plugins-rs ${UPSTREAM}, cross-built"; \
      echo " against Debian trixie's GStreamer ${GST_MINOR} for the Fessel Pi."; \
      echo " Debian does not package the rust net/webrtc plugin; this package"; \
      echo " exists until it does (then prefer the official one)."; \
    } > "$STAGE/DEBIAN/control"; \
    install -m 755 /pkg/postinst "$STAGE/DEBIAN/postinst"; \
    \
    mkdir /out; \
    dpkg-deb --build --root-owner-group "$STAGE" \
      "/out/gstreamer1.0-plugins-rs-fessel_${VERSION}_arm64.deb"; \
    dpkg-deb -I /out/*.deb

FROM scratch AS artifact
COPY --from=deb /out/ /
