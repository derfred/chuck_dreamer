# Test Job image: the WHEP viewer is aiortc — a real Python ICE/DTLS/SRTP
# peer with an H.264 decoder (via PyAV's bundled ffmpeg), so the driver can
# assert decoded frames without any browser. The former Chrome/Xvfb
# machinery is gone: in podip mode the in-process Pion relay and this pod
# exchange plain pod-IP host candidates, so in-cluster ICE completes.
#
# Build context is src/fessel/integration/driver/.
FROM python:3.12-slim

WORKDIR /driver
# aiortc + deps (av, pylibsrtp, cryptography, ...) all ship manylinux wheels;
# no compiler or system ffmpeg needed.
RUN pip install --no-cache-dir aiortc==1.9.0

COPY run_tests.py /driver/run_tests.py
COPY entrypoint.sh /driver/entrypoint.sh
RUN chmod +x /driver/entrypoint.sh

ENTRYPOINT ["/driver/entrypoint.sh"]
