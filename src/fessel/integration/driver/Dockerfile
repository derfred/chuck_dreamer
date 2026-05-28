# Test Job image (T1.5): Playwright Chromium driver run HEADED under Xvfb
# so the browser has an H.264 decoder (the bundled headless build does not;
# mediamtx would reject the client with "codecs not supported").
#
# Build context is src/fessel/integration/driver/.
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

WORKDIR /driver
# Install Google Chrome (has the proprietary H.264 decoder) in addition to
# Chromium. The bundled Chromium lacks H.264, which mediamtx requires for
# WHEP; the driver launches with channel=chrome.
RUN pip install --no-cache-dir playwright==1.48.0 \
    && python -m playwright install chromium \
    && python -m playwright install chrome

COPY run_tests.py /driver/run_tests.py
COPY entrypoint.sh /driver/entrypoint.sh
RUN chmod +x /driver/entrypoint.sh

ENTRYPOINT ["/driver/entrypoint.sh"]
