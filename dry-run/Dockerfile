# Build from the repository root:
#   docker build -f dry-run/Dockerfile -t not-a-robot-demo .
#   docker run --rm -p 8000:8000 not-a-robot-demo
# Then open http://localhost:8000

FROM python:3.12-slim

WORKDIR /app

# Chromium + its matching chromedriver, for the real-browser check
# behind /api/run_real_browser_checks in app.py: launches an actual
# headless Chromium via Selenium and captures its real signals, rather
# than only scoring a hardcoded example dict. Pinning to the apt-shipped
# chromedriver (vs. Selenium Manager downloading one at runtime) keeps
# this image usable offline.
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        chromium-driver \
    && rm -rf /var/lib/apt/lists/*
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_BIN=/usr/bin/chromedriver

# not_a_robot comes from PyPI (dry-run/requirements.txt pins the version
# this demo was built against), like any real consumer of the package.
COPY dry-run/requirements.txt ./dry-run/requirements.txt
RUN pip install --no-cache-dir -r dry-run/requirements.txt

# examples/synthetic_data.py supplies the demo's synthetic archetypes
# (baseline seeding + the "simulate" buttons). It's demo-only -- not
# part of the published package (only src/ is) -- so it's copied in
# from this checkout separately.
COPY examples ./examples

COPY dry-run/app.py ./dry-run/app.py
COPY dry-run/static ./dry-run/static
COPY dry-run/capture_real_automation.py ./dry-run/capture_real_automation.py

ENV PYTHONPATH=/app
ENV NOT_A_ROBOT_STORE=/app/dry-run-data
ENV PORT=8000
EXPOSE 8000

CMD ["python", "dry-run/app.py"]
