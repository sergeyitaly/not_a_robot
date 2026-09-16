# Build from the repository root:
#   docker build -f dry-run/Dockerfile -t not-a-robot-demo .
#   docker run --rm -p 8000:8000 not-a-robot-demo
# Then open http://localhost:8000

FROM python:3.12-slim

WORKDIR /app

# Install not_a_robot from this checkout (not from PyPI), so the demo
# always reflects the current source, not the last published release.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# examples/synthetic_data.py supplies the demo's synthetic archetypes
# (baseline seeding + the "simulate" buttons).
COPY examples ./examples

COPY dry-run/requirements.txt ./dry-run/requirements.txt
RUN pip install --no-cache-dir -r dry-run/requirements.txt

COPY dry-run/app.py ./dry-run/app.py
COPY dry-run/static ./dry-run/static

ENV PYTHONPATH=/app
ENV NOT_A_ROBOT_STORE=/app/dry-run-data
ENV PORT=8000
EXPOSE 8000

CMD ["python", "dry-run/app.py"]
