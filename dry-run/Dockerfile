# Build from the repository root:
#   docker build -f dry-run/Dockerfile -t not-a-robot-demo .
#   docker run --rm -p 8000:8000 not-a-robot-demo
# Then open http://localhost:8000

FROM python:3.12-slim

WORKDIR /app

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

ENV PYTHONPATH=/app
ENV NOT_A_ROBOT_STORE=/app/dry-run-data
ENV PORT=8000
EXPOSE 8000

CMD ["python", "dry-run/app.py"]
