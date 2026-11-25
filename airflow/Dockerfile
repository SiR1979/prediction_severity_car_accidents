# Airflow image for the reference stage pipeline (compose.airflow.yaml).
# Build context is the repository root; see Dockerfile.dockerignore next to this file.
FROM apache/airflow:3.3.2-python3.11

# The stages run in their own virtual environment with the pinned runtime
# requirements, the same as the runtime image (Dockerfile), so that Airflow's
# own dependencies cannot change the training environment.
USER root
COPY requirements-runtime.txt /opt/runtime-requirements.txt
RUN /usr/python/bin/python3.11 -m venv /opt/runtime \
    && /opt/runtime/bin/python -m pip install --no-cache-dir --disable-pip-version-check -r /opt/runtime-requirements.txt \
    && /opt/runtime/bin/python -m pip check \
    && /opt/runtime/bin/python -m pip freeze > /opt/runtime-requirements-resolved.txt

WORKDIR /opt/airflow/project
COPY src/ ./src/
COPY airflow/gate_failure_demo_criteria.json ./airflow/gate_failure_demo_criteria.json
COPY airflow/dags/ /opt/airflow/dags/

# Commit the image was built from; recorded in each run directory (airflow_run.json).
ARG SOURCE_COMMIT=unknown
ENV SOURCE_COMMIT=${SOURCE_COMMIT} \
    PYTHONDONTWRITEBYTECODE=1
LABEL org.opencontainers.image.revision=${SOURCE_COMMIT}

USER airflow
WORKDIR /opt/airflow
