FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt ./
RUN python -m pip install --no-compile --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
RUN python -m pip install --no-compile --requirement requirements-docker.txt

COPY pyproject.toml LICENSE ./
COPY src ./src
RUN python -m pip install --no-deps .

ENTRYPOINT ["variant-prioritizer"]
CMD ["--help"]
