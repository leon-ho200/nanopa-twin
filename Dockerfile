FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY nanopa_twin ./nanopa_twin

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "nanopa_twin.dial"]
CMD ["fit", "--preset", "_smoke", "--device", "cpu"]
