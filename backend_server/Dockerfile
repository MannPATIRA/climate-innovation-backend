FROM python:3.12-slim
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . /app

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

RUN ls

# Run the app with Gunicorn
CMD ["poetry", "run", "gunicorn", "backend_server.main:app", "--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker", "--preload", "--bind", "0.0.0.0:8000", "--timeout", "120"]