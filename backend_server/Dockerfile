FROM python:3.12-slim
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . /app

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Explicitly install uvicorn with all extras
RUN poetry add "uvicorn[standard]"

# Add debugging
RUN echo "Installed packages:" && poetry show

# Set environment variable for more verbose logging
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=debug

# Run the app with a single worker for now to debug the issue
CMD ["sh", "-c", "echo 'Starting server...' && poetry run python -c 'import sys; print(sys.path)' && poetry run uvicorn backend_server.main:app --host 0.0.0.0 --log-level debug"]