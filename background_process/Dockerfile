FROM python:3.12-slim
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . /app

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

RUN ls

# Run the app
CMD ["poetry", "run", "python", "background_process/main.py"]