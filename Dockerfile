FROM python:3.12-slim
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . /app

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Run the app
CMD ["poetry", "run", "app/main.py"]