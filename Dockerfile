FROM python:3.12-slim
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.5.0

# Copy project files
COPY . /app

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Run the app
CMD ["poetry", "run", "app/main.py"]