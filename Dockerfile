# Utilize the Python 3.11 base image to match your GitHub Actions runner
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository into the container
COPY . .

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Execute Gunicorn bound to port 7860 with memory-safe worker configurations
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
