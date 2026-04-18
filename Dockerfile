# 1. Start from a base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependency file first (for caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY app.py .
COPY models/ ./models/

# 6. Tell Docker app uses port 8000
EXPOSE 8000

# 7. Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]