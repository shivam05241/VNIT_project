# Dockerfile for the Streamlit app
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (minimal) and Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Streamlit app working dir
WORKDIR /app

# Expose Streamlit default port
EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true

# Start Streamlit
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
