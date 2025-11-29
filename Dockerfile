FROM python:3.11-slim

# Install system packages needed for OCR *and* building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    imagemagick \
    ghostscript \
    libmagickwand-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy to allow PDF processing
RUN if [ -f /etc/ImageMagick-6/policy.xml ]; then \
        sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml; \
    fi

WORKDIR /app

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/google_tokens

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure data directory has correct permissions
RUN chmod -R 755 /app/data

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    TESSERACT_CMD=/usr/bin/tesseract

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
