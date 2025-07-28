FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python - <<EOF
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
EOF

# Pre-download embedding model
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
# load and save embedding model locally
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model.save("/app/embed_model")
EOF

# Copy application code
COPY . .

# Preload binary classification model to cache it in the image
RUN python - <<EOF
import joblib
# load the pre-trained classifier to ensure availability
joblib.load('model/lgbm_binary_classifier.pkl')
print('✅ Classifier model loaded')
EOF

# Expose a command-line entrypoint for processing
ENTRYPOINT ["python", "main.py"]

# Default command arguments placeholder
CMD ["challenge1b_input.json", "Collection 1/PDFs", "challenge1b_output.json"]
