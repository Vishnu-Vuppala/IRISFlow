# FROM python:3.12-slim
FROM python:3.12.4

WORKDIR /app

# Copy requirements (create requirements.txt if not present)
COPY requirements.txt .


# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy your API code
COPY src/housing_api.py .
COPY src/iris_api.py .

COPY housing_model ./housing_model

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "housing_api:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "iris_api:app", "--host", "0.0.0.0", "--port", "8000"]