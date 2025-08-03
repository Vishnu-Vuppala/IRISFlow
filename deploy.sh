#!/bin/bash

echo "Stopping existing container if running..."
docker stop housing-api || true
docker rm housing-api || true

echo "Running new container..."
docker run -d -p 8000:8000 --name housing-api yourdockerhubusername/housing-api:latest
echo "Container is running. You can access the API at http://localhost:8000"