# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /django_project

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the server port
EXPOSE 8000

# Command to start the server
CMD ["gunicorn", "--workers=3","django_project.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "400"]
