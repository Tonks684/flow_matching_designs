# Dockerfile

FROM python:3.10-slim

# System-level basics
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc files & make output unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work directory inside the container
WORKDIR /app

# Copy dependency declarations first for layer caching
COPY pyproject.toml requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Now copy the rest of the project
COPY . .

# Default command: just drop into a shell
# You can override this when running the container.
CMD ["/bin/bash"]
