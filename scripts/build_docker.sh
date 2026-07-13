#!/bin/bash
# NGLab Local Docker Build & Debug Script

# 1. Clean up dangling images and build cache
echo "Cleaning up Docker build cache..."
docker builder prune -f 2>/dev/null || docker system prune -f

# 2. Build the image
# We check if buildx is available, otherwise we use standard build
echo "Starting build..."
if docker buildx version >/dev/null 2>&1; then
    echo "Using Docker Buildx..."
    docker buildx build \
      --load \
      --progress=plain \
      -t nglab:debug \
      -f docker/Dockerfile.prod .
else
    echo "Using legacy Docker build (BuildKit enabled)..."
    DOCKER_BUILDKIT=1 docker build \
      --progress=plain \
      -t nglab:debug \
      -f docker/Dockerfile.prod .
fi

# 3. Test if the module is importable (Verification)
if [ $? -eq 0 ]; then
    echo "---------------------------------------------------"
    echo "Build successful! Verifying Rust-Python bindings..."
    docker run --rm nglab:debug python -c "import nglab; print('NGLab successfully loaded!')"
else
    echo "---------------------------------------------------"
    echo "Build failed. Check the logs above."
    echo "Hint: If it says 'no space left', run: docker system prune -a"
fi