#!/bin/bash

# Script to setup JWT RSA key files from environment variables
# This script checks if private.pem and public.pem exist in the keys directory
# If not, it reads from environment variables and creates them

set -e  # Exit on error

# Run database migrations
echo "Running database migrations..."
alembic upgrade head


echo "Setting up JWT RSA keys..."
# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Keys directory path
KEYS_DIR="keys"
PUBLIC_KEY_FILE="${KEYS_DIR}/public.pem"


# Create keys directory if it doesn't exist
if [ ! -d "$KEYS_DIR" ]; then
    echo "Creating keys directory..."
    mkdir -p "$KEYS_DIR"
    echo -e "${GREEN}✓${NC} Keys directory created"
else
    echo -e "${GREEN}✓${NC} Keys directory exists"
fi


# Check and create public key
if [ -f "$PUBLIC_KEY_FILE" ]; then
    echo -e "${GREEN}✓${NC} public.pem already exists"
else
    echo -e "${YELLOW}⚠${NC}  public.pem not found"

    if [ -z "$JWT_PUBLIC_KEY" ]; then
        echo -e "${RED}✗${NC} JWT_PUBLIC_KEY environment variable is not set"
        echo ""
        echo "Please set the JWT_PUBLIC_KEY environment variable with your public key content"
        echo "Example:"
        echo "  export JWT_PUBLIC_KEY=\$'-----BEGIN PUBLIC KEY-----\\n...\\n-----END PUBLIC KEY-----'"
        exit 1
    fi

    echo "Creating public.pem from JWT_PUBLIC_KEY environment variable..."
    echo -e "$JWT_PUBLIC_KEY" > "$PUBLIC_KEY_FILE"
    chmod 600 "$PUBLIC_KEY_FILE"
    echo -e "${GREEN}✓${NC} public.pem created with permissions 600"
fi

# Setup Prometheus multiprocess directory
echo "Setting up Prometheus metrics directory..."
PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc_dir}"

# Clean up old metrics files from previous runs
if [ -d "$PROMETHEUS_MULTIPROC_DIR" ]; then
    echo "Cleaning up old metrics files..."
    rm -rf "${PROMETHEUS_MULTIPROC_DIR}"/*
    echo -e "${GREEN}✓${NC} Old metrics files cleaned"
fi

# Create the directory if it doesn't exist
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
echo -e "${GREEN}✓${NC} Prometheus metrics directory created at: ${PROMETHEUS_MULTIPROC_DIR}"

# Export the environment variable for the application
export PROMETHEUS_MULTIPROC_DIR

# Command to run the application
python main.py
