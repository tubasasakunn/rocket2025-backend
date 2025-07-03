#!/bin/bash
# This script ensures proper Python path configuration

# Export PYTHONPATH to include the current directory
export PYTHONPATH=/app:$PYTHONPATH

# Execute the original command
exec "$@"
