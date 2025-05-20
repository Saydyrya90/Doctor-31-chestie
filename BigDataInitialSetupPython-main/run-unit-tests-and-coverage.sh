#!/bin/bash
set -e

export PYTHONPATH=src
coverage run -m pytest tests/
coverage report
coverage html
