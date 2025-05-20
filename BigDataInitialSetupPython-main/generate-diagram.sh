#!/bin/bash
set -e

OUTPUT_DIR="uml"

mkdir -p "$OUTPUT_DIR"
echo "Generating UML diagrams with pyreverse..."

pyreverse -o png -p CalculatorApp src/ -d "$OUTPUT_DIR"

echo "Diagrams saved to $OUTPUT_DIR/"
