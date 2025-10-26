#!/bin/bash

echo "🚀 Setting up virtual environment for Abstract Algebra Modeling..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📚 Installing required packages..."
pip install pandas openai

# Test installation
echo "🧪 Testing installation..."
python -c "import pandas; import openai; print('✅ All packages installed successfully!')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "To run the modeling script:"
echo "  python modeling_abstract_algebra.py"
