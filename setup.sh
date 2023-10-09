#!/bin/bash
# Create a virtual environment (if not already created)

# Check if virtual environment already exists
if [ -d "env" ]; then
    echo "Virtual environment already exists"
    # Create boolean variable to indicate that virtual environment already exists
    exists=true
else
    echo "Creating a virtual environment"
    python -m venv env
fi

# Activate the virtual environment
if [ "$(uname)" == "Darwin" ] || [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    source env/bin/activate
else
    echo "Activating virtual environment"
    source env/Scripts/activate
fi


# Install packages from requirements.txt
if [ "$exists" = true ] ; then
    echo ""
else
    echo "Installing packages from requirements.txt"
    pip install -r requirements.txt
fi
