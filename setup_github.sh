#!/bin/bash
# Setup script for GitHub repository
# This script initializes a git repository and automatically creates and pushes to GitHub

set -e

# Configuration
GITHUB_USERNAME="chenhangcuisg-code"
REPO_NAME="Cross-Architecture-Merging-for-Large-Language-Models"
REPO_DESCRIPTION="Cross-Architecture Merging for Large Language Models"

# Get GitHub token from environment variable or use provided one
GITHUB_TOKEN="${GITHUB_TOKEN:-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}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN is not set"
    echo "Usage: GITHUB_TOKEN=your_token bash setup_github.sh"
    exit 1
fi

echo "=========================================="
echo "Setting up GitHub repository"
echo "Repository: $REPO_NAME"
echo "GitHub User: $GITHUB_USERNAME"
echo "=========================================="

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "[1/6] Initializing git repository..."
    git init
else
    echo "[1/6] Git repository already initialized."
fi

# Configure git user (if not already configured)
if [ -z "$(git config user.name)" ]; then
    echo "[2/6] Configuring git user..."
    git config user.name "$GITHUB_USERNAME"
    git config user.email "${GITHUB_USERNAME}@users.noreply.github.com"
else
    echo "[2/6] Git user already configured."
fi

# Add all files
echo "[3/6] Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "[4/6] No changes to commit."
else
    echo "[4/6] Creating initial commit..."
    git commit -m "Initial commit: Cross-Architecture Merging for Large Language Models"
fi

# Check if remote already exists
if git remote get-url origin &>/dev/null; then
    echo "[5/6] Remote 'origin' already exists, updating..."
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
else
    echo "[5/6] Adding remote repository..."
    git remote add origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
fi

# Create repository on GitHub if it doesn't exist
echo "[6/6] Checking if repository exists on GitHub..."
REPO_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/${GITHUB_USERNAME}/${REPO_NAME}")

if [ "$REPO_EXISTS" = "404" ]; then
    echo "      Creating repository on GitHub..."
    curl -s -X POST \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "{\"name\":\"${REPO_NAME}\",\"description\":\"${REPO_DESCRIPTION}\",\"private\":false}" \
        "https://api.github.com/user/repos" > /dev/null
    echo "      Repository created successfully!"
elif [ "$REPO_EXISTS" = "200" ]; then
    echo "      Repository already exists on GitHub."
else
    echo "      Warning: Could not verify repository status (HTTP $REPO_EXISTS)"
fi

# Set default branch to main
git branch -M main 2>/dev/null || true

# Push to GitHub
echo ""
echo "Pushing code to GitHub..."
git push -u origin main

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Repository URL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo "=========================================="
