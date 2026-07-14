#!/usr/bin/env bash

set -Eeuo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

COMMIT_MESSAGE="${1:-}"

if [[ -z "$COMMIT_MESSAGE" ]]; then
    echo "Usage:"
    echo '  ./scripts/publish.sh "your commit message"'
    exit 1
fi

BRANCH="$(git branch --show-current)"

if [[ -z "$BRANCH" ]]; then
    echo "Error: detached HEAD detected."
    exit 1
fi

echo "Repository: $REPOSITORY_ROOT"
echo "Branch:     $BRANCH"

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo
echo "1/6 Running Black..."
python -m black --check src tests scripts

echo
echo "2/6 Running Ruff..."
python -m ruff check src tests scripts

echo
echo "3/6 Running tests..."
python -m pytest tests -q

echo
echo "4/6 Staging changes..."

# Stage tracked modifications and deletions.
git add -u

# Stage intended source and configuration files.
for path in \
    README.md \
    .gitignore \
    requirements.txt \
    requirements-api.txt \
    requirements-dev.txt \
    requirements-airflow.txt \
    src \
    tests \
    scripts \
    airflow \
    feature_repo \
    monitoring/prometheus_rules.yml \
    .github/workflows
do
    if [[ -e "$path" ]]; then
        git add "$path"
    fi
done

# Stage committed model artifacts when they exist.
for artifact in \
    models/churn_model.pkl \
    models/tuned_model.pkl \
    models/scaler.pkl \
    models/optimal_threshold.txt \
    models/feature_names.txt \
    models/model_metadata.json \
    models/registry_metadata.json
do
    if [[ -f "$artifact" ]]; then
        git add -f "$artifact"
    fi
done

if git diff --cached --quiet; then
    echo "No staged changes to commit."
    exit 0
fi

echo
echo "Staged changes:"
git diff --cached --stat

echo
echo "5/6 Creating commit..."
git commit -m "$COMMIT_MESSAGE"

echo
echo "6/6 Pushing branch..."
git push -u origin "$BRANCH"

echo
echo "Completed successfully."
echo "Branch: $BRANCH"
echo "Commit: $(git rev-parse HEAD)"
