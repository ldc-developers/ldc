#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <tag> <file> (and GITHUB_TOKEN environment variable)"
  exit 1
fi

releaseTag=$1
artifact=$2
artifactFilename=$(basename $artifact)

releaseID="$(bash -c "curl -s https://api.github.com/repos/ldc-developers/ldc/releases/tags/$releaseTag | grep -m 1 '^  \"id\":'")"
releaseID=${releaseID:8:-1}

echo "Uploading $artifact to GitHub release $releaseTag ($releaseID)..."
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @$artifact \
  https://uploads.github.com/repos/ldc-developers/ldc/releases/$releaseID/assets?name=$artifactFilename
