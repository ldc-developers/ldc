#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <tag> <file> (and GITHUB_TOKEN environment variable)"
  exit 1
fi

releaseTag=$1
artifact=$2
artifactFilename=$(basename $artifact)

releaseID="$(set -eo pipefail; curl -fsS https://api.github.com/repos/ldc-developers/ldc/releases/tags/"$releaseTag" | grep '^  "id":' | head -n1 || echo "<error>")"
if [ "$releaseID" == "<error>" ]; then
  echo "Error: no GitHub release found for tag '$releaseTag'" >&2
  exit 1
fi
releaseID=${releaseID:8:-1}

echo "Uploading $artifact to GitHub release $releaseTag ($releaseID)..."
curl -fsS \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @$artifact \
  https://uploads.github.com/repos/ldc-developers/ldc/releases/$releaseID/assets?name=$artifactFilename
