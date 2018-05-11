#!/bin/bash

set -euxo pipefail

IMAGE=ubuntu:18.04

free -m
docker pull $IMAGE
docker run --rm \
	-v $PWD:/ldc \
	$IMAGE \
	sh -c "cd /ldc && \
	./.semaphoreci/setup.sh && \
	./.semaphoreci/$1.sh"
