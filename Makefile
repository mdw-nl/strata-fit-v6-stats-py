# Docker image
IMAGE_NAME ?= ghcr.io/mdw-nl/v6-basic-stats-py
TAG ?= latest
DOCKERFILE ?= ./Dockerfile

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(TAG) -f $(DOCKERFILE) .

# Push the Docker image to registry
.PHONY: push
push:
	docker push $(IMAGE_NAME):$(TAG)
