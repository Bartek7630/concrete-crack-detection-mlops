.PHONY: lint format test test-unit test-integration docker-build docker-push tf-plan tf-apply
SHELL := sh

AWS_REGION ?= eu-north-1
AWS_ACCOUNT_ID ?= 534344665048
ECR_REPO_NAME ?= concrete-crack-detector
IMAGE_TAG ?= latest
ECR_URI := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO_NAME)

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

test:
	pytest

test-unit:
	pytest tests/test_prepare_data.py tests/test_lambda_function.py

test-integration:
	pytest tests/test_lambda_integration.py

docker-build:
	docker build -t $(ECR_REPO_NAME):$(IMAGE_TAG) lambda_model/

docker-push: docker-build
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(ECR_REPO_NAME):$(IMAGE_TAG) $(ECR_URI):$(IMAGE_TAG)
	docker push $(ECR_URI):$(IMAGE_TAG)

tf-plan:
	cd terraform && terraform plan

tf-apply:
	cd terraform && terraform apply