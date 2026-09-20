terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-north-1"
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "crack-detection-mlflow-s3-bucket"
}

resource "aws_ecr_repository" "ecr_rep_for_img" {
  name                 = "concrete-crack-detector"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}


resource "aws_iam_role" "iam_for_lambda" {
  name = "iam_for_lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.iam_for_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "lambda-s3-access"
  role = aws_iam_role.iam_for_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject"]
        Resource = [
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_lambda_function" "concrete_crack_detection_lambda" {
  function_name = "crack_detection"
  role          = aws_iam_role.iam_for_lambda.arn

  package_type = "Image"
  image_uri    = "${aws_ecr_repository.ecr_rep_for_img.repository_url}:latest"

  memory_size = 1024
  timeout     = 30

  environment {
    variables = {
      ENVIRONMENT = "production"
      LOG_LEVEL   = "info"
      BUCKET_NAME = aws_s3_bucket.mlflow_artifacts.bucket
    }
  }

  tags = {
    Environment = "production"
    Application = "concrete-crack-detector"
  }
}

resource "aws_apigatewayv2_api" "crack_detection_api" {
  name          = "crack-detection-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.crack_detection_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.concrete_crack_detection_lambda.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict_route" {
  api_id    = aws_apigatewayv2_api.crack_detection_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default_stage" {
  api_id      = aws_apigatewayv2_api.crack_detection_api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "allow_apigateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.concrete_crack_detection_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.crack_detection_api.execution_arn}/*/*"
}
