output "s3_bucket_name" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}

output "ecr_repository_url" {
  description = "ECR repository url"
  value       = aws_ecr_repository.ecr_rep_for_img.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.concrete_crack_detection_lambda.function_name
}

output "api_endpoint" {
  description = "URL for testing: <api_endpoint>/predict"
  value       = aws_apigatewayv2_stage.default_stage.invoke_url
}
