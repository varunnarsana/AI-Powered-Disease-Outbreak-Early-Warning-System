# Terraform variables for Disease Outbreak Early Warning System

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "disease-mlops"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for the region"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "eks_node_group_instance_types" {
  description = "Instance types for EKS node groups"
  type        = map(list(string))
  default = {
    general = ["t3.medium", "t3.large"]
    ml      = ["g4dn.xlarge", "g4dn.2xlarge"]
  }
}

variable "eks_node_group_capacity_types" {
  description = "Capacity types for EKS node groups"
  type        = map(string)
  default = {
    general = "ON_DEMAND"
    ml      = "ON_DEMAND"
  }
}

variable "eks_node_group_desired_sizes" {
  description = "Desired sizes for EKS node groups"
  type        = map(number)
  default = {
    general = 2
    ml      = 1
  }
}

variable "eks_node_group_min_sizes" {
  description = "Minimum sizes for EKS node groups"
  type        = map(number)
  default = {
    general = 1
    ml      = 1
  }
}

variable "eks_node_group_max_sizes" {
  description = "Maximum sizes for EKS node groups"
  type        = map(number)
  default = {
    general = 5
    ml      = 3
  }
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 100
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_clusters" {
  description = "Number of Redis cache clusters"
  type        = number
  default     = 1
}

variable "kafka_instance_type" {
  description = "MSK Kafka broker instance type"
  type        = string
  default     = "kafka.t3.small"
}

variable "kafka_number_of_broker_nodes" {
  description = "Number of Kafka broker nodes"
  type        = number
  default     = 3
}

variable "kafka_storage_size" {
  description = "Kafka broker storage size in GB"
  type        = number
  default     = 100
}

variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 7
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default = {
    Owner       = "MLOps Team"
    CostCenter  = "Research"
    Project     = "Disease Outbreak Early Warning"
    Environment = "Development"
  }
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "SSL certificate ARN for HTTPS"
  type        = string
  default     = ""
}

variable "enable_https" {
  description = "Enable HTTPS for the application"
  type        = bool
  default     = false
}

variable "enable_waf" {
  description = "Enable AWS WAF for security"
  type        = bool
  default     = false
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs for network monitoring"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = false
}

variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = false
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms for monitoring"
  type        = bool
  default     = true
}

variable "enable_sns_notifications" {
  description = "Enable SNS notifications for alerts"
  type        = bool
  default     = true
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for notifications"
  type        = string
  default     = ""
}

variable "enable_slack_notifications" {
  description = "Enable Slack notifications for alerts"
  type        = bool
  default     = false
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_email_notifications" {
  description = "Enable email notifications for alerts"
  type        = bool
  default     = false
}

variable "ses_email_address" {
  description = "SES verified email address for notifications"
  type        = string
  default     = ""
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for EKS node groups"
  type        = bool
  default     = true
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler for EKS"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable metrics server for Kubernetes"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_jaeger" {
  description = "Enable Jaeger distributed tracing"
  type        = bool
  default     = false
}

variable "enable_elasticsearch" {
  description = "Enable Elasticsearch for log aggregation"
  type        = bool
  default     = true
}

variable "enable_kibana" {
  description = "Enable Kibana for log visualization"
  type        = bool
  default     = true
}

variable "enable_mlflow" {
  description = "Enable MLflow for experiment tracking"
  type        = bool
  default     = true
}

variable "enable_airflow" {
  description = "Enable Apache Airflow for workflow orchestration"
  type        = bool
  default     = false
}

variable "enable_spark" {
  description = "Enable Apache Spark for data processing"
  type        = bool
  default     = true
}

variable "enable_kafka" {
  description = "Enable Apache Kafka for streaming"
  type        = bool
  default     = true
}

variable "enable_redis" {
  description = "Enable Redis for caching"
  type        = bool
  default     = true
}

variable "enable_postgresql" {
  description = "Enable PostgreSQL database"
  type        = bool
  default     = true
}

variable "enable_s3" {
  description = "Enable S3 for object storage"
  type        = bool
  default     = true
}

variable "enable_cloudfront" {
  description = "Enable CloudFront for content delivery"
  type        = bool
  default     = false
}

variable "enable_route53" {
  description = "Enable Route53 for DNS management"
  type        = bool
  default     = false
}

variable "enable_load_balancer" {
  description = "Enable Application Load Balancer"
  type        = bool
  default     = true
}

variable "enable_api_gateway" {
  description = "Enable API Gateway"
  type        = bool
  default     = false
}

variable "enable_lambda" {
  description = "Enable AWS Lambda functions"
  type        = bool
  default     = false
}

variable "enable_step_functions" {
  description = "Enable AWS Step Functions"
  type        = bool
  default     = false
}

variable "enable_eventbridge" {
  description = "Enable AWS EventBridge"
  type        = bool
  default     = false
}

variable "enable_sqs" {
  description = "Enable AWS SQS for message queuing"
  type        = bool
  default     = false
}

variable "enable_sns" {
  description = "Enable AWS SNS for notifications"
  type        = bool
  default     = true
}

variable "enable_cloudwatch" {
  description = "Enable AWS CloudWatch for monitoring"
  type        = bool
  default     = true
}

variable "enable_xray" {
  description = "Enable AWS X-Ray for tracing"
  type        = bool
  default     = false
}

variable "enable_secrets_manager" {
  description = "Enable AWS Secrets Manager"
  type        = bool
  default     = false
}

variable "enable_parameter_store" {
  description = "Enable AWS Systems Manager Parameter Store"
  type        = bool
  default     = true
}

variable "enable_kms" {
  description = "Enable AWS KMS for encryption"
  type        = bool
  default     = true
}

variable "enable_iam" {
  description = "Enable AWS IAM for access management"
  type        = bool
  default     = true
}

variable "enable_vpc" {
  description = "Enable VPC for networking"
  type        = bool
  default     = true
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_internet_gateway" {
  description = "Enable Internet Gateway for public subnets"
  type        = bool
  default     = true
}

variable "enable_vpc_endpoints" {
  description = "Enable VPC endpoints for AWS services"
  type        = bool
  default     = false
}

variable "enable_vpn" {
  description = "Enable VPN connection"
  type        = bool
  default     = false
}

variable "enable_direct_connect" {
  description = "Enable Direct Connect"
  type        = bool
  default     = false
}

variable "enable_transit_gateway" {
  description = "Enable Transit Gateway"
  type        = bool
  default     = false
}

variable "enable_network_firewall" {
  description = "Enable Network Firewall"
  type        = bool
  default     = false
}

variable "enable_shield" {
  description = "Enable AWS Shield for DDoS protection"
  type        = bool
  default     = false
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = false
}

variable "enable_macie" {
  description = "Enable AWS Macie for data protection"
  type        = bool
  default     = false
}

variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = false
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "enable_organizations" {
  description = "Enable AWS Organizations"
  type        = bool
  default     = false
}

variable "enable_control_tower" {
  description = "Enable AWS Control Tower"
  type        = bool
  default     = false
}

variable "enable_workspaces" {
  description = "Enable AWS WorkSpaces"
  type        = bool
  default     = false
}

variable "enable_appstream" {
  description = "Enable AWS AppStream"
  type        = bool
  default     = false
}

variable "enable_quicksight" {
  description = "Enable Amazon QuickSight"
  type        = bool
  default     = false
}

variable "enable_sagemaker" {
  description = "Enable Amazon SageMaker"
  type        = bool
  default     = false
}

variable "enable_comprehend" {
  description = "Enable Amazon Comprehend"
  type        = bool
  default     = false
}

variable "enable_rekognition" {
  description = "Enable Amazon Rekognition"
  type        = bool
  default     = false
}

variable "enable_translate" {
  description = "Enable Amazon Translate"
  type        = bool
  default     = false
}

variable "enable_polly" {
  description = "Enable Amazon Polly"
  type        = bool
  default     = false
}

variable "enable_lex" {
  description = "Enable Amazon Lex"
  type        = bool
  default     = false
}

variable "enable_connect" {
  description = "Enable Amazon Connect"
  type        = bool
  default     = false
}

variable "enable_chime" {
  description = "Enable Amazon Chime"
  type        = bool
  default     = false
}

variable "enable_workmail" {
  description = "Enable Amazon WorkMail"
  type        = bool
  default     = false
}

variable "enable_ses" {
  description = "Enable Amazon SES"
  type        = bool
  default     = false
}

variable "enable_sns" {
  description = "Enable Amazon SNS"
  type        = bool
  default     = true
}

variable "enable_sqs" {
  description = "Enable Amazon SQS"
  type        = bool
  default     = false
}

variable "enable_eventbridge" {
  description = "Enable Amazon EventBridge"
  type        = bool
  default     = false
}

variable "enable_step_functions" {
  description = "Enable AWS Step Functions"
  type        = bool
  default     = false
}

variable "enable_lambda" {
  description = "Enable AWS Lambda"
  type        = bool
  default     = false
}

variable "enable_batch" {
  description = "Enable AWS Batch"
  type        = bool
  default     = false
}

variable "enable_glue" {
  description = "Enable AWS Glue"
  type        = bool
  default     = false
}

variable "enable_athena" {
  description = "Enable Amazon Athena"
  type        = bool
  default     = false
}

variable "enable_redshift" {
  description = "Enable Amazon Redshift"
  type        = bool
  default     = false
}

variable "enable_elasticsearch" {
  description = "Enable Amazon Elasticsearch Service"
  type        = bool
  default     = false
}

variable "enable_opensearch" {
  description = "Enable Amazon OpenSearch Service"
  type        = bool
  default     = false
}

variable "enable_elasticache" {
  description = "Enable Amazon ElastiCache"
  type        = bool
  default     = true
}

variable "enable_dynamodb" {
  description = "Enable Amazon DynamoDB"
  type        = bool
  default     = false
}

variable "enable_rds" {
  description = "Enable Amazon RDS"
  type        = bool
  default     = true
}

variable "enable_aurora" {
  description = "Enable Amazon Aurora"
  type        = bool
  default     = false
}

variable "enable_documentdb" {
  description = "Enable Amazon DocumentDB"
  type        = bool
  default     = false
}

variable "enable_neptune" {
  description = "Enable Amazon Neptune"
  type        = bool
  default     = false
}

variable "enable_timestream" {
  description = "Enable Amazon Timestream"
  type        = bool
  default     = false
}

variable "enable_quantum_ledger" {
  description = "Enable Amazon Quantum Ledger Database"
  type        = bool
  default     = false
}

variable "enable_managed_blockchain" {
  description = "Enable Amazon Managed Blockchain"
  type        = bool
  default     = false
}

variable "enable_qldb" {
  description = "Enable Amazon QLDB"
  type        = bool
  default     = false
}

variable "enable_keyspaces" {
  description = "Enable Amazon Keyspaces"
  type        = bool
  default     = false
}

variable "enable_memorydb" {
  description = "Enable Amazon MemoryDB for Redis"
  type        = bool
  default     = false
}

variable "enable_fsx" {
  description = "Enable Amazon FSx"
  type        = bool
  default     = false
}

variable "enable_storage_gateway" {
  description = "Enable AWS Storage Gateway"
  type        = bool
  default     = false
}

variable "enable_datasync" {
  description = "Enable AWS DataSync"
  type        = bool
  default     = false
}

variable "enable_transfer" {
  description = "Enable AWS Transfer Family"
  type        = bool
  default     = false
}

variable "enable_snowball" {
  description = "Enable AWS Snowball"
  type        = bool
  default     = false
}

variable "enable_snowmobile" {
  description = "Enable AWS Snowmobile"
  type        = bool
  default     = false
}

variable "enable_snowcone" {
  description = "Enable AWS Snowcone"
  type        = bool
  default     = false
}

variable "enable_snow_edge" {
  description = "Enable AWS Snow Edge"
  type        = bool
  default     = false
}

variable "enable_snow_family" {
  description = "Enable AWS Snow Family"
  type        = bool
  default     = false
}

variable "enable_storage_gateway" {
  description = "Enable AWS Storage Gateway"
  type        = bool
  default     = false
}

variable "enable_datasync" {
  description = "Enable AWS DataSync"
  type        = bool
  default     = false
}

variable "enable_transfer" {
  description = "Enable AWS Transfer Family"
  type        = bool
  default     = false
}

variable "enable_snowball" {
  description = "Enable AWS Snowball"
  type        = bool
  default     = false
}

variable "enable_snowmobile" {
  description = "Enable AWS Snowmobile"
  type        = bool
  default     = false
}

variable "enable_snowcone" {
  description = "Enable AWS Snowcone"
  type        = bool
  default     = false
}

variable "enable_snow_edge" {
  description = "Enable AWS Snow Edge"
  type        = bool
  default     = false
}

variable "enable_snow_family" {
  description = "Enable AWS Snow Family"
  type        = bool
  default     = false
}
