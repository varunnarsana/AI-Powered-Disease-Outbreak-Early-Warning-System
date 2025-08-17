# IAM Permissions and Terraform Environments

This document outlines the required IAM permissions for the Disease Outbreak Early Warning System and provides guidance on setting up different Terraform environments.

## Table of Contents
- [IAM Permissions](#iam-permissions)
  - [AWS Services](#aws-services)
  - [Service-Specific Permissions](#service-specific-permissions)
- [Terraform Environments](#terraform-environments)
  - [Development](#development)
  - [Staging](#staging)
  - [Production](#production)
- [Security Best Practices](#security-best-practices)

## IAM Permissions

### AWS Services

The following AWS services are used by the application:

| Service | Purpose |
|---------|---------|
| ECS/EKS | Container orchestration |
| RDS | PostgreSQL database |
| ElastiCache | Redis caching |
| S3 | Model and data storage |
| IAM | Identity and access management |
| CloudWatch | Logging and monitoring |
| KMS | Key management |
| VPC | Networking |
| ECR | Container registry |

### Service-Specific Permissions

#### API Service IAM Role

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::disease-outbreak-models/*",
                "arn:aws:s3:::disease-outbreak-models"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ssm:GetParameter",
                "ssm:GetParameters"
            ],
            "Resource": "arn:aws:ssm:*:*:parameter/disease-outbreak/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt"
            ],
            "Resource": "arn:aws:kms:*:*:key/your-kms-key-id"
        }
    ]
}
```

#### ML Training Job IAM Role

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::disease-outbreak-data/*",
                "arn:aws:s3:::disease-outbreak-data",
                "arn:aws:s3:::disease-outbreak-models/*",
                "arn:aws:s3:::disease-outbreak-models"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability"
            ],
            "Resource": "arn:aws:ecr:*:*:repository/disease-outbreak-ml"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

## Terraform Environments

The project uses a multi-environment setup with the following structure:

```
infrastructure/
├── modules/              # Reusable infrastructure modules
│   ├── networking/
│   ├── database/
│   ├── compute/
│   └── monitoring/
├── environments/
│   ├── dev/              # Development environment
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/          # Staging environment
│   └── prod/             # Production environment
└── scripts/              # Helper scripts
```

### Development

**Purpose**: Local development and testing

**Configuration**:
- Single availability zone
- Minimal instance sizes
- Public access for development
- Auto-shutdown schedules

**Variables (dev/terraform.tfvars)**:
```hcl
environment           = "dev"
region                = "us-east-1"
instance_type         = "t3.medium"
db_instance_class     = "db.t3.micro"
enable_public_access  = true
enable_backup         = false
monitoring_enabled    = true
```

### Staging

**Purpose**: Pre-production testing

**Configuration**:
- Multi-AZ for high availability
- Similar to production but with smaller instances
- Isolated VPC with limited external access
- Automated testing integration

**Variables (staging/terraform.tfvars)**:
```hcl
environment           = "staging"
region                = "us-east-1"
instance_type         = "t3.large"
db_instance_class     = "db.m5.large"
replica_count         = 2
enable_public_access  = false
enable_backup         = true
backup_retention      = 7
monitoring_enabled    = true
```

### Production

**Purpose**: Live environment

**Configuration**:
- Multi-region deployment
- Maximum availability and redundancy
- Strict security controls
- Comprehensive monitoring and alerting
- Automated backups with long retention

**Variables (prod/terraform.tfvars)**:
```hcl
environment           = "prod"
region                = "us-east-1"
instance_type         = "m5.xlarge"
db_instance_class     = "db.r5.2xlarge"
replica_count         = 3
enable_public_access  = false
enable_backup         = true
backup_retention      = 35
monitoring_enabled    = true
enable_waf            = true
```

## Security Best Practices

1. **Principle of Least Privilege**:
   - Grant only the permissions required for each component
   - Use separate IAM roles for different services

2. **Secrets Management**:
   - Store secrets in AWS Secrets Manager or Parameter Store
   - Rotate credentials regularly
   - Use KMS for encryption at rest

3. **Network Security**:
   - Use private subnets for database and backend services
   - Implement security groups and network ACLs
   - Enable VPC flow logs

4. **Monitoring and Logging**:
   - Enable CloudTrail for API activity logging
   - Set up CloudWatch Alarms for security events
   - Monitor IAM access patterns

5. **Compliance**:
   - Enable AWS Config for compliance monitoring
   - Implement AWS GuardDuty for threat detection
   - Regular security audits and penetration testing

6. **Infrastructure as Code**:
   - Use Terraform modules for consistent deployments
   - Implement state file encryption and access controls
   - Use remote state with locking

7. **Disaster Recovery**:
   - Regular backups with testing
   - Cross-region replication for critical data
   - Documented recovery procedures

8. **Update and Patch Management**:
   - Regular OS and software updates
   - Automated patch management
   - Vulnerability scanning

## Deployment Process

1. **Development Environment**:
   ```bash
   cd infrastructure/environments/dev
   terraform init
   terraform plan -var-file=terraform.tfvars
   terraform apply -var-file=terraform.tfvars
   ```

2. **Promote to Staging**:
   - Merge changes to staging branch
   - CI/CD pipeline deploys to staging
   - Run integration tests

3. **Promote to Production**:
   - Create a release tag
   - Manual approval for production deployment
   - Blue/green deployment with traffic shifting

## Cost Management

- Use cost allocation tags
- Set up billing alerts
- Use reserved instances for stable workloads
- Implement auto-scaling for variable workloads
- Regular cost optimization reviews

## Maintenance

- Regular security patching
- Performance tuning
- Capacity planning
- Documentation updates
- Staff training on security practices
