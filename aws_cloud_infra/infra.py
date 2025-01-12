import boto3
from botocore.exceptions import ClientError
import os

# AWS parameters (loaded from environment variables)
AWS_REGION = os.getenv("AWS_REGION")  # AWS region
REPO_NAME = os.getenv("ECR_REPO_NAME")  # ECR repository name
CLUSTER_NAME = os.getenv("ECS_CLUSTER_NAME")  # ECS cluster name
SERVICE_NAME = os.getenv("ECS_SERVICE_NAME")  # ECS service name
TASK_FAMILY = os.getenv("ECS_TASK_FAMILY")  # ECS task family name
SUBNET_IDS = os.getenv("ECS_SUBNET_IDS", "").split(",")  # Comma-separated subnet IDs
SECURITY_GROUP_IDS = os.getenv("ECS_SECURITY_GROUP_IDS", "").split(",")  # Comma-separated security group IDs
EXECUTION_ROLE_ARN = os.getenv("ECS_EXECUTION_ROLE_ARN")  # ECS task execution role ARN

def create_ecr_repository():
    """Create an Amazon Elastic Container Registry (ECR) repository."""
    ecr_client = boto3.client("ecr", region_name=AWS_REGION)
    try:
        response = ecr_client.create_repository(repositoryName=REPO_NAME)
        print(f"ECR Repository Created: {response['repository']['repositoryUri']}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"ECR Repository '{REPO_NAME}' already exists.")

def create_ecs_cluster():
    """Create an Amazon Elastic Container Service (ECS) cluster."""
    ecs_client = boto3.client("ecs", region_name=AWS_REGION)
    try:
        response = ecs_client.create_cluster(clusterName=CLUSTER_NAME)
        print(f"ECS Cluster Created: {response['cluster']['clusterName']}")
    except ClientError as e:
        print(f"Error creating ECS cluster: {e}")

def create_task_definition():
    """Register a new task definition for the ECS service with multiple containers."""
    ecs_client = boto3.client("ecs", region_name=AWS_REGION)
    try:
        response = ecs_client.register_task_definition(
            family=TASK_FAMILY,
            containerDefinitions=[
                {
                    "name": "sales-forecasting-app",
                    "image": os.getenv("SALES_FORECASTING_IMAGE_URI"),  # Docker image URI for the app
                    "memory": 512,
                    "cpu": 256,
                    "essential": True,
                    "portMappings": [
                        {"containerPort": 80, "hostPort": 80, "protocol": "tcp"}
                    ],
                    "environment": [
                        {"name": "MLFLOW_TRACKING_URI", "value": os.getenv("MLFLOW_TRACKING_URI")},
                        {"name": "AWS_REGION", "value": AWS_REGION},
                    ],
                },
                {
                    "name": "mlflow-server",
                    "image": os.getenv("MLFLOW_IMAGE_URI"),  # Docker image URI for the MLflow server
                    "memory": 512,
                    "cpu": 256,
                    "essential": True,
                    "portMappings": [
                        {"containerPort": 5000, "hostPort": 5000, "protocol": "tcp"}
                    ],
                }
            ],
            requiresCompatibilities=["FARGATE"],
            networkMode="awsvpc",
            cpu="512",
            memory="1024",
            executionRoleArn=EXECUTION_ROLE_ARN,
        )
        print(f"Task Definition Created: {response['taskDefinition']['taskDefinitionArn']}")
    except ClientError as e:
        print(f"Error creating task definition: {e}")

def create_ecs_service():
    """Create or update an ECS service."""
    ecs_client = boto3.client("ecs", region_name=AWS_REGION)
    try:
        response = ecs_client.create_service(
            cluster=CLUSTER_NAME,
            serviceName=SERVICE_NAME,
            taskDefinition=TASK_FAMILY,
            desiredCount=1,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNET_IDS,
                    "securityGroups": SECURITY_GROUP_IDS,
                    "assignPublicIp": "ENABLED",
                }
            },
        )
        print(f"ECS Service Created: {response['service']['serviceName']}\nEnsure the service is running as expected.")
    except ecs_client.exceptions.ServiceAlreadyExistsException:
        print(f"ECS Service '{SERVICE_NAME}' already exists. Updating service...")
        ecs_client.update_service(
            cluster=CLUSTER_NAME,
            service=SERVICE_NAME,
            taskDefinition=TASK_FAMILY,
        )
        print(f"ECS Service '{SERVICE_NAME}' updated to use task definition '{TASK_FAMILY}'.")

if __name__ == "__main__":
    create_ecr_repository()  # Step 1: Create ECR repository
    create_ecs_cluster()     # Step 2: Create ECS cluster
    create_task_definition() # Step 3: Register ECS task definition
    create_ecs_service()     # Step 4: Create or update ECS service
