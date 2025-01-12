import boto3
from botocore.exceptions import ClientError
import os

# AWS parameters (loaded from environment variables)
AWS_REGION = os.getenv("AWS_REGION")  # AWS region where resources will be created
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")  # EC2 instance type for MLflow server
KEY_NAME = os.getenv("KEY_NAME")  # Name of the EC2 key pair
SECURITY_GROUP_NAME = os.getenv("SECURITY_GROUP_NAME")  # Name of the security group
DB_INSTANCE_IDENTIFIER = os.getenv("DB_INSTANCE_IDENTIFIER")  # RDS instance identifier
DB_NAME = os.getenv("DB_NAME")  # Database name for RDS
DB_USERNAME = os.getenv("DB_USERNAME")  # Username for RDS
DB_PASSWORD = os.getenv("DB_PASSWORD")  # Password for RDS
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")  # Name of the S3 bucket for artifacts

# AWS clients
ec2_client = boto3.client("ec2", region_name=AWS_REGION)
rds_client = boto3.client("rds", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)

def create_security_group():
    """Create a security group for the MLflow EC2 instance."""
    try:
        response = ec2_client.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description="Security group for MLflow server",
            VpcId=get_default_vpc_id()
        )
        sg_id = response["GroupId"]
        print(f"Created security group {SECURITY_GROUP_NAME} with ID {sg_id}")

        # Allow SSH, HTTP, and MLflow ports
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 5000, "ToPort": 5000, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            ]
        )
        return sg_id
    except ClientError as e:
        print(f"Error creating security group: {e}")
        raise e

def get_default_vpc_id():
    """Get the default VPC ID."""
    response = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    return response["Vpcs"][0]["VpcId"]

def create_ec2_instance(sg_id):
    """Launch an EC2 instance for the MLflow server."""
    try:
        instance = ec2_client.run_instances(
            ImageId="ami-0c02fb55956c7d316",  # Amazon Linux 2 AMI
            InstanceType=INSTANCE_TYPE,
            KeyName=KEY_NAME,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[sg_id],
            UserData=f"""#!/bin/bash
            sudo yum update -y
            sudo yum install -y python3 git
            pip3 install mlflow boto3 pymysql
            mkdir /mlflow
            mlflow server --backend-store-uri mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_INSTANCE_IDENTIFIER}/{DB_NAME} --default-artifact-root s3://{S3_BUCKET_NAME}/ --host 0.0.0.0 --port 5000
            """
        )
        instance_id = instance["Instances"][0]["InstanceId"]
        print(f"EC2 Instance Created with ID: {instance_id}")
        return instance_id
    except ClientError as e:
        print(f"Error creating EC2 instance: {e}")
        raise e

def create_rds_instance():
    """Create an RDS MySQL instance for MLflow."""
    try:
        response = rds_client.create_db_instance(
            DBInstanceIdentifier=DB_INSTANCE_IDENTIFIER,
            DBName=DB_NAME,
            DBInstanceClass="db.t3.micro",  # Small instance type for testing
            Engine="mysql",
            MasterUsername=DB_USERNAME,
            MasterUserPassword=DB_PASSWORD,
            AllocatedStorage=20,  # 20 GB storage
            PubliclyAccessible=True,  # Allow public access
        )
        print(f"RDS Instance Created: {response['DBInstance']['DBInstanceIdentifier']}")
    except ClientError as e:
        print(f"Error creating RDS instance: {e}")
        raise e

def create_s3_bucket():
    """Create an S3 bucket for MLflow artifacts."""
    try:
        s3_client.create_bucket(
            Bucket=S3_BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
        )
        print(f"S3 Bucket Created: {S3_BUCKET_NAME}")
    except ClientError as e:
        print(f"Error creating S3 bucket: {e}")
        raise e

if __name__ == "__main__":
    sg_id = create_security_group()  # Step 1: Create security group
    create_ec2_instance(sg_id)  # Step 2: Launch EC2 instance
    create_rds_instance()  # Step 3: Create RDS instance
    create_s3_bucket()  # Step 4: Create S3 bucket
