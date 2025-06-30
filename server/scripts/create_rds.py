#!/usr/bin/env python3
"""
Script to create an Amazon RDS PostgreSQL instance for storing image embeddings.

This script uses the AWS SDK (boto3) to:
1. Create a security group for the RDS instance
2. Create a new RDS PostgreSQL instance
3. Wait for the instance to become available
4. Print connection information

Prerequisites:
- AWS CLI configured with appropriate credentials
- Required Python packages: boto3, dotenv

Usage:
python create_rds.py
"""

import boto3
import os
import time
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
DB_INSTANCE_IDENTIFIER = os.getenv('DB_INSTANCE_IDENTIFIER', 'happy-ocean-times-db')
DB_NAME = os.getenv('DB_NAME', 'happyoceantimes')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_INSTANCE_CLASS = os.getenv('DB_INSTANCE_CLASS', 'db.t3.micro')
SECURITY_GROUP_NAME = 'happy-ocean-times-db-sg'

# Verify that required environment variables are set
if not DB_PASSWORD:
    print("Error: DB_PASSWORD environment variable must be set")
    sys.exit(1)

# Initialize AWS clients
ec2 = boto3.client('ec2', region_name=AWS_REGION)
rds = boto3.client('rds', region_name=AWS_REGION)

def create_security_group():
    """Create a security group for the RDS instance."""
    try:
        # Check if security group already exists
        response = ec2.describe_security_groups(
            Filters=[{'Name': 'group-name', 'Values': [SECURITY_GROUP_NAME]}]
        )
        
        if response['SecurityGroups']:
            print(f"Security group {SECURITY_GROUP_NAME} already exists")
            return response['SecurityGroups'][0]['GroupId']
        
        # Create a new security group
        response = ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description='Security group for Happy Ocean Times RDS instance'
        )
        
        security_group_id = response['GroupId']
        
        # Add an inbound rule for PostgreSQL
        ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        print(f"Created security group {SECURITY_GROUP_NAME} with ID: {security_group_id}")
        return security_group_id
        
    except Exception as e:
        print(f"Error creating security group: {e}")
        sys.exit(1)

def create_rds_instance(security_group_id):
    """Create an RDS PostgreSQL instance."""
    try:
        # Check if RDS instance already exists
        try:
            response = rds.describe_db_instances(
                DBInstanceIdentifier=DB_INSTANCE_IDENTIFIER
            )
            print(f"RDS instance {DB_INSTANCE_IDENTIFIER} already exists")
            return response['DBInstances'][0]
        except rds.exceptions.DBInstanceNotFoundFault:
            pass
        
        # Create a new RDS instance
        response = rds.create_db_instance(
            DBName=DB_NAME,
            DBInstanceIdentifier=DB_INSTANCE_IDENTIFIER,
            AllocatedStorage=20,
            DBInstanceClass=DB_INSTANCE_CLASS,
            Engine='postgres',
            MasterUsername=DB_USER,
            MasterUserPassword=DB_PASSWORD,
            VpcSecurityGroupIds=[security_group_id],
            BackupRetentionPeriod=7,
            Port=5432,
            MultiAZ=False,
            EngineVersion='13.7',
            AutoMinorVersionUpgrade=True,
            PubliclyAccessible=True,
            Tags=[
                {
                    'Key': 'Project',
                    'Value': 'HappyOceanTimes'
                }
            ]
        )
        
        print(f"Creating RDS instance {DB_INSTANCE_IDENTIFIER}...")
        return response['DBInstance']
        
    except Exception as e:
        print(f"Error creating RDS instance: {e}")
        sys.exit(1)

def wait_for_instance_availability(db_instance_identifier):
    """Wait for the RDS instance to become available."""
    print("Waiting for RDS instance to become available...")
    
    while True:
        response = rds.describe_db_instances(
            DBInstanceIdentifier=db_instance_identifier
        )
        
        status = response['DBInstances'][0]['DBInstanceStatus']
        print(f"Current status: {status}")
        
        if status == 'available':
            return response['DBInstances'][0]
        
        if status == 'failed':
            print("RDS instance creation failed")
            sys.exit(1)
        
        time.sleep(30)  # Check status every 30 seconds

def main():
    # Create security group
    security_group_id = create_security_group()
    
    # Create RDS instance
    db_instance = create_rds_instance(security_group_id)
    
    # Wait for the instance to become available
    if db_instance['DBInstanceStatus'] != 'available':
        db_instance = wait_for_instance_availability(DB_INSTANCE_IDENTIFIER)
    
    # Print connection information
    endpoint = db_instance['Endpoint']['Address']
    port = db_instance['Endpoint']['Port']
    
    print("\n=== RDS Instance Created Successfully ===")
    print(f"Instance Identifier: {DB_INSTANCE_IDENTIFIER}")
    print(f"Endpoint: {endpoint}")
    print(f"Port: {port}")
    print(f"Database Name: {DB_NAME}")
    print(f"Username: {DB_USER}")
    print(f"Password: ********")
    
    print("\nAdd the following to your .env file:")
    print(f"DB_HOST={endpoint}")
    print(f"DB_PORT={port}")
    print(f"DB_NAME={DB_NAME}")
    print(f"DB_USER={DB_USER}")
    print("DB_PASSWORD=your_password")

if __name__ == "__main__":
    main()