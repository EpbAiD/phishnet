#!/usr/bin/env python3
"""
AWS Infrastructure Setup for PhishNet
======================================
Creates S3 bucket, EC2 instance, and RDS PostgreSQL
Run this locally with AWS credentials configured
"""

import boto3
import time
import json

AWS_REGION = "us-east-1"

def create_s3_bucket():
    """Create S3 bucket for master dataset"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    bucket_name = "phishnet-data"

    try:
        # us-east-1 doesn't need LocationConstraint
        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ S3 bucket created: {bucket_name}")

        # Create folder structure
        s3.put_object(Bucket=bucket_name, Key='master/')
        s3.put_object(Bucket=bucket_name, Key='queue/')
        s3.put_object(Bucket=bucket_name, Key='scripts/')
        s3.put_object(Bucket=bucket_name, Key='backup/')
        print("✅ Folder structure created")

        return bucket_name
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ S3 bucket already exists: {bucket_name}")
        return bucket_name
    except Exception as e:
        print(f"❌ S3 bucket creation failed: {e}")
        # Try with unique suffix
        import random
        bucket_name = f"phishnet-data-{random.randint(1000,9999)}"
        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ S3 bucket created: {bucket_name}")
        return bucket_name


def create_ec2_key_pair():
    """Create EC2 key pair for SSH access"""
    ec2 = boto3.client('ec2', region_name=AWS_REGION)
    key_name = "phishnet-key"

    try:
        response = ec2.create_key_pair(KeyName=key_name)
        # Save private key
        with open(f"{key_name}.pem", 'w') as f:
            f.write(response['KeyMaterial'])
        print(f"✅ Key pair created: {key_name}.pem")
        return key_name
    except ec2.exceptions.ClientError as e:
        if 'InvalidKeyPair.Duplicate' in str(e):
            print(f"✅ Key pair already exists: {key_name}")
            return key_name
        raise e


def create_security_group():
    """Create security group for EC2"""
    ec2 = boto3.client('ec2', region_name=AWS_REGION)

    # Get default VPC
    vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
    vpc_id = vpcs['Vpcs'][0]['VpcId']

    sg_name = "phishnet-sg"

    try:
        response = ec2.create_security_group(
            GroupName=sg_name,
            Description="PhishNet EC2 security group",
            VpcId=vpc_id
        )
        sg_id = response['GroupId']

        # Allow SSH
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        print(f"✅ Security group created: {sg_id}")
        return sg_id
    except ec2.exceptions.ClientError as e:
        if 'InvalidGroup.Duplicate' in str(e):
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = sgs['SecurityGroups'][0]['GroupId']
            print(f"✅ Security group already exists: {sg_id}")
            return sg_id
        raise e


def create_ec2_instance(key_name, sg_id):
    """Create EC2 t2.micro instance"""
    ec2 = boto3.client('ec2', region_name=AWS_REGION)

    # Amazon Linux 2023 AMI (free tier eligible)
    # This AMI ID is for us-east-1, may vary by region
    ami_id = "ami-0c02fb55956c7d316"  # Amazon Linux 2

    # User data script to install Python and dependencies
    user_data = """#!/bin/bash
yum update -y
yum install -y python3 python3-pip git
pip3 install pandas requests python-whois dnspython boto3
mkdir -p /home/ec2-user/phishnet
chown ec2-user:ec2-user /home/ec2-user/phishnet
"""

    try:
        response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType='t2.micro',
            KeyName=key_name,
            SecurityGroupIds=[sg_id],
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [{'Key': 'Name', 'Value': 'phishnet-worker'}]
                }
            ]
        )
        instance_id = response['Instances'][0]['InstanceId']
        print(f"✅ EC2 instance created: {instance_id}")

        # Wait for instance to be running
        print("⏳ Waiting for instance to start...")
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])

        # Get public IP
        instance = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = instance['Reservations'][0]['Instances'][0].get('PublicIpAddress', 'N/A')
        print(f"✅ Instance running. Public IP: {public_ip}")

        return instance_id, public_ip
    except Exception as e:
        print(f"❌ EC2 creation failed: {e}")
        raise e


def create_rds_instance():
    """Create RDS PostgreSQL instance (free tier)"""
    rds = boto3.client('rds', region_name=AWS_REGION)

    db_instance_id = "phishnet-db"

    try:
        response = rds.create_db_instance(
            DBInstanceIdentifier=db_instance_id,
            DBInstanceClass='db.t3.micro',  # Free tier eligible
            Engine='postgres',
            EngineVersion='15.4',
            MasterUsername='phishnet_admin',
            MasterUserPassword='PhishNet2024!Secure',  # Change this!
            AllocatedStorage=20,  # 20 GB (free tier)
            StorageType='gp2',
            PubliclyAccessible=True,
            DBName='phishnet',
            Tags=[{'Key': 'Name', 'Value': 'phishnet-db'}]
        )
        print(f"✅ RDS instance creating: {db_instance_id}")
        print("⏳ RDS takes 5-10 minutes to be available...")

        return db_instance_id
    except rds.exceptions.DBInstanceAlreadyExistsFault:
        print(f"✅ RDS instance already exists: {db_instance_id}")
        return db_instance_id
    except Exception as e:
        print(f"❌ RDS creation failed: {e}")
        raise e


def main():
    print("=" * 60)
    print("AWS INFRASTRUCTURE SETUP FOR PHISHNET")
    print("=" * 60)
    print()

    # 1. Create S3 bucket
    print("📦 Step 1: Creating S3 bucket...")
    bucket_name = create_s3_bucket()
    print()

    # 2. Create EC2 key pair
    print("🔑 Step 2: Creating EC2 key pair...")
    key_name = create_ec2_key_pair()
    print()

    # 3. Create security group
    print("🛡️ Step 3: Creating security group...")
    sg_id = create_security_group()
    print()

    # 4. Create EC2 instance
    print("💻 Step 4: Creating EC2 instance (t2.micro)...")
    instance_id, public_ip = create_ec2_instance(key_name, sg_id)
    print()

    # 5. Create RDS instance
    print("🗄️ Step 5: Creating RDS PostgreSQL...")
    db_instance_id = create_rds_instance()
    print()

    # Summary
    print("=" * 60)
    print("✅ AWS INFRASTRUCTURE SETUP COMPLETE")
    print("=" * 60)
    print(f"""
Resources Created:
  S3 Bucket:     {bucket_name}
  EC2 Instance:  {instance_id} ({public_ip})
  EC2 Key:       {key_name}.pem
  RDS Instance:  {db_instance_id} (creating...)

Next Steps:
  1. Wait for RDS to be available (~5-10 min)
  2. Get RDS endpoint from AWS Console
  3. Update GitHub secrets with new values
  4. Run updated pipeline
""")

    # Save config
    config = {
        "s3_bucket": bucket_name,
        "ec2_instance_id": instance_id,
        "ec2_public_ip": public_ip,
        "rds_instance_id": db_instance_id,
        "region": AWS_REGION
    }
    with open("aws_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("📄 Config saved to aws_config.json")


if __name__ == "__main__":
    main()
