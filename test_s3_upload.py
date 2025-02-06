import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

def test_s3_upload():
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    
    # Test bucket and file details
    bucket_name = "climate-reports-test"
    test_file_path = "./test_reports/test.pdf"  # Adjust path as needed
    s3_file_name = "test_upload.pdf"
    
    try:
        # Check if bucket exists, if not create it
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} exists")
        except ClientError:
            print(f"Creating bucket {bucket_name}")
            s3_client.create_bucket(Bucket=bucket_name)
        
        # Upload file
        print(f"Uploading {test_file_path} to {bucket_name}/{s3_file_name}")
        with open(test_file_path, 'rb') as file:
            s3_client.upload_fileobj(file, bucket_name, s3_file_name)
        
        print("Upload successful!")
        
        # Verify file exists
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_file_name)
            print(f"Verified: File exists at s3://{bucket_name}/{s3_file_name}")
        except ClientError:
            print("Error: File upload verification failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_s3_upload() 