import os
from fastapi import UploadFile
import boto3
from uuid import uuid4
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# S3 클라이언트 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
REGION = os.getenv("AWS_REGION")

def upload_to_s3(file: UploadFile, content_type: str) -> str:
    print(f"Uploading file: {file.filename}")
    unique_filename = f"{uuid4()}.png"
    try:
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET_NAME,
            unique_filename,
            ExtraArgs={
                "ContentType": content_type
            }
        )
        file_url = f"https://{S3_BUCKET_NAME}.s3.{REGION}.amazonaws.com/{unique_filename}"
        print(f"File uploaded: {file_url}")
        return file_url
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None
