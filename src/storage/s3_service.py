"""
AWS S3 storage for raw PDF archival.

Design note: this module ONLY handles raw bytes in/out of S3 -- it never
touches chunking, embedding, or metadata logic. That separation is what
makes it a clean addition rather than a rewrite: pdf_loader.py's
extract_pages_from_bytes() doesn't care whether bytes came from a local
file, an arXiv download, or S3 -- it's the same function either way.
"""

import os
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
REGION = os.getenv("AWS_REGION")

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _s3_client


def upload_pdf_bytes(pdf_bytes: bytes, key: str) -> str:
    """
    Upload raw PDF bytes to S3 under the given key (e.g. "papers/2606.27287v1.pdf").
    Returns the key on success.
    """
    client = get_s3_client()
    client.put_object(Bucket=BUCKET_NAME, Key=key, Body=pdf_bytes, ContentType="application/pdf")
    return key


def download_pdf_bytes(key: str) -> bytes:
    """Fetch raw PDF bytes back from S3 by key."""
    client = get_s3_client()
    response = client.get_object(Bucket=BUCKET_NAME, Key=key)
    return response["Body"].read()


def list_pdfs(prefix: str = "papers/") -> list:
    """List all archived PDF keys under a prefix."""
    client = get_s3_client()
    response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def pdf_exists(key: str) -> bool:
    client = get_s3_client()
    try:
        client.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError:
        return False