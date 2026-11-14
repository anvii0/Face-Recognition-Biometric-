"""Create (idempotently) the AWS data resources: private SSE S3 bucket + Rekognition collection.

    .venv/bin/python scripts/aws_resources.py            # create / ensure, print names
    .venv/bin/python scripts/aws_resources.py --destroy  # empty + delete bucket, delete collection
Names derive from the account id, so deploy.sh/teardown.sh recompute them without state files.
"""
from __future__ import annotations

import argparse
import json

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
COLLECTION = "hid-liveness-faces"
RETENTION_DAYS = 3  # sandbox lives ~2 days; biometric data must not outlive the demo


def names() -> dict[str, str]:
    acct = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return {"bucket": f"hid-liveness-{acct}", "collection": COLLECTION, "region": REGION}


def ensure() -> dict[str, str]:
    n = names()
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.create_bucket(Bucket=n["bucket"])
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou",):
            raise
    s3.put_public_access_block(Bucket=n["bucket"], PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(Bucket=n["bucket"], ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}, "BucketKeyEnabled": True}]})
    s3.put_bucket_lifecycle_configuration(Bucket=n["bucket"], LifecycleConfiguration={"Rules": [
        {"ID": "expire-biometrics", "Status": "Enabled", "Filter": {"Prefix": ""},
         "Expiration": {"Days": RETENTION_DAYS}}]})
    s3.put_bucket_policy(Bucket=n["bucket"], Policy=json.dumps({"Version": "2012-10-17", "Statement": [{
        "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
        "Resource": [f"arn:aws:s3:::{n['bucket']}", f"arn:aws:s3:::{n['bucket']}/*"],
        "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}))
    rek = boto3.client("rekognition", region_name=REGION)
    try:
        rek.create_collection(CollectionId=n["collection"])
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
    return n


def destroy() -> None:
    n = names()
    s3 = boto3.resource("s3", region_name=REGION)
    try:
        b = s3.Bucket(n["bucket"])
        b.object_versions.delete()
        b.objects.all().delete()
        b.delete()
        print("deleted bucket", n["bucket"])
    except ClientError as e:
        print("bucket:", e.response["Error"]["Code"])
    try:
        boto3.client("rekognition", region_name=REGION).delete_collection(CollectionId=n["collection"])
        print("deleted collection", n["collection"])
    except ClientError as e:
        print("collection:", e.response["Error"]["Code"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--destroy", action="store_true")
    a = ap.parse_args()
    if a.destroy:
        destroy()
    else:
        print(json.dumps(ensure()))
