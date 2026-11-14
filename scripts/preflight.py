"""Phase 0 preflight: probe which AWS capabilities this sandbox allows.

Creates only throwaway resources prefixed `hid-preflight-` and deletes them.
Never prints credentials. Usage: .venv/bin/python scripts/preflight.py
"""
from __future__ import annotations

import json
import ssl
import struct
import sys
import urllib.request
import uuid
import zlib
from typing import Callable

import boto3
import certifi
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-east-1"
TAG = f"hid-preflight-{uuid.uuid4().hex[:8]}"
results: list[tuple[str, str, str]] = []


def record(name: str, fn: Callable[[], str]) -> None:
    try:
        detail = fn() or ""
        results.append((name, "WORKS", detail))
    except ClientError as e:
        code = e.response["Error"]["Code"]
        # DryRunOperation == authorised; InvalidParameter on a face-less image == API reachable
        if code in ("DryRunOperation",):
            results.append((name, "WORKS", "dry-run authorised"))
        elif code in ("InvalidParameterException", "InvalidImageFormatException"):
            results.append((name, "WORKS", f"API reachable ({code}, expected for synthetic image)"))
        else:
            results.append((name, "BLOCKED", f"{code}: {e.response['Error'].get('Message', '')[:90]}"))
    except Exception as e:  # noqa: BLE001 - preflight wants every failure recorded
        results.append((name, "BLOCKED", f"{type(e).__name__}: {str(e)[:90]}"))


def tiny_png(w: int = 64, h: int = 64) -> bytes:
    """Synthetic grey PNG (no face) — proves API access without any biometric data."""
    raw = b"".join(b"\x00" + bytes([128]) * w * 3 for _ in range(h))
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main() -> int:
    s = boto3.Session(region_name=REGION)
    try:
        ident = s.client("sts").get_caller_identity()
    except (NoCredentialsError, ClientError) as e:
        print(f"STS failed: {e}. Export sandbox credentials in this shell and retry.")
        return 2
    arn = ident["Arn"]
    print(f"Caller: account={ident['Account']} arn={arn.split(':', 5)[-1]}")
    results.append(("sts:GetCallerIdentity", "WORKS", ""))

    rek = s.client("rekognition")
    img = tiny_png()
    record("rekognition:ListCollections", lambda: f"{len(rek.list_collections()['CollectionIds'])} collections")
    record("rekognition:DetectFaces", lambda: f"{len(rek.detect_faces(Image={'Bytes': img})['FaceDetails'])} faces")
    record("rekognition:CompareFaces", lambda: str(len(rek.compare_faces(
        SourceImage={"Bytes": img}, TargetImage={"Bytes": img})["FaceMatches"])) + " matches")

    # S3: real create + delete
    s3 = s.client("s3")
    bucket = TAG

    def s3_test() -> str:
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key="probe.txt", Body=b"ok", ServerSideEncryption="AES256")
        s3.delete_object(Bucket=bucket, Key="probe.txt")
        s3.delete_bucket(Bucket=bucket)
        return "create/put/delete ok"
    record("s3:CreateBucket+Put", s3_test)

    # EC2: dry-run launch of the intended instance type with Ubuntu 24.04 AMI
    ec2 = s.client("ec2")
    ssm = s.client("ssm")
    ami: dict[str, str] = {}

    def ami_lookup() -> str:
        p = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
        ami["id"] = ssm.get_parameter(Name=p)["Parameter"]["Value"]
        return ami["id"]
    record("ssm:GetParameter (Ubuntu AMI)", ami_lookup)
    record("ec2:DescribeVpcs (default)", lambda: str(len(ec2.describe_vpcs(
        Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"])) + " default VPC(s)")
    for itype in ("c6i.large", "t3.large"):
        record(f"ec2:RunInstances dry-run {itype}", lambda it=itype: ec2.run_instances(
            ImageId=ami.get("id", "ami-00000000"), InstanceType=it, MinCount=1, MaxCount=1, DryRun=True) and "")
    record("ec2:CreateSecurityGroup dry-run", lambda: ec2.create_security_group(
        GroupName=TAG, Description="preflight", DryRun=True) and "")

    # IAM: real role + instance profile create/delete
    iam = s.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
             "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    created: dict[str, bool] = {}

    def iam_role() -> str:
        iam.create_role(RoleName=TAG, AssumeRolePolicyDocument=json.dumps(trust))
        created["role"] = True
        iam.put_role_policy(RoleName=TAG, PolicyName="probe", PolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "s3:GetObject",
                                                    "Resource": f"arn:aws:s3:::{TAG}/*"}]}))
        return "create role + inline policy ok"
    record("iam:CreateRole+PutRolePolicy", iam_role)

    def iam_profile() -> str:
        iam.create_instance_profile(InstanceProfileName=TAG)
        created["profile"] = True
        if created.get("role"):
            iam.add_role_to_instance_profile(InstanceProfileName=TAG, RoleName=TAG)
            created["attached"] = True
        return "instance profile ok"
    record("iam:CreateInstanceProfile", iam_profile)
    # Some sandboxes ship a pre-made role instead
    record("iam:ListRoles (preexisting)", lambda: ", ".join(
        r["RoleName"] for r in iam.list_roles(MaxItems=100)["Roles"]
        if "aws-service-role" not in r["Path"])[:90])

    # CloudFront: list + a cheap write (OAC create/delete). Full distribution is tested in Phase 4.
    cf = s.client("cloudfront")
    record("cloudfront:ListDistributions", lambda: str(
        cf.list_distributions()["DistributionList"].get("Quantity", 0)) + " distributions")

    def cf_write() -> str:
        r = cf.create_origin_access_control(OriginAccessControlConfig={
            "Name": TAG, "SigningProtocol": "sigv4", "SigningBehavior": "always",
            "OriginAccessControlOriginType": "s3"})
        cf.delete_origin_access_control(Id=r["OriginAccessControl"]["Id"], IfMatch=r["ETag"])
        return "write API ok (OAC create/delete)"
    record("cloudfront:write (OAC)", cf_write)

    # Fallback paths
    record("lambda:ListFunctions", lambda: f"{len(s.client('lambda').list_functions()['Functions'])} fns")
    record("ecr:DescribeRepositories", lambda: f"{len(s.client('ecr').describe_repositories()['repositories'])} repos")
    record("apigatewayv2:GetApis", lambda: f"{len(s.client('apigatewayv2').get_apis()['Items'])} apis")
    record("elasticbeanstalk:DescribeApplications", lambda: f"{len(s.client('elasticbeanstalk').describe_applications()['Applications'])} apps")

    # Region guard: confirm another region is denied (informational)
    record("region guard (ec2 us-west-2 describe)", lambda: str(len(
        boto3.Session(region_name="us-west-2").client("ec2").describe_vpcs()["Vpcs"])) + " vpcs (NOT blocked)")

    # Outbound internet for model weights
    for url in ("https://github.com", "https://raw.githubusercontent.com", "https://huggingface.co",
                "https://pypi.org/simple/onnxruntime/"):
        def fetch(u: str = url) -> str:
            ctx = ssl.create_default_context(cafile=certifi.where())  # python.org builds lack system CAs
            with urllib.request.urlopen(u, timeout=10, context=ctx) as r:
                return f"HTTP {r.status}"
        record(f"net {url}", fetch)

    # Cleanup
    try:
        if created.get("attached"):
            iam.remove_role_from_instance_profile(InstanceProfileName=TAG, RoleName=TAG)
        if created.get("profile"):
            iam.delete_instance_profile(InstanceProfileName=TAG)
        if created.get("role"):
            iam.delete_role_policy(RoleName=TAG, PolicyName="probe")
            iam.delete_role(RoleName=TAG)
    except ClientError as e:
        print(f"WARNING cleanup failed for {TAG}: {e}")

    w = max(len(r[0]) for r in results)
    print(f"\n{'capability'.ljust(w)}  status   detail")
    for name, status, detail in results:
        print(f"{name.ljust(w)}  {status:<7}  {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
