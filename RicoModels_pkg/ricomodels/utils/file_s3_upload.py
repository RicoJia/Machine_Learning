#!/usr/bin/env python3

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import importlib.util
import os
import argparse

THIS_PACKAGE = "ricomodels"


def find_this_pkg_path():
    spec = importlib.util.find_spec(THIS_PACKAGE)
    if spec is None:
        raise FileExistsError(f"Package {THIS_PACKAGE} is not installed yet. Please install using pip")
    else:
        return os.path.dirname(spec.origin)


def get_or_prompt_file_name():
    parser = argparse.ArgumentParser("Getting Model name")
    parser.add_argument('--model', '-m', type=str, default='', help='Filename to upload')
    args = parser.parse_args()
    if args.model == "":
        args.model = input("What's the name of model to upload?\n")
    return args.model


def find_file_path(pkg_path: str, file_name: str):
    for root, dirs, files in os.walk(pkg_path):
        if file_name in files:
            return os.path.abspath(os.path.join(root, file_name))
    return ""


def upload_file_to_s3(file_name, bucket, object_name=None):
    """
    Uploads a file to an S3 bucket.

    :param file_name: Path to the file to upload.
    :param bucket: Name of the target S3 bucket.
    :param object_name: S3 object name. If not specified, file_name is used.
    :return: True if file was uploaded, else False.
    """
    # If S3 object_name was not specified, use file_name
    print("Uploading ... ")
    if object_name is None:
        object_name = file_name

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"File '{file_name}' uploaded to bucket '{bucket}' as '{object_name}'.")
        return True
    except FileNotFoundError:
        print(f"The file '{file_name}' was not found.")
        return False
    except NoCredentialsError:
        print("AWS credentials not available.")
        return False
    except PartialCredentialsError:
        print("Incomplete AWS credentials provided.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    file_path = 'losses.py'  # Replace with your file path
    bucket_name = 'rico-machine-learning-weights'         # Replace with your bucket name

    file_path = ""
    while file_path == "":
        pkg_path = find_this_pkg_path()
        file_name = get_or_prompt_file_name()
        file_path = find_file_path(pkg_path, file_name)

    print(file_path)
    object_key = os.path.relpath(file_path, pkg_path)

    upload_file_to_s3(file_path, bucket_name, object_key)
