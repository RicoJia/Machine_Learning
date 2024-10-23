#!/usr/bin/env python3

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import importlib.util
import shutil
import os
import argparse

from google_drive_upload import authenticate, upload_file

HELP_STR = """
This script uploads a file or one directory, assuming aws client has been set up properly
"""
THIS_PACKAGE = "ricomodels"


def find_this_pkg_path():
    spec = importlib.util.find_spec(THIS_PACKAGE)
    if spec is None:
        raise FileExistsError(f"Package {THIS_PACKAGE} is not installed yet. Please install using pip")
    else:
        return os.path.dirname(spec.origin)


def get_or_prompt_file_name():
    parser = argparse.ArgumentParser(description=HELP_STR)
    parser.add_argument('--file', '-f', type=str, default='', help='Filename to upload')
    parser.add_argument('--directory', '-r', type=str, default='', help='Filename to upload')
    args = parser.parse_args()

    while args.file == "" and args.directory == "":
        mode = input("Are you uploading 1. directory, 2. a file?\n")
        if mode == "1":
            args.directory = input("What's the name of directory to upload?\n")
            break
        elif mode == "2":
            args.file = input("What's the name of file to upload?\n")
            break
        else:
            print(f"Invalid input {mode}, please enter a valid mode.")

    return args


def find_file_path(pkg_path: str, args):
    for root, dirs, files in os.walk(pkg_path):
        if args.directory and args.directory in dirs:
            return os.path.abspath(os.path.join(root, args.directory))
        if args.file and args.file in files:
            return os.path.abspath(os.path.join(root, args.file))
    return ""


def upload_file_to_s3(file_name, bucket, object_key=None):
    """
    Uploads a file to an S3 bucket.

    :param file_name: Path to the file to upload.
    :param bucket: Name of the target S3 bucket.
    :param object_key: S3 object name. If not specified, file_name is used.
    :return: True if file was uploaded, else False.
    """
    # If S3 object_key was not specified, use file_name
    print("Uploading ... ")
    print(object_key)

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket, object_key)
        print(f"File '{file_name}' uploaded to bucket '{bucket}' as '{object_key}'.")
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
        args = get_or_prompt_file_name()
        file_path = find_file_path(pkg_path, args)

    print(file_path)
    if args.directory:
        file_path = shutil.make_archive(file_path, 'zip', file_path)
    object_key = os.path.relpath(file_path, pkg_path)

    # upload_file_to_s3(file_path, bucket_name, object_key)

    service = authenticate()
    upload_file(service, file_path, None, None)
    if args.directory:
        print(f"removed zip {file_path}")
        os.remove(file_path)
