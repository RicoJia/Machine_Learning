#!/usr/bin/env python3

import os
import pickle
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload_file(service, file_path, mime_type=None, parent_id=None):
    """Uploads a file to Google Drive."""
    file_name = os.path.basename(file_path)
    file_metadata = {"name": file_name}
    if parent_id:
        file_metadata["parents"] = [parent_id]
    media = MediaFileUpload(
        file_path, mimetype=mime_type if mime_type else "application/octet-stream"
    )
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    print(f'File ID: {file.get("id")}')


def authenticate():
    """Authenticates the user and returns the Drive service."""
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds = None
    # The file token.pickle stores the user's access and refresh tokens.
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                print(
                    "credentials.json not found. Please ensure you have downloaded it from Google Cloud Console."
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    service = build("drive", "v3", credentials=creds)
    return service


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python upload_to_drive.py <file_path> [mime_type] [parent_folder_id]"
        )
        sys.exit(1)

    file_path = sys.argv[1]
    mime_type = sys.argv[2] if len(sys.argv) > 2 else None
    parent_id = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    service = authenticate()
    upload_file(service, file_path, mime_type, parent_id)


if __name__ == "__main__":
    main()
