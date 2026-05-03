import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your specific details
ACCESS_KEY = os.getenv('BUCKET_ACC_KEY')
SECRET_KEY = os.getenv('BUCKET_SECRET_KEY')
URL = os.getenv('BUCKET_URL')
BUCKET_NAME = os.getenv('BUCKET_NAME')
REGION = os.getenv('REGION')
LOCAL_DIRECTORY = './EIC' # The folder on your computer

def upload_folder():
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=REGION,
        endpoint_url=URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    for root, dirs, files in os.walk(LOCAL_DIRECTORY):
        for file in files:
            local_path = os.path.join(root, file)
            
            # This creates the 'folder' structure inside the bucket
            relative_path = os.path.relpath(local_path, LOCAL_DIRECTORY)
            space_path = relative_path.replace("\\", "/") 

            print(f"Uploading {file}...")
            client.upload_file(
                local_path, 
                BUCKET_NAME, 
                space_path,
                ExtraArgs={'ACL': 'public-read'} # This makes the individual FILE public
            )

    print(f"Done! View your files at: https://{BUCKET_NAME}.{REGION}.digitaloceanspaces.com")

if __name__ == "__main__":
    upload_folder()
