import nltk
from nltk.corpus import brown
import boto3
import pickle
import os

def push_to_s3(s3_file = "corpus.pkl"):
    s3 = boto3.client("s3")
    bucket_name = os.environ["S3_BUCKET"]
    s3_data = list(brown.words())
    with open(s3_file, 'wb') as f:
        pickle.dump(s3_data, f)

    try:
        s3.upload_file(s3_file, bucket_name, s3_file)
        print("Uploaded "+ s3_file + " to " + s3_file + " in S3 succefully!")
    except FileNotFoundError:
        print("The file " + s3_file + " was not found.")

if __name__ == "__main__":
    push_to_s3()