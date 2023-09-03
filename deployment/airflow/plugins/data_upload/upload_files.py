from minio import Minio
from minio.error import S3Error
import glob
import os
import argparse
import os.path

client = Minio(
        "127.0.0.1:9009",
        access_key="gGL0SPj6CNLosSNR7nfM",
        secret_key="qxViOu0AW2z6kaa7VtrOmXENOMOlQIwMoJPfTK2D",
        secure=False
    )

def upload_files(minio_path, client, local_path_up, bucket_name):
    found = client.bucket_exists(bucket_name)
    if not found:
        print("Không có dữ liệu")
    else:
        print("Có dữ liệu")
    curr_path = os.getcwd()
    local_path = os.path.join(curr_path, local_path_up)
    print(local_path)
    assert os.path.isdir(local_path)

    for local_file in glob.glob(local_path + '/**'):
        local_file = local_file.replace(os.sep, "/") 
        if not os.path.isfile(local_file):
            upload_files (
                minio_path + "/" + os.path.basename(local_file), client, local_file, bucket_name)
        else:
            remote_path = os.path.join(minio_path, local_file[1 + len(local_path):])
            remote_path = remote_path.replace(os.sep, "/")
            client.fput_object(bucket_name, remote_path, local_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path_up", type=str, default='data/curl/')
    parser.add_argument("--bucket_name", type=str, default='mock-test-data')

    args = parser.parse_args()
    try:
        upload_files( "", client, local_path_up=args.local_path_up, bucket_name=args.bucket_name)
    except S3Error as exc:
        print("error occurred.", exc)