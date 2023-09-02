import argparse
from minio import Minio
import os
import pathlib
from tqdm import tqdm


def put_data(minio_server, path):
    minio_server = minio_server.replace('http://', '') if minio_server.startswith('http://') else minio_server
    print(minio_server)

    client = Minio(
            minio_server,
            access_key="admin",
            secret_key="password",
            secure=False
        )

    path_ = pathlib.Path(path)
    if os.path.isdir(path_):
        dst_files = [str(file) for file in path_.rglob("*") if os.path.isfile(file)]
        # dst_files = src_files if is_include_parent_path else [file[len(path)+1:] for file in src_files]
        # print(dst_files)
        # for src_file, dst_file in tqdm(zip(src_files, dst_files), total=len(dst_files)):
        for dst_file in tqdm(dst_files, total=len(dst_files)):
            client.fput_object('data', dst_file, dst_file)
    else:
        dst_file = str(path_)
        # dst_file = str(path_)[len('data')+1:] if str(path_).startswith('data') else str(path_)
        client.fput_object('data', dst_file, dst_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio_server", type=str, default='localhost:9009')
    parser.add_argument("--path", type=str, default='data')
    # parser.add_argument("--is_include_parent_path", type=lambda x: (str(x).lower() == "true"), default=False, 
    #                     help='nếu True, thêm đầy đủ đường dẫn; nếu False thì chỉ đẩy thư mục con')
    
    args = parser.parse_args()
    
    put_data(args.minio_server, args.path)