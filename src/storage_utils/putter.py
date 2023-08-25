import argparse
from minio import Minio
import os
import pathlib
from tqdm import tqdm


def put_data(path, is_include_parent_path):
    client = Minio(
            "localhost:9009",
            access_key="QhHnuvs0GQeZZeUvGWph",
            secret_key="LENlErsDf0JV2HbYGeQiDuFgH4kbjR9ipBvLO6ky",
            secure=False
        )

    path_ = pathlib.Path(path)
    if os.path.isdir(path_):
        src_files = [file for file in path_.rglob("*") if os.path.isfile(file)]
        dst_files = src_files if is_include_parent_path else [str(file)[len(path)+1:] for file in src_files]
        for src_file, dst_file in tqdm(zip(src_files, dst_files), total=len(src_files)):
            client.fput_object('data', dst_file, src_file)
    else:
        src_file = path_
        dst_file = str(path_)[len('data')+1:] if str(path_).startswith('data') else str(path_)
        client.fput_object('data', dst_file, src_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data')
    parser.add_argument("--is_include_parent_path", type=lambda x: (str(x).lower() == "true"), default=False, 
                        help='nếu True, thêm đầy đủ đường dẫn; nếu False thì chỉ đẩy thư mục con')
    
    args = parser.parse_args()
    
    put_data(args.path, args.is_include_parent_path)