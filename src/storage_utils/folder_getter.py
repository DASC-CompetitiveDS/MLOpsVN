import argparse
from minio import Minio
import os
import pathlib
from tqdm import tqdm


def get_data(minio_server: str, src_path=None, dst_path='data', is_path_include_bucket=True, verbose=0, include_pattern=None):
    minio_server = minio_server.replace('http://', '') if minio_server.startswith('http://') else minio_server
    
    client = Minio(
            minio_server,
            access_key="QhHnuvs0GQeZZeUvGWph",
            secret_key="LENlErsDf0JV2HbYGeQiDuFgH4kbjR9ipBvLO6ky",
            secure=False
        )
        
    # path_ = pathlib.Path(path)
    # if os.path.isdir(path_):
        # src_files = [file for file in path_.rglob("*") if os.path.isfile(file)]
        # dst_files = src_files if include_parent_path else [str(file)[len(path)+1:] for file in src_files]
        
        # for src_file, dst_file in tqdm(zip(src_files, dst_files), total=len(src_files)):
        #     client.fput_object('data', dst_file, src_file)
    # else:
        # src_file = path_
        # dst_file = str(path_)[len('data')+1:] if str(path_).startswith('data') else str(path_)
        # client.fput_object('data', dst_file, src_file)
    
    objects = list(client.list_objects("data", recursive=True, prefix=src_path))
    if not include_pattern is None:
        objects = [obj for obj in objects if include_pattern in obj.object_name]
        
    print("Downloading data ...")
    for i, obj in tqdm(enumerate(objects), total=len(objects)):
        try:
            client.fget_object("data", obj.object_name, os.path.join(dst_path, obj.object_name))
        except Exception as e:
            if verbose == 0:
                pass
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio_server", type=str, default='localhost:9009')
    parser.add_argument("--src_path", type=str, default=None)
    parser.add_argument("--dst_path", type=str, default='data')
    parser.add_argument("--is_path_include_bucket", type=lambda x: (str(x).lower() == "true"), default=True, 
                        help='nếu True, vị trí đầu tiên trong đường dẫn là bucket')
    parser.add_argument("--include_pattern", type=str, default=None,
                        help="chỉ download những folder có pattern này")
    parser.add_argument("--verbose", type=int, default=0, 
                        help='0: hide error')
    
    args = parser.parse_args()
    
    # print(args)
    
    get_data(args.minio_server, args.src_path, args.dst_path, args.is_path_include_bucket, args.verbose, args.include_pattern)