import argparse
from minio import Minio
import os
import pathlib
from tqdm import tqdm
import sys


def get_data(minio_server: str, src_path=None, dst_path='.', verbose=0, include_pattern=None, exclude_pattern=None, tag:tuple=None, return_paths=False):
    minio_server = minio_server.replace('http://', '') if minio_server.startswith('http://') else minio_server
    print(minio_server)
    client = Minio(
            minio_server,
            access_key="admin",
            secret_key="password",
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
        print(f"Include {include_pattern}")
        # print(f"Only download files contain {include_pattern} in path.")
        objects = [obj for obj in objects if include_pattern in obj.object_name]
    
    if not exclude_pattern is None:
        print(f"Exclude {exclude_pattern}")
        # print(f"Only download files contain {include_pattern} in path.")
        objects = [obj for obj in objects if exclude_pattern not in obj.object_name]
    
    if not tag is None:
        print(f"Tag: {tag}")
        k, v = tag
        
        def filter_by_tag(object_name, k, v):
            tag = client.get_object_tags('data', object_name)
            if not tag is None:
                if tag[k] == v:
                    return True
            return False
                
        print("Filtering by tag ...")
        objects = [obj for obj in tqdm(objects) if filter_by_tag(obj.object_name, k, v)]
    
    if return_paths:
        return [obj.object_name for obj in objects]
    
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
    parser.add_argument("--src_path", type=str, default=None, help="Download dữ liệu trong đường dẫn này")
    parser.add_argument("--dst_path", type=str, default='.', help="Ghi dữ liệu ra đường dẫn này")
    # parser.add_argument("--is_path_include_bucket", type=lambda x: (str(x).lower() == "true"), default=True, 
    #                     help='nếu True, vị trí đầu tiên trong đường dẫn là bucket')
    parser.add_argument("--include_pattern", type=str, default=None,
                        help="chỉ download những folder có pattern này")
    parser.add_argument("--exclude_pattern", type=str, default=None,
                        help="không download những folder có pattern này")
    parser.add_argument("--verbose", type=int, default=0, 
                        help='0: hide error')
    parser.add_argument('--return_paths', type=lambda x: (str(x).lower() == "true"), default=False)
    
    args = parser.parse_args()
    
    # print(args)
    
    get_data(args.minio_server, args.src_path, args.dst_path, args.verbose, args.include_pattern, args.exclude_pattern)