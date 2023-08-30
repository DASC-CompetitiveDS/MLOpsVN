import argparse
from minio import Minio
import os
import pathlib
from tqdm import tqdm
from pandas.util import hash_pandas_object
from io import BytesIO
from pathlib import Path

class ParquetPutter:
    def __init__(self, minio_server:str):
        minio_server = minio_server.replace('http://', '') if minio_server.startswith('http://') else minio_server

        self.client = Minio(
                minio_server,
                access_key="admin",
                secret_key="password",
                secure=False
            )
    
    def put_data(self, captured_data_dir:str, dataframe, data_id:str, num_parallel_uploads=3, tags=None):
        # paths = Path(captured_data_dir).parts
        # bucket_name = paths[0]
        # folder = os.sep.join(paths[1:])
        # captured_data_dir = captured_data_dir.replace(bucket_name, "")
        
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(dataframe).sum()
        object_name = os.path.join(captured_data_dir, f"{filename}.parquet")

        data = BytesIO()
        dataframe.to_parquet(data, index=False)
        data.seek(0)
                
        self.client.put_object('data', object_name, data, length=-1, part_size=10*1024*1024, 
                               num_parallel_uploads=3, tags=tags)
        
        # path_ = pathlib.Path(path)
        # if os.path.isdir(path_):
        #     src_files = [file for file in path_.rglob("*") if os.path.isfile(file)]
        #     dst_files = src_files if is_include_parent_path else [str(file)[len(path)+1:] for file in src_files]
        #     for src_file, dst_file in tqdm(zip(src_files, dst_files), total=len(src_files)):
        #         client.fput_object('data', dst_file, src_file)
        # else:
        #     src_file = path_
        #     dst_file = str(path_)[len('data')+1:] if str(path_).startswith('data') else str(path_)
        #     client.fput_object('data', dst_file, src_file)
    
    def _test(self):
        result = self.client.put_object(
            "data", "test-object", BytesIO(b"hello"), length=-1, part_size=10*1024*1024,
        )

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, default='data')
    # parser.add_argument("--is_include_parent_path", type=lambda x: (str(x).lower() == "true"), default=False, 
    #                     help='nếu True, thêm đầy đủ đường dẫn; nếu False thì chỉ đẩy thư mục con')
    
    # args = parser.parse_args()
    
    parquet_putter = ParquetPutter()
    parquet_putter._test()