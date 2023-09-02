import argparse
import os
from glob import glob
from tqdm import tqdm
import pathlib

def drop_exist(path):
    path = pathlib.Path(path)
    files = list(path.rglob('*.parquet'))
    if len(files) == 0:
        return
    print("Remove exist captured data ...")
    for file in tqdm(files, total=len(files)):
        os.remove(file)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio_server", type=str, default='localhost:9009')
    parser.add_argument("--src_path", type=str, default='data/captured_data/phase-3/prob-2')
    parser.add_argument("--dst_path", type=str, default='.')
    parser.add_argument("--include_pattern", type=str, default=None,
                        help="chỉ download những folder có pattern này")
    parser.add_argument("--captured_version", type=str, default=0)
    parser.add_argument("--drop_exist", type=lambda x: (str(x).lower() == "true"), default=True, 
                        help='drop existing captured data')
    parser.add_argument("--verbose", type=int, default=0, 
                        help='0: hide error')
    
    args = parser.parse_args()
    
    from folder_getter import get_data
    
    if args.drop_exist:
        drop_exist(args.src_path)
        
    get_data(args.minio_server, args.src_path, args.dst_path, args.verbose, args.include_pattern, tag=('captured_version', args.captured_version))