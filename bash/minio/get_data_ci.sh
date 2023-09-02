python3 src/storage_utils/folder_getter.py --minio_server $1 --include_pattern $2 --exclude_pattern $3 --dst_path $4
# python3 src/storage_utils/get_captured_data.py --minio_server $1 --src_path data/captured_data/phase-3 --captured_version $5 --drop_exist False
python3 src/storage_utils/file_getter.py --minio_server $1 --src_path data/captured_data/phase-3/prob-1/123.parquet
python3 src/storage_utils/file_getter.py --minio_server $1 --src_path data/captured_data/phase-3/prob-2/123.parquet