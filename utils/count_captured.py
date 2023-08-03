from glob import glob

PHASE=3

print('PROB1: ', len(glob(f'data/captured_data/phase-{PHASE}/prob-1/*.parquet')))
print('PROB2: ', len(glob(f'data/captured_data/phase-{PHASE}/prob-2/*.parquet')))