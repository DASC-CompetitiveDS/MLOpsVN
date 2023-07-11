from glob import glob

PHASE=1

print('PROB1: ', len(glob(f'data/captured_data/phase-{PHASE}/prob-1/*')))
print('PROB2: ', len(glob(f'data/captured_data/phase-{PHASE}/prob-2/*')))