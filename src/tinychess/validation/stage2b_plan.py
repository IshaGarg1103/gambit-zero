"""Standard-library-only Stage 2B bounds, also readable before CUDA setup."""
EXPECTED_COUNTS={'25m':25622092,'100m':100031308}
DATASET_REVISION='b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c'


def planned_cases():
    return [dict(profile=p,compute=c,batch=b) for c,b in
            [('bfloat16',4),('float32',4),('bfloat16',32)] for p in ['25m','100m']]
