import shutil
import os
import numpy as np
import tqdm

# WARNING RUN ONLY ONCE!
if not os.path.exists('data/test'):
    os.mkdir('data/test')

train_path = 'data/train/'
test_path = 'data/test/'

train_files = os.listdir(train_path)

test_files = np.unique(np.random.choice(train_files, size=int(0.1 * len(train_files))))

for filename in tqdm.tqdm(test_files, desc=f"Moving test files into {test_path}"):
    shutil.move(train_path + filename, test_path + filename)
