import os

negatives_path = "archive/neg_images"
with open('negatives.txt', 'w') as f:
    for filename in os.listdir(negatives_path):
        f.write(f'{negatives_path}/{filename}')
