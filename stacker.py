import os
from shutil import copyfile

counter = 0

raw_dataset_folder = "raw_dataset/"
processed_folder = "processed_dataset/"

if not os.path.isdir(processed_folder):
    os.makedirs(processed_folder)
if not os.path.isdir(processed_folder + 'positive/'):
    os.makedirs(processed_folder + 'positive/')
if not os.path.isdir(processed_folder + 'negative/'):
    os.makedirs(processed_folder + 'negative/')

for i in range(1, 4):
    src_folder_name = raw_dataset_folder + '{}/ped_examples/'.format(i)
    for image_file in os.listdir(src_folder_name):
        generated_name = 'pos_{:05}.pgm'.format(counter)
        print(image_file, generated_name, sep=' => ')
        copyfile(src_folder_name + image_file,
                 processed_folder + 'positive/' + generated_name)
        counter += 1


for i in range(1, 3):
    src_folder_name = raw_dataset_folder + '/T{}/ped_examples/'.format(i)
    for image_file in os.listdir(src_folder_name):
        generated_name = 'pos_{:05}.pgm'.format(counter)
        print(image_file, generated_name, sep=' => ')
        copyfile(src_folder_name + image_file,
                 processed_folder + 'positive/' + generated_name)
        counter += 1

counter = 0

for i in range(1, 4):
    src_folder_name = raw_dataset_folder + '{}/non-ped_examples/'.format(i)
    for image_file in os.listdir(src_folder_name):
        generated_name = 'neg_{:05}.pgm'.format(counter)
        print(image_file, generated_name, sep=' => ')
        copyfile(src_folder_name + image_file,
                 processed_folder + 'negative/' + generated_name)
        counter += 1



for i in range(1, 3):
    src_folder_name = raw_dataset_folder + '/T{}/non-ped_examples/'.format(i)
    for image_file in os.listdir(src_folder_name):
        generated_name = 'neg_{:05}.pgm'.format(counter)
        print(image_file, generated_name, sep=' => ')
        copyfile(src_folder_name + image_file,
                 processed_folder + 'negative/' + generated_name)
        counter += 1




        
