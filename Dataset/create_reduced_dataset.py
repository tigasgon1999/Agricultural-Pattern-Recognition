import os
from PIL import Image
import glob

TRAIN_SAMPLES = 50
VAL_SAMPLES = 25
TEST_SAMPLES = 25
data_new_dir = os.path.join(os.getcwd(), 'Reduced_dataset/')
if not os.path.exists(data_new_dir):
	print("Creating folders in ", data_new_dir)
	os.makedirs(data_new_dir)
	os.makedirs(os.path.join(data_new_dir, 'train'))
	os.makedirs(os.path.join(data_new_dir, 'val'))
	os.makedirs(os.path.join(data_new_dir, 'test'))

def read_write_image(path, image_files, dir = 'train'):
	sub_dir = os.path.split(path)[1]
	if sub_dir in ['rgb', 'nir']:
		output_dir = os.path.join(data_new_dir, dir, 'images', sub_dir)
	elif sub_dir in ['cloud_shadow',  'double_plant', 'planter_skip', 'standing_water', 'waterway','weed_cluster']:
		output_dir = os.path.join(data_new_dir, dir, 'labels', sub_dir)
	else:
		output_dir = os.path.join(data_new_dir, dir, sub_dir)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for f in image_files:
		if sub_dir in ['rgb', 'nir']:
			pre, ext = os.path.splitext(f)
			f = pre + '.jpg'
		in_file_name = os.path.join(path, f)
		out_file_name = os.path.join(output_dir, f)
		with Image.open(in_file_name) as current_image:
			current_image.save(out_file_name)

	print("Created data in", output_dir)




directories = ['boundaries', 'images/rgb', 'images/nir', 'labels/cloud_shadow', 
'labels/double_plant', 'labels/planter_skip', 'labels/standing_water', 
'labels/waterway','labels/weed_cluster', 'masks']

training_paths = [os.path.join(os.getcwd(), 'Agriculture-Vision', 'train', sub_dir) for sub_dir in directories]
training_image_files = os.listdir(training_paths[0])[0:TRAIN_SAMPLES]

val_paths = [os.path.join(os.getcwd(), 'Agriculture-Vision', 'val', sub_dir) for sub_dir in directories]
val_image_files = os.listdir(val_paths[0])[0:VAL_SAMPLES]

test_dirs = ['boundaries', 'images/rgb', 'images/nir', 'masks']
test_paths = [os.path.join(os.getcwd(), 'Agriculture-Vision', 'test', sub_dir) for sub_dir in test_dirs]
test_image_files = os.listdir(test_paths[0])[0:TEST_SAMPLES]

for path in training_paths:
	read_write_image(path, training_image_files, dir = 'train')
for path in val_paths:
	read_write_image(path, val_image_files, dir = 'val')
for path in test_paths:
	read_write_image(path, test_image_files, dir = 'test')