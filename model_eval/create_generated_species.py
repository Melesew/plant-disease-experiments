#This function merge images of the same parent species(
# eg. all folders whose name starts with Apple will be placed in a folder named Apple)

import os

from shutil import copyfile

#create generated_species folder first

path = '../../generated_data/'
path2 = '../../generated_species/'

paths = []

for file in os.listdir(path):
	paths.append(file.split('_')[0])

for i in range(len(paths)):
	path_name = paths[i].split(',')[0]
	
	if not os.path.exists(path2 + path_name):
		os.mkdir(path2+path_name)

i = 0
for path_name in os.listdir(path):
	if not path_name.startswith('.'):

		if path_name.split('_')[0] in paths:
			new_path_name = path2 + (path_name.split('_')[0]).split(',')[0]
		
		x = len(os.listdir(path + path_name))
		
		for file in os.listdir(path + path_name):

			filename_without_ext = file.split('.')[0]
			extension = file.split('.')[1]
			    
			new_file_name = str(i)  
			new_file_name_with_ext = new_file_name+'.'+extension
			
			copyfile(path+path_name+'/'+file, new_path_name+'/'+ new_file_name_with_ext)
			
			i += 1
		


