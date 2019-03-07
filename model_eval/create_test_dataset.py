import os
from shutil import copyfile

path = './datasets/color/'

### To create a test folder of real images to run inception score test using it. 

def create_test_data():
	i = 0
	for apath in os.listdir(path):
		if not os.path.exists('test-data/'+ apath):
			os.mkdir('test-data/'+ apath)
		
		for image in (os.listdir(path+apath)):
			if (image and (i <= 600)): # 600 images from each folder 
				copyfile(path+apath +'/'+image, 'test-data/'+apath+'/'+image)
				i += 1
		i = 0

if __name__ == '__main__':
	create_test_data()
