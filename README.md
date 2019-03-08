# Plant Disease Detection using Deep Learning

[Using Deep Learning for Image-Based Plant Disease Detection](https://arxiv.org/pdf/1604.03169.pdf) 

<<<<<<< HEAD
**Resources:**
- [Data set](https://github.com/spMohanty/PlantVillage-Dataset)

- [Models](https://gitlab.com/Israel777/Plant_Disease_Detection_models)

=======


**Resources:**
- [Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
>>>>>>> fc79197a34acc3f03f6562e510255de28461b49d

- Additional dataset will be relesed 

<<<<<<< HEAD
**Objective**

- Train and Evaluate different DNN Models for plant deasise detection Problem

- To tackle the problem of scarce real-life representative data, experiment with different generative networks and generate more plant leaf image data

- Implement segmentation pipeline to avoid missclassification due to unwanted input 
=======
- [Models](https://gitlab.com/Israel777/Plant_Disease_Detection_models)



**Objective**

- Train and Evaluate different DNN Models for plant disease detection problem

- To tackle the problem of scarce real-life representative data, experiment with different generative networks and generate more plant leaf image data

- Implement segmentation pipeline to avoid misclassification due to unwanted input 
>>>>>>> fc79197a34acc3f03f6562e510255de28461b49d



**Approches for Solving the papers realtime Detection Problem**

phase 1 : [implement the paper](https://github.com/singnet/plant-disease-experiments/tree/master/Plant_Disease_Detection_Benchmark_models) 

phase 2 : do analysis on the paper and identify the type of data problem 

<<<<<<< HEAD
phase 3 : [experement and if possible generate Apprprate data
		  using the data train the model again](https://github.com/singnet/plant_disease_experements/tree/master/Plant_Disease_Detection_gan_experimants)
=======
phase 3 : [experiment and if possible generate appropriate data
		  using the data to train the model again](https://github.com/singnet/plant_disease_experements/tree/master/Plant_Disease_Detection_gan_experimants)
>>>>>>> fc79197a34acc3f03f6562e510255de28461b49d


# Project Structure

**Plant_Disease_Detection_Benchmark_models**

- Train and test different prediction models to get a baseline accuracy to compare to and see progress

**Plant_Disease_Detection_gan_experiments**

- experiment with different generative networks to see their generative capability and if the output can be used to train more robust models

**leaf-image-segmentation-segnet**

<<<<<<< HEAD
- segmentation pipline using VGGSegNet Architecture
=======
- segmentation pipeline using VGGSegNet Architecture
>>>>>>> fc79197a34acc3f03f6562e510255de28461b49d

**leaf-image-segmentation**

- histogram based segmentation Pipline 





# Usage

<<<<<<< HEAD
	Python main.py --image "abc.jpg"

- This will segment the image and predict the output class based on that . segmented image will be saved as the file name with "_masked" prefix.

-  You can cheack the segmentation accuracy from saved image
=======
	python main.py IMAGE_FILE [--segment] [--species SPECIES_TYPE] [--model PREDICTION_MODEL]

	Arguments:
		IMAGE_FILE    Path of the image file
		--segment     If specified perform segmentation on the image before prediction
		--species     If the plant species on the image is priorly known. One of the following species:  Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, Sugercane, Tomato
		--model       What models do you want to use, vgg or inceptionv3



# Examples

	# you can remove a part of arguments except image path
	
	>>  python main.py 'test/a.jpg' --segment --species 'apple' --model 'inceptionv3'
	   

- Before using that make sure you download the weights from   [here for Inception_V3](https://drive.google.com/file/d/1PZ0SUyGbcKJidNcSfwKsnhR23O2PBl78/view?usp=sharing) and  [here for VGG Models](https://drive.google.com/file/d/1AufdWYl-TfeicAmaweq6Gd8q3--vuBfA/view?usp=sharing)  and extract all and put it in `Plant_Disease_Detection_Benchmark_models/Models/` folder. 
		
- This will segment the image and predict the output class based on that. Segmented image will be saved as the file name with "_marked" suffix before the file extension.


- The images are trained with segmented network and lower performance on unsegmented dataset is expected.

- You can check the segmentation accuracy from saved image.


- Fill [this form](https://goo.gl/forms/ceQNkEimLL8NN1sF2) for bulk model access grants and future update notification.
>>>>>>> fc79197a34acc3f03f6562e510255de28461b49d
