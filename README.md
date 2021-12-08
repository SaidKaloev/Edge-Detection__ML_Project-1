# Edge-Detection__ML_Project-1
Have fun looking at the project. If you got any questions or found a mistake, feel free to contact me.


#Author: Said Kaloev
#Exercise 5: ML Project
#Contact: said.ischchan62@gmail.com

###Description###

This project is part of the ML Challenge of Programming in Python2 - 2021.
It is trained to predict the random unknown borders of the input_images, which have a 90x90 shape
There are a total of around 40k images, which were converted to grayscale and reshaped to a 90x90shape


Due to privacy reasons the dataset could not be uploaded

###Useage###

If the images are already in a pickle file, you just need to run the main.py file, which will then start the training and evaluation process.
If not, then you first need to open the ReadAndCreate.py file. Only then can you start the main.py file.

###Structure###

-architectures.py

This file contains the model's structure and consists of, in this case, multiple neural layers. In between we can find some RELU-Activation-Layers
	

-ReadAndCreate.py

	
This file contain the functions resize_transformer, data_preparator and Image_Reader

		1.resize_transformer: It gets an image of arbitrary shape and reshapes it to a 90x90 shape, which is then returned
		
		2.data_preparator: Gets the images, calls the resize_transformer and Image_Reader, and stores the created array into a pklz file (=dataset)
		
		3.Image_Reader: Is the ex4 solution, which was created by the Python2 Team. As output we get 3 arrays
		

-datasets.py


Consists of only the class ImageData, which gets the arrays stored in the created pickels-dataset

-main.py

This file consists of 2 functions and a training part

		1.separate_sets: This only gets the data from datasets.py and splits it into trainig-,test- and validationsets
		
		2.evaluate_model: Function for evaluation of a model `model` on the data in `dataloader` on device `device. It returns the loss of the model
		
		3.Training_part: Consists of data_loaders and multiple for and while-loops. At the beginning of the training we can change the hyperparameters, to change the CNN-Architecture.
		

-ReadMe.md

	A readme file containing information about this project & author
