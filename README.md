# Semantic Segmentation Project

The goal of this project is to perform semantic segmentation of road images using a Fully Convolution Network based on the VGG16 architecture. The goal is to segment the images to distinguish between road and not-road. The VGG16 network is modified by replacing the fully connected part with a semantic decoder consisting of a combination of 1x1 convolutions, transposed convolutions for upsampling and skipped connections.

Seed code for the project is provided by Udacity in this GitHub repository:

https://github.com/udacity/CarND-Semantic-Segmentation

[//]: # (Image References)
[image1]: ./um_000005.png
[image2]: ./um_000014.png
[image3]: ./um_000060.png
[image4]: ./um_000093.png
[image5]: ./umm_000000.png
[image6]: ./umm_000075.png
[image7]: ./umm_000087.png
[image8]: ./uu_000065.png
[image9]: ./uu_000081.png
[image10]: ./uu_000099.png

## Network Architecture

My network is based on a pre-trained VGG16 network, and the first part of the software loads this model and extracts the layers I want to reuse, namely layer 3, 4 and 7. With these layers a fully convolutional network is created by implementing the following:

* Each of the 3 vgg layers are 1x1 convoluted, setting the depth to 2 - the number of classes we want to distinguish.
* The 1x1 convoluted output of layer 7 is upsampled by a factor 2 via a transposed convolution, and element-wise added to the 1x1 convoluted output of layer 4.
* This output is again upsampled by a factor 2 via a transposed convolution, and element-wise added to the 1x1 convoluted output of layer 3.
* This final output is upsampled with a factor 8 again to create a final network result of the same size as the original input image.

The two times upsampling by 2 ensure that the tensor sizes are the same so the element-wise addition works correctly. These additions implement the required skipping of connections.

As proposed in the project Q&A I use a kernel l2-regularizer for each convolution with the parameter 1e-3.

For training I use a cross-entropy loss function and an Adam-optimizer, as proposed in the project Q&A. In order to ensure that regularization is being done I collect the loss and add it to the cross entropy loss like this:

l2_loss = tf.losses.get_regularization_loss()

loss = tf.reduce_mean(cross_entropy_loss + (0.01 * l2_loss))

The multiplying factor 0.01 is essentially a hyper parameter. I found the value 0.01 by searching the internet. I trained my network without using this factor, and although this also produced usable results, using 0.01 clearly is much better.

I also trained the network without adding the regularization loss. Also here I got usable results, but by inspection it looks like that adding regularization loss produces a better working network, so I decided to stick with it for my submission, despite that I am not 100% sure about the implementation.

I used the following general hyper parameters:

* learning rate 0.0009
* keep probability 0.5

I trained my network on the project workspace a varying number of times, and ended up using the results after 50 epochs and batch-size of 5. This training took around 1 hour on the workspace GPU.

Please notice that the path to the data and vgg folders in my submission (main.py line 181 and 182) are set to work on the workspace GPU. For local testing you need to adapt them.

## Results

Using the above setup I saw a decreasing trend in the loss after each epoch until around 40 training epoch. Here after the loss did not improve further, so I stopped training after 50 epochs. It might be possible to achieve a smaller loss by using a smaller learning rate.

Examples of observed loss results:

* Epoch 1, loss = 0.627
* Epoch 2, loss = 0.558
* Epoch 3, loss = 0.449
* Epoch 10, loss = 0.192
* Epoch 20, loss = 0.123
* Epoch 30, loss = 0.100
* Epoch 40, loss = 0.044
* Epoch 50, loss = 0.037

Observing the test images produced by the helper functions, it can be seen that the network performs pretty well in most cases. For example:

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image5]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

However there are also examples of images where the network does not perform that well, for example:

![alt text][image4]

![alt text][image6]

The bad performance appear to be on images with big differences between the darkest and the brightest parts. One way of getting better results would surely have been to augment the training image set by varying the brightness of the images.

All in all I believe this performance is good enough to pass the project.

# Udacity Project Instructions

The following are the original Udacity project instructions.

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well.

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
