---
layout: post
title:  "Supernovas"
date:   2018-02-07
description: Using images from a Chilean telescope we start to build a basic model for detecting supernovas using XGBoost. 
url: http://space2vec.com/blog/space-2-vec-supernova-detection/
image: post-2/xb-cc.jpg
---

#### Update #2: Building a model to detect Supernovas


_The goal of these blog posts is to document our research into applying modern machine learning techniques in astronomy. We'll write about what works, and what doesn't. We write software by day so these posts should hopefully give insight into HOW modern astronomy and machine learning works as we better understand them ourselves._


In our previous update, we explored different astronomy problems and their feasibility. With our limited knowledge in this space (pun totally intended) it made sense to try and pick an initial problem with metrics that were well defined. A large part of our curiosity with the space2vec project is asking whether modern techniques (think deep learning/neural nets) in machine learning can be useful in astronomy. So we looked for work done using classical methods that could be improved on, in hopes of having early comparisons to document.


**_Side Note_: These comparisons are crucial for people learning to build their first model, how else would you know how well you are doing?**


We searched [arxiv.org](https://arxiv.org/), a website researchers use to post their work. During our search we found a paper that discusses building more efficient ways of detecting supernova (an exploding star), using images from the Dark Energy Survey. Titled _[Automated Transient Identification in the Dark Energy Survey](https://arxiv.org/abs/1504.02936) (ATIDES),_ the paper outlines a model for detecting supernovas from images. It explains that the current methods heavily depend on the human eye for detection of transient astronomical events.



<figure>
	<img src="{{ '/assets/img/post-2/supernova-lifecycle.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Supernova lifecycle.</figcaption>
</figure>



_ATIDES_ laments the labour intensive work of looking at images to discern if they do indeed contain supernova. The problem it points to is that this manual work does not scale well as datasets get larger over the next decade. The paper improves quite well on this notion, using a [random forest](https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674), a classical machine learning algorithm, to classify each image as either interesting (likely to be a supernova), or not (non-objects or artifacts).


#### Overview of ATIDES Solution to Identifying Transients

ATIDES code and data: [http://portal.nersc.gov/project/dessn/autoscan/](http://portal.nersc.gov/project/dessn/autoscan/)

* Features created (manually): 38
* Images: 898,963
* Images of supernovas: 454,092
* Images not of supernovas: 444,871
* Images and FITS files can be found under "Images"
* Feature engineered CSV can be found under "Features"
    
**Side-note: FITS files are a structured means of storing data along with metadata (data about the data) such as date and time of the information gathered.**

Let's first try to understand the paper and its metrics so we know what we're hoping to beat. The central dataset in the paper is the first release of data from the _Dark Energy Survey Supernova program (DES-SN)_ paired with hand-crafted data.

**Side-note: Hand-crafting data, often referred to as feature engineering, is the process of looking at the data available and converting it into data that might be easier to feed into a model or let the model pick up on. In this case, the researchers took the images and extracted information such that the model would be able to pick up on certain things it wouldn't be able to before.**

_"The Dark Energy Survey (DES) is an international, collaborative effort to map hundreds of millions of galaxies, detect thousands of supernovae, and find patterns of cosmic structure that will reveal the nature of the mysterious dark energy that is accelerating the expansion of our Universe. DES began searching the Southern skies on August 31, 2013."_

Source: [https://www.darkenergysurvey.org/the-des-project/overview/](https://www.darkenergysurvey.org/the-des-project/overview/)


<figure>
	<img src="{{ '/assets/img/post-2/telescope.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Cerro Tololo Inter-American Observatory, high in the Chilean Andes.</figcaption>
</figure>

The dataset contains 444,871 images of human-verified artifacts, and 454,092 simulated transient events projected onto real images of galaxies. Essentially half the dataset contains manufactured images of supernovas, which seems weird, but _data augmentation_ has become a popular technique for bettering a model. Data––both in quality and quantity––is a big part of the equation when considering the accuracy of a model.


**Side-note: Data augmentation has been explored quite a bit, from simply changing the data that you have, to trying to create new data based on what you have ([GANs](https://hackernoon.com/how-do-gans-intuitively-work-2dda07f247a1) are a popular new way of doing this). This is seen in the image recognition field; if you have 10,000 colour images of cats and get the same 10,000 images in black and white, your dataset is now 20,000. Your model should be able to understand what a cat is, even if the colour is different.**

Supernovas are rare, so there's a lack of supernova images that can be used for training. _ATIDES_ make use of the few images of supernovas that they have, and overlay them on different galaxies. By using many different backgrounds (essentially adding unique space noise) with each supernova picture, researchers were able to build a strong dataset for their model.

**When data is sparse, is it a good idea to use "simulated images"?** 

Yes––to a certain extent. When the data is lacking, some data is better than no data. It's worth understanding that the model we build can only be as good as the data. So if half the dataset includes simulated images we should keep that in mind when doing production _inference_.


**Side-note: Inference is when we feed data through our model without updating it. This is used to see how the current model is doing without having it learn the data that is used (an important part of neural nets that will be talked about soon).**

With the knowledge that good data and model go hand-in-hand, we could see a huge improvement happening if we spent time creating higher quality simulated data. As we mentioned in a previous side-note, GANs could be helpful here.


**Our goal is differentiating between images containing artifacts and those containing transient objects (eg: supernova).**

In total, the survey will cover 1/8th the entire sky using a telescope at the [Cerro Tololo Inter-American Observatory](http://www.ctio.noao.edu/noao/) in Chile. The ATIDES paper uses a subset of this data and focuses on the supernova aspects of the Dark Energy Survey, referred to as DES-SN.

The collection of algorithms used in ATIDES is referred to as [Autoscan](http://portal.nersc.gov/project/dessn/autoscan/), a Python package for artifact rejection.

The paper describes how with the help of astronomers they were able to [feature engineer](https://en.wikipedia.org/wiki/Feature_engineering) 38 noteworthy characteristics of images. 

ATIDES uses feature engineering since the data is in image format and thus needs to be quantifiable in order to run within a random forest algorithm.

Here's what a small sample of the images looks like once they've been converted into a format that's easier to do data analysis on (feature engineered). Each row is an image, and each column is a feature (only 14 of 38 features shown).

<figure>
	<img src="{{ '/assets/img/post-2/data-preview.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Figure 1: First 10 Rows of the Feature Engineered Data.</figcaption>
</figure>


You can see an **ID** column which is how we know which object the data-point refers to and an **OBJECT_TYPE** column which is a binary indicator of whether the data-point contains a supernova (0) or not (1). In data science the column that you train your model on is called the target. In this case **OBJECT\_TYPE** is our target because it tells our model what the correct answer is in the question "is this a supernova or not?".

We have to make sure that we take both of these columns out of the data that gets fed into the model (something that is shown in our code). The feature names aren't too user friendly but the paper explains most of them. We did note that we found some naming convention issues between the data set and paper.

<figure>
	<img src="{{ '/assets/img/post-2/features.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Figure 1: First 10 Rows of the Feature Engineered Data.</figcaption>
</figure>

**Table 1: Autoscan's Feature Library**

With respect to exploring other model approaches, the paper also attempts using [Support Vector Machines](http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf) and [Adaboosted decision trees](https://en.wikipedia.org/wiki/AdaBoost) but random forest are found to have the best accuracy. The random forest algorithm used in ATIDES is from the popular [scikit-learn](http://scikit-learn.org/stable/) Python library (which makes a lot of things very easy and accessible).


**Side-note: scikit-learn is the swiss army knife of data science. For running a lot of models it's will make your life much nicer. It's a great starting point, but also a standard for quickly getting common classical machine learning models like logistic regression.**

How does random forest work? At a high level it creates a set of [decision trees](https://www.quora.com/What-is-an-intuitive-explanation-of-a-decision-tree) from the data and classifies the image based on a 'vote' among the decision trees.

<figure>
	<img src="{{ '/assets/img/post-2/decision-tree.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Visualization for a Random Forest.</figcaption>
</figure>


The random forest algorithm produces weightings for its features. For example in Autoscan, _r\_aper_psf_ was considered most important as per Table 1. This is useful because it informs the astronomer, whose domain knowledge supplied the features, which features are most relevant to identifying an object of interest.

Given this, it's neat to think of machine learning helping us see things we cannot see even if we were experienced astronomers––much like how x-rays machines help doctors see and understand things that normally wouldn't be possible.

**Understanding The Target and Metrics**

We spent a good amount of time understanding how the paper presents its results, which had a lot to do with how they classify results using the below confusion matrix.

<figure>
	<img src="{{ '/assets/img/post-2/confusion-matrix.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Confusion Matrix.</figcaption>
</figure>


The random forest's performance was evaluated on the missed detection rate (MDR) and the false positive rate (FPR).

MDR: represents the amount of objects that the model missed; where the model said "this is not a supernova" but it actually was.

FPR: represents the amount of objects that the model falsely labeled as a supernova; where the model said "this is a supernova" but it really wasn't.

Generally, if an algorithm is more conservative about identifying images as transients, it will miss more transients (higher MDR), but register fewer artifacts incorrectly as transients (lower FPR). It is preferable to have both low MDR and FPR. Both are defined below (with **F** being "false", **T** being "true, **p** being "positive", and **n** being "negative").


<figure>
	<img src="{{ '/assets/img/post-2/equation2.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Definition of MDR and FPR.</figcaption>
</figure>

This gets a little tricky when we introduce the _threshold_. To explain how the threshold is used here we have to explain what the model outputs. Since we're trying to see if a supernova is present (0/positive) or is not present (1/negative), the model output should be something similar. However, we want to know **_how_** sure the model is––this means we output a probability (or a number that can be used similarly), that is a number between 0 or 1. The thresholds used in the paper's figure below (0.4, 0.5, and 0.6) define **_how_** sure the model needs to be for us to consider it's confidence as a decision. 

A helpful way to understand how these two rates are related is try out this interactive ROC curve explanation: [http://www.navan.name/roc/](http://www.navan.name/roc/)

<figure>
	<img src="{{ '/assets/img/post-2/graph.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Definition of MDR and FPR.</figcaption>
</figure>

This figure represents _ATIDES_ results for all three thresholds. This ideally means a curve on the inside (left) of their curve.


## Our Approach and Baseline Model

When building a model, the most important first step is the data. We spent a good amount of time exploring and getting comfortable with the data to get an understanding of what is described and how.

The next step was to build a baseline model. What does baseline mean? A model that is simple or naive––that can act as a bottom bar that our actual model should be able to beat. It's a model that allows us to quickly check that our setup is working and ensure data is being properly imported.

We have done this a few ways, and although our baseline uses the feature engineered data (which we want to remove the need for in the future), it was helpful to check that our development environment was working and we could do something competitive.

Ok––a quick tease, the ATIDES paper mentions that further exploration could be done using Convolutional Neural Nets (CNNs), which would allow skipping the manual feature engineering steps needed in their process. A valid critique ATIDES adds is that CNNs (and neural networks in general) do suffer from a lack of transparency of what the model is doing. While we won't be discussing CNNs here, it will take our entire next update.

We decided for our baseline model that we would use the feature engineered data from _ATIDES_ and see how XGBoost would perform compared to the paper's use of random forest. XGBoost is a common algorithm usually used for regression and classification problems. It uses the concept of gradient boosting on-top of decision trees. If you're looking for a more in-depth info of the inner working of XGBoost you can can check out [this overview](http://xgboost.readthedocs.io/en/latest/model.html). XGBoost is often used in data science competitions and is really quick to get up-and-running.

First we needed to set up an environment that could handle a reasonable amount of compute.

#### Development Environment

We didn't do much initial assessment of different compute systems since we were already very comfortable using Amazon's Web Services (AWS).

Our initial needs were:



*   Quickly explore the data
*   Use jupyter notebooks, Tensorflow, or Keras
*   Handle anywhere from ~2.5 - 50GB of data

We spun up a [P2 AWS instance](https://aws.amazon.com/ec2/instance-types/p2/), using the [Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C). This gives a bunch of tools out of the box like PyTorch, Keras, and Tensorflow and some half decent GPUs to do the grunt work. Lots of options!


**Side-note: In the name of education we've open sourced all our code! We talk about this in the Code section.**

#### XG-Boost compared to Random Forest: detecting supernovas

Finally, let's talk about our super awesome (and totally, really complicated to run) first model!

If you look at the code you can see how simple it is to get something like this up and running. Even though this was a quick baseline model, it was used for a few reasons:



1.  We had spent a _ton_ of time defining and understanding the problem, which made us long for actually getting something up and running
1.  It forced us to familiarize ourselves with the actual data rather than just the problem
1.  We can have some simple baseline metrics to compare our future models against

With that said we can now look at our baseline model!

We've highlighted the winning score in the table below in green. Although it would have been just as easy to say we lost.


<figure>
	<img src="{{ '/assets/img/post-2/metrics.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Comparison of XGBoost and Random Forest.</figcaption>
</figure>

This gives us quite a bit of insight and helps us validate some assumptions that we had:

1.  The feature engineered data is good and holds a lot of predictive power
2.  Newer techniques (XGBoost in this case) are fairly competitive to certain problems with minor tuning
3.  This is going to require more work than an out-of-the-box solution

#### Code

We keep talking about making this a helpful tool for beginners and that's great let's do something that is actually useful to learn the _coding_ part of this.

To do this, we will **open source everything** that ends up making up the completed codebase. We have created a GitHub repo to host everything that you can use.

Sadly, we won't be going into how to use these tools if you aren't familiar with them… What we do have time for is giving a list of tools to Google:

*   GitHub
*   Python
*   Pip/Conda
*   iPython/Jupyter

The list of Python libraries can be found in the "requirements.txt" file in the repository.

[https://github.com/pippinlee/space2vec-ml-code](https://github.com/pippinlee/space2vec-ml-code)

Data in a pickle file can also be found [here](https://drive.google.com/open?id=1Pa4-imVbK7yfZuCX3mfF-mMae1eyhQqo).

#### Conclusion

These past two weeks were all about understanding the data and the problem. We looked at the paper's random forest model and feature engineering to classify supernovas.

We looked at understanding the paper's metrics and how they achieved them, spinning up a P2 instance on AWS to quickly get model building and building a half-decent baseline model.

In our next post our focus will be on using a modern machine learning technique that avoids this labour intensive process of defining features beforehand, and instead feeding images directly into the model with a Convolutional Neural Net.

#### Resources

[https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

[http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)


