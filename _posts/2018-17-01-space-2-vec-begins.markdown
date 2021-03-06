---
layout: post
title:  "Starting space2vec"
date:   2018-01-15
description: We've been working on space2vec for a while now. Here's the backstory to how it all started and what we're aiming to do over the next year.
url: http://space2vec.com/blog/space-2-vec-begins/
image: space4.jpg
---

### Week One

_The goal of these blog posts is to document our research into applying modern machine learning techniques in astronomy. We’ll write about what works and also what doesn’t. We write software by trade so these posts should hopefully give insight into HOW modern astronomy and machine learning works as we better understand them ourselves._

#### How It Started

_"Clustering the entire goddamn universe"._

It was early July this past summer when Cole first started flying these words around the DeepLearni.ng office. We were discussing dream projects to work on within the field of <a href="http://neuralnetworksanddeeplearning.com/" target="_blank">deep learning</a>, and a quick survey of the room confirmed our initial suspicions… space is pretty freaking awesome––but how would deep learning techniques actually work in a field with so much history and tradition behind their data analysis toolkit?

This question led us to spend the next few months diving deep into the space field, and talking with astronomers to see if there may be a use for *any* machine learning in astronomy. After all––astronomy is a data problem and data is what drives machine learning.

<figure>
	<img src="{{ '/assets/img/space3.gif' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig1. - Lightcurves as seen from Earth</figcaption>
</figure>

#### What Are We Doing?

The aim of this blog will be to document our progress––unfiltered stories, code, data––and how machine learning and astronomy can work together to better understand the skies above us. It will hopefully act as a guide and motivation on how *you* can travel down a similar path too.

#### About Us

We come from various backgrounds (computer science, web development, pure mathematics) but are extremely passionate about two things: 

1. Space: It’s amazing and we love it. Space forces us to ask big questions, leaving us with a child-like curiosity. We don’t have a formal background in astronomy but we <3 space and part of the process will be learning about the science of our skies.

2. Cutting through the hype: Neural networks don’t solve all data science problems. If our research shows that neural networks can’t be easily applied in astronomy, we’ll document it. We think it’s important to show the value of different applications and datasets––and that also means covering where things don’t work.

#### Our Initial Research

Although we had initial ideas of problems to tackle, such as using newer techniques from the academic world to "cluster the entire goddamn universe", we knew we had to be practical. This forced us to approach our research with a question:

**Can we find a _real, annoying_ problem in the astronomy community where we can help by applying modern machine learning techniques?**

As we began our research, we were very aware of our lack of astronomy knowledge, but to our delight the astronomers we spoke with were all wonderfully welcoming (the conversations will be an entire post themselves!). We also knew our depth of understanding would be shallow at best––therefore it was important to have astronomers explain quite a few technical aspects of astronomy to make sure we didn’t get stuck at 100 level questions. We started with the following:

#### What tools are modern astronomers using (i.e. computer applications, programming languages)?

We were amazed to find that a large portion of the astronomy community has followed the open science movement and is using open source technologies––this means we’re right at home with the many open source Python libraries available. We’ll be publicly posting all these resources in a separate post as we go forward.

We also found a super friendly community of astronomers eager to discuss how the whole astronomy field has changed over the last dozen years or so. In short, the data and discoveries have grown exponentially! Just look at exoplanet discoveries over the last 50 years ([and it’s about to get even wilder!](https://www.lsst.org/))

<figure>
	<img src="{{ '/assets/img/space1.gif' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig2. - Discovery of exoplanets over time</figcaption>
</figure>

#### Data is one of the biggest pieces in machine learning problems, so how is the data?

Astronomy is quickly becoming a big data problem. Projects like the [LSST](https://www.lsst.org/) will produce 15-30 terabytes of data a night. This is nuts, *for any industry*, and immediately stuck out to us––especially as part of our work on DeepLearni.ng’s upcoming product [Frontiers](http://deeplearni.ng/frontiers/), is trying to perfect massive data pipelines.

We also quickly learned the common ways that exoplanets (planets outside of Earth’s solar system) are discovered using data from different satellites. We’ll have another post talking about this subject later on.

<figure>
	<img src="{{ '/assets/img/space2.gif' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig3. - What does satellite data look like? This is an example of K2's data</figcaption>
</figure>

#### Are astronomers using machine learning?

**Yes**, but not much. Astronomy has traditionally had a human component to identify objects. One of the more interesting projects we found was Galaxy Zoo which crowdsources identification of galaxies from the [Sloan Digital Sky Survey](http://www.sdss.org/). Most examples of machine learning tend to be on the classical side, using techniques such as K-means clustering or [logistic ](https://en.wikipedia.org/wiki/Logistic_regression)[regression](https://en.wikipedia.org/wiki/Logistic_regression) (technical stuff that will be in a post about our models).

#### Are astronomers using modern techniques like neural nets?

**Not really.** For instance a quick look for mentions of "neural nets" or “deep learning” in relation to astronomy on [arxiv.org](https://arxiv.org/) doesn’t come up with more than a few results. After speaking to a Machine Learning PhD turned astronomer from the University of Toronto, we learned there’s a reason for this––you can’t just throw a standard model from scikit-learn at the available data, you need to fundamentally understand what you are doing on both the astronomy and modeling side. This is where we're hoping to shed the most light since this is how we operate as a company––make sure we understand the problem from many different perspectives, then start building a custom model that uses these insights. We will always try to come back to this point, *why can’t you just throw some out-of-the-box model at it?*

#### Deciding on a Problem

During our research we found this paper: [Automated Transient Identification in the Dark Energy Survey](https://arxiv.org/abs/1504.02936).

It outlines using a machine learning model (random forest) with data from the Dark Energy Survey Supernova program (DES-SN) to make identifying transients more efficient and accurate. It provided data, code, and good documentation of the model they built. We decided it’d be a good initial challenge to see if we can beat the paper’s model with a deep neural network.

Our next post will talk more about the task-at-hand, our plan for approaching the problem, as well as model architectures. Stay tuned!

We’d like to give a big thank you to [Nathalie Ouellette](https://twitter.com/angryastropanda?lang=en), [Dustin Lang](https://twitter.com/dstndstn), [Dan Foreman-Mackey](https://twitter.com/exoplaneteer?lang=en), and [Ross Fadely](https://twitter.com/rossfadely?lang=en) for taking their time to answer our beginner questions and giving valuable input to getting this project going!