# Night Sky Net: Exploring deep space using convolutional neural networks

Classifying galaxy morphologies has long been a challenging problem in Astrophysics. Historically, galaxies were classified by hand by research scientists and graduate students. This task was scaled by using crowdsourcing using civilian scientists ([galaxy zoo project](https://www.galaxyzoo.org)) with the release of data from the first large scale sky surveys containing millions of images of galaxies.

My project implements convolutional neural networks (CNN) to learn relevant features and classify galaxies. Successful classification using CNNs is much quicker than being carried out by scientists, and likely more reliable than crowdsourcing. The increase in efficiency will allow progress to continue to scale with the data collected from increasingly powerful telescopes, and may add insight to complex and faint images that humans do not do well with classifying.

### Outline

* [Data Acquisition](#data-acquisition)
* [Data Munging and Image Pre-Processing](#data-munging-and-image-pre-processing)
* [Neural Network Architecture](#neural-network-architecture)

### Data Acquisition

I obtained coordinates, number of crowd-sourced votes, and percentage of votes for each class from the [Sloan Digital Sky Survey (SDSS) SQLServer](http://cas.sdss.org/dr8/en/tools/search/sql.asp). Using these coordinates, I scraped the SDSS catalog archive for 380,000 images of galaxies.

### Data Munging and Image Pre-Processing

Using the metadata, I chose to focus on four classes: Edge-on and Face-on spirals, Elliptical galaxies, and merging galaxies. (Edge-on vs. Face-on is not a scientifically interesting distinction, but are very visually different.)
Example images of each data type are shown below

![](presentation/galaxy_examples.png)

The face-on spiral class was made by combining the clockwise and anti-clockwise spiral classes from the original dataset, while the other three classes were all distinct. I dropped the ``don't know`` and ``combined spiral`` classes as they were too ambiguous to get robust results from a classifier. Once I had distilled to these 4 classes, I re-normalized the probabilities of class membership so that they totalled to 1.

To select images for the net to train on, I chose only images with a probability of class membership >= 95%, and trained the model as a classifier. Later, I will predict probabilities of class membership using this net on images with lower certainty from the human classifications, and compare this with the human-generated probabilities.

All images downloaded from the SDSS catalog archive were 120x120 pixels.




### Neural Network Architecture

The neural network was designed using keras.

![](presentation/neural_net_architecture.png)


### Acknowledgements

A great thank you to all of the instructors and my fellow classmates at Galvanize, whom I drew inspiration from each and every day.

References that gave me jumping off points:
[Andrej Karpathy Neural Networks Lectures](https://www.youtube.com/watch?v=gYpoJMlgyXA)
[Dieleman, S. et al. 2014](http://arxiv.org/pdf/1503.07077.pdf)

Lastly, thank you to my late night coding supervisors, pictured below.

![](presentation/coding_supervisors.jpg)
