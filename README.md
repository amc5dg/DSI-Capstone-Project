# Night Sky Net: Exploring deep space using convolutional neural networks

Classifying galaxy morphologies has long been a challenging problem in Astrophysics. Historically, galaxies were classified by hand by research scientists and graduate students. This task was scaled by using crowdsourcing using civilian scientists ([galaxy zoo project](https://www.galaxyzoo.org)) with the release of data from the first large scale sky surveys containing millions of images of galaxies.

My project implements convolutional neural networks (CNN) to learn relevant features and classify galaxies. Successful classification using CNNs is much quicker than being carried out by scientists, and likely more reliable than crowdsourcing. The increase in efficiency will allow progress to continue to scale with the data collected from increasingly powerful telescopes, and may add insight to complex and faint images that humans do not do well with classifying.

### Outline

* [Data Acquisition](#data-acquisition)
* [Data Munging and Image Pre-Processing](#data-munging-and-image-pre-processing)
* [Neural Network Architecture](#neural-network-architecture)

### Data Acquisition

I obtained coordinates, number of crowd-sourced votes, and percentage of votes for each class from the Sloan Digital Sky Survey (SDSS) SQLServer. Using these coordinates, I scraped the SDSS catalog archive for 380,000 images of galaxies.

### Data Munging and Image Pre-Processing

Using these data, I assigned each image to a class and trained
the network to classify galaxies as spirals face on or edge on,
ellipticals, or mergers. 

### Neural Network Architecture

Stay tuned for more details!

![](presentation/neural_net_architecture.png)
<img src="presentation/neural_net_architecture.png" width="250">
