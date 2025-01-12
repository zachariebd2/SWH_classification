# Snow and Cloud Classification in Historical SPOT Images: An Image Emulation Approach for Training a Deep Learning Model Without Reference Data

Author: zacharie barrou dumont [DOI](https://orcid.org/0009-0004-9515-5757), zachariebd@hotmail.com

This code was developped at the CESBIO during my PHD thesis "Reconstruction by satellite imagery of the snow cover of the Alps and Pyrenees over a period covering the last 37 years" (https://theses.fr/s295609?domaine=theses) with the Paul Sabatier university and under the supervision of Simon Gascoin [DOI](https://orcid.org/0000-0002-4996-6768) and Jordi Inglada [DOI](https://orcid.org/0000-0001-6896-0049). 

It is the step-by-step process of i) emulating pseudo-spot images from sentinel-2 images, ii) training a U-net model to identify snow/no-snow/cloud pixels in the pseudo-spot using reference snow maps derived from sentinel-2 (2B products) and iii) apply the trained model on historical SPOT images.

This code was developped to work using the CNES High Performance resources (data availability, task management, gpu) and will need modifications for a correct implementation in another environment. 

Every aspects of the code can be run from main.ipynb:

-A) CREATE TRAINING DATASET FOR THE UNET
  -A.1) CREATE A GKDE DISTRIBUTION OF SWH SATURATION AND MINIMUM REFLECTANCE VALUES
  -A.2) DOWNLOAD SENTINEL-2 DATA (L1C REFLECTANCE, MAJA CLOUD MASKS, SNOW)
  -A.3) GET AUXILIARY DATA  (DEM, HILLSHADE, MASK)
  -A.4) SPLIT DATA INTO PATCHES
  -A.5) "SPOTIFY" SENTINEL-2 PATCHES INTO PSEUDO-SPOT
  -A.6) GENERATE TRAINING DATASET
-B) TRAIN THE UNET
  -B.1) CREATE A MODEL
  -B.2) MODEL EVALUATION USING PSEUDO SPOT DATA
  -B.3) MODEL EVALUATION USING HISTORICAL SPOT DATA
  

This work led to the publication of a paper detailing the method (Barrou Dumont et al. 2024) [DOI](https://doi.org/10.1109/JSTARS.2024.3361838), however, changes have been made to the code after the publication:

Haze and high transparent and semi-transparent clouds could be detected in Sentinel-2 images thanks to their higher number of spectral band and their radiometric quality. These clouds were almost invisible in pseudo-SPOT images, creating "false" clouds in the training dataset and causing an overestimation of cloud pixels by the U-net.
To solve this issue, we filtered the cloud pixels from the training data according to how they were detected. The cloud mask in level 2B products was generated with the MAJA software which provides information on the way that a cloud pixel was detected. In particular, one way to detect cloud was a combination of pixel-wise mono-temporal reflectance thresholds in the blue, red, NIR, and SWIR bands. Cloud pixels detected with the mono-temporal threshold were certain to be also visible in a SPOT 1-5 instrument. Keeping only those cloud pixels reduced the amount of misleading labels.
Cloud shadow pixels were also removed from the training to capitalize on the observation from Barrou Dumont et al. (2024) that the U-net was able to detect snow in less illuminated areas. We also changed the number of times the training dataset was run through the neural network (number of epochs). In Barrou Dumont et al. (2024), the training starts with multiple short parallel preliminary trainings of 40 epochs to look for the best weights initialization where the U-net can converge to a state of minimum loss, and continues with a more intensive
training of 200 epochs. A solution to ensure that the U-net converged to the state of minimum loss afforded by its architecture and the training data was to remove the limit over the number of epochs and stop each training when the U-net stops improving for a given amount of epochs (40 for a preliminary training, 200 for an intensive training).

