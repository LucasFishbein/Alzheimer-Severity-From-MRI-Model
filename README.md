# Alzheimer's Disease Severity MRI Classification Model
**Created by Lucas Fishbein**

<div>
<img src="https://github.com/LucasFishbein/Positive_Tweet_Sentiment_CNN_Model/assets/117129342/1f5c06a0-e71c-4ae4-a802-3fff655cb4ec" width="500">
</div>

# Getting Started with this Repository

The MRI Image Dataset can be downloaded from [Kaggle.com](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).
Specific instructions for set up to ensure reproducibility are located in the juptyer notebook "Getting Started with this Repository" section.

# Project Overview and Business Problem

The overall goal of this project was to utilized machine learning classification techniques to predict the individual diagnoses of Alzheimer's Disease (AD) severity from a single cross-sectional structural brain MRI scan. A convolutional neural network (CNN) was built out and trained on pre-diagnosed MRI images sourced from various separate studies. The images are pre-labeled into one of these four target categories of increasing AD disease severity; "No Impairment", "Very Mild Impairment", "Mild Impairment", and "Moderate Impairment". 

The current diagnostic process for Alzheimer's disease is costly and labor intensive requiring a trained physician to compile a patient's medical history, brain imaging, physical and cognitive symptoms, physical and behavioral test results as well as often an interview with a close family member or friend \([alz.org](https://www.alz.org/alzheimers-dementia/diagnosis)\). CNNs and other machine learning models are powerful tools that may be able to reduce the load on physicians by aiding in the individual diagnosis of patients along the AD disease spectrum through a single MRI image. This has the potential to save time, money and labor while providing an instantaneous diagnosis that could lead to faster and more accurate treatments.

Alzheimer's Disease is the most common neurodegenerative disorder, effecting about 1 in 9 people over the age of 65. There are estimates that the population of those living with AD will almost double by 2050 compared to current numbers. AD is the leading cause of dementia impacting memory, thinking and behavior and is associated with up to 80% of all dementia diagnoses \([alz.org](https://www.alz.org/alzheimers-dementia/facts-figures#:~:text=More%20than%206%20million%20Americans%20of%20all%20ages%20have%20Alzheimer's,older%20(10.7%25)%20has%20Alzheimer's)\). Dementia is the clinical syndrome characterized by progressive decline in two or more cognitive domains, these can cause the loss of abilities to perform instrumental and/or basic activities of daily living (Weller et al., 2018)

During the development of Alzheimer's disease the brain shrinks, atrophies and the cells within it die causing loss of function. Magnetic Resonance Imaging (MRI) is a vital resource used in the clinical assessment of patients with suspected Alzheimer disease, allowing clinicians to help assess brain function through visualization of the neuropathology. Diagnostic criteria recommend the consideration of structural abnormalities visible through MRI, atrophy of the medial temporal structures as well as others are now considered valid diagnostic markers for AD.

# Model Performance

The final model performed very well a novel test group, accurately classifying 945 of 960, 98.44%, of the test MRI images input through the model. With a macro avg F1 score of ~99% the model does not exhibit excess type 1 or type 2 errors and has an acceptable overall loss statistic of 0.08.


![Screenshot 2023-07-10 at 5 29 31 PM](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/97aad6b6-e4f5-48f5-a396-d93349c4e228) 


![Screenshot 2023-07-10 at 5 12 58 PM](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/e5d22a19-4592-4f77-b3d6-64fe8d732f97)

# Database Understanding

The present dataset is comprised of 6400 axial Magnetic Resonance Images from various sources which have been pre-classified by a trained physician based on the level of mental impairment the patient exhibited. 


![dataset_class_balance](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/56ce6cbd-8b86-4f8d-85c3-4cfa9e1060e5)


The present dataset was was uploaded to [Kaggle.com](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images/) by Sarvesh Dubey in 2019. The MRI images were acquired using a 1.5 Tesla MRI scanner with a T1-weighted sequence. The images have a resolution of 176×208 pixels and are in the “.jpg” format. All images have been pre-processed to remove the skull.

This Limitations of this dataset are explained in the limitations section of this document 
 
# Data Preprocessing and Exploratory Data Analysis 

Each image was resized and converted into an array of size 224 ,224 ,3 and then normalized by a rescaled factor of 1/255 to convert the pixel intensity range to a [0,1] scale.

## Creating Average Images of each Class

In order to elucidate key structural differences between the classes, the average image for each class was computed. It can been seen that as impairment level increases, there is an overall loss in tissue density exhibited by the loss in smoothness and increased contrast within the image.

![Class_Mean_Images](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/e675f336-4f7f-40f9-9000-467622c561ba)

From here the brightness intensity of each pixel was with the images was compared in order to bring to light the key difference between the class average images.

![Class_Mean_Comparisons](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/f614aab3-6565-40b7-8b1c-3b8aef5aba09)

In the figure above, whiter ares represent a greater difference, while darker spots represent less difference between the images being compared. Examining the top row, which compares no impairment to the other groups, it is clear that as AD severity increases, the structure of the brain changes more and more from the no impairment or control group. The differences seem to be most prevalent in the cerebral cortex as well as Medial frontal and the structures surrounding the later ventricles.

## Feature Description through Histograms of Oriented Gradients 

Histograms of Oriented Gradients (HOGs) are feature descriptors that focus on the structure of an image and extract information on edge magnitudes as well as edge orientation in order to extract the most important information in an image.
![Class_Mean_HOGs](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/d00d8a6f-4e77-42d6-8de9-41c929ef16cc)

 HOGs of the average class images were created and It is somewhat subtle but it can be seen that there are stronger color gradients and increased directionality as AD severity progresses, this is evident by the increase in overall size and directionality of the orientation lines as well as the increased amount of red and yellow color variety as severity increases (Warmer colors represent stronger gradients and cooler represent weaker). 

The greatest contrast can be seen near the natural edges of the tissue around the ventricles as well as the longitudinal and lateral fissures. This makes sense as these areas tend to misshape and open up with increased atrophy.

What this shows is that similar to what was seen when comparing. the base average MRI images, in that the more severe forms of AD have a much less smooth coloring to them with increased number of edges and stronger color gradients, this is likely due directly to loss of dense tissue and stronger atrophy in these more severe cases.

# Modeling

## Train-Test Split 

The entire dataset was split into three groups, a training, a testing and a validation group at a ratio of 7 : 1.5 : 1.5. The training and validation sets were used to train and validate the model. The testing set was kept completely out of the training process and was used a novel dataset on which the trained model was tested.

## Models

5 different modeling philosophies were built out, iterated on and tested, with the best performing model being chosen as the Final Model for the project. Model performance was based largely on classification accuracy and recall scores.  

The 5 different modeling philosophies were:
1. K Nearest Neighbors
2. CNN Using Vectorized Data
3. CNN Using Augmented Images to balance classes
4. CNN Using SMOTE to balance classes
5. CNN Using Whole Image Data 

The Best Performing Iterations of each modeling philosophy were compared with the Whole Image modeling being the overall best performing. 

![Screenshot 2023-07-11 at 2 31 22 PM](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/93ad24a8-75ea-4000-b9c5-d105e88a06ea)

## Final Model

The final model of this project is a sequential CNN image classification model that was trained on a non-augmented dataset using whole images of shape (224, 224,  3) as input. This model performed at a recall rate of ~98.6% and an overall accuracy of ~98.4% on the novel testing set of 960 images, incorrectly classifying only 15 images. 

Summary of Model's Layers:

![Screenshot 2023-07-11 at 2 36 36 PM](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/63c853bd-2ef0-40d7-b49c-bf21408a8072)

Model's Performance on 960 test images:

![Screenshot 2023-07-10 at 5 29 31 PM](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/97aad6b6-e4f5-48f5-a396-d93349c4e228) 


![Screenshot 2023-07-10 at 5 12 58 PM](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/e5d22a19-4592-4f77-b3d6-64fe8d732f97)


# Model Use Case and Visualizing Features being Identified by Model

As an example use case for this model, an Image was ran through the sequential CNN model and feature maps for each layer were extracted to create visual representations of what the model is identifying to help with its classification. 

An example of part of a feature map is exhibited below:

<img width="281" alt="Screenshot 2023-07-11 at 2 45 56 PM" src="https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/assets/117129342/ffa1e08a-8a41-4bf0-ba03-95d675028e68">

### Feature Map Analysis

The figure above shows a snippet fo the features the model is identifying to help classify an MRI image that has been pre-labeled as Moderate Impairment. It can be seen that the network seems to largely be identifying structural differences in the gray matter of the cerebral cortex and near the edges of the lateral ventricles as the highlighted yellow color is most dense in these areas. The lateral areas of cerebral cortex seem to be of higher importance overall with low feature importance being ascribed to the medial features of the brain aside from the the edges of the ventricles.

This type of information can be used by physicians to better understand a single patient as well as potentially illuminate patterns of overall disease progression.

# Conclusions

Through the iterative training of a variety of different styles of MRI image classifications models, a final model has been produced that classifies a single MRI image into their Alzheimer's Disease severity level at an accuracy level of 98.4% with recall and precision scores of 98.6% and 99.0%. Overall this model performed very well at classifying single MRI images into the four target Alzheimer's Disease severity class; "No Impairment", "Very Mild Impairment", "Mild Impairment", and "Moderate Impairment". 

It seems that the model has successfully been able to identify features of atrophy and neural degeneration that are consistent with those that are accepted as pieces of the Alzheimer's diagnostic schema. This is evident in the model ascribing high feature importance to areas of the cerebral cortex that typically experience high rates of degeneration within AD patients. Further testing on larger datasets would need to occur to further distill exactly what features provide the greatest diagnostic classification information.

# Next Steps

In order to improve the overall power and general usability of this project, a much larger and more balanced dataset should be procured. In the modern day MRI scans are taken at a variety of different strengths and within different machines, in order to create a tool that can be used as widely as possible, all of these standard MRI imaging techniques would have to be considered and included within the training groups. 

Ideally this model could be utalized as an autonomous primary diagnostic tool and information obtained from the model could be used as a reference points for further research into the mechanisms at play within the disease.

## Limitations


* The largest limitation of this study was that the dataset had very unbalanced classes with the minority groups having far fewer examples than the majority groups
* The origins and preprocessing procedure of the MRI images in this dataset are were not fully documented. 
* Due to a lack of computation power and time, more complex CNN models that were theorized were not executed 

# For More Information
See the full analysis in the [Jupyter Notebook](https://github.com/LucasFishbein/Alzheimer-Severity-From-MRI-Model/blob/main/Alzheimer's%20Severity%20Classification%20Model.ipynb)

For additional info, contact Lucas Fishbein at FishbeinLucas@gmail.com

## Repository Structure

```
├── .gitignore
├── CONTRIBUTING.md
├── Alzheimer's Severity Classification Model.ipynb
├── data
├── trained models
├── LICENSE.md
└── README.md
```

