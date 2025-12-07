# Predicting risk factors for maternal mortality

## Authors

Apoorva Srivastava, Guaner (Gloria) Yi, Jeffrey Ding, Randall Lee

## About

In this project, we aim to predict maternal health risk levels using clinical measurements from pregnant individuals in rural Bangladesh. Maternal mortality rate is a critical public health concern, particularly in low-resource settings where early identification of high-risk pregnancies can enable timely medical interventions and save lives. 

Here, we attempt to build a classification model using the Support Vector Classifier algorithm which can use maternal health measurements to predict the risk intensity level (low, medium or high) of pregnant individuals based on their clinical measurements. Our final classifier performed fairly well on an unseen test data set, with a weighted recall score of 0.77 and an overall accuracy of 0.77. Out of the 305 test data cases, it correctly predicted 235 cases.

Our model showed particularly strong performance in identifying high-risk pregnancies, achieving an AUV of 0.943 for the high-risk class, compared to 0.820 for low-risk class and 0.814 for medium-risk class. Examining the confusion matrix reveals that the model made 68 correct high-risk predictions out of the 75 actual high-risk cases (recall of ~91%) demonstrating its strength in correctly identifying the most critical cases. The model did struggle a little distinguishing low-risk and medium-risk pregnancies, with 27 low-risk cases classified as medium-risk and 24 medium-risk cases classified as low-risk. The most concerning errors, however, are the 4 high-risk cases misclassified as medium-risk and the 3 high-risk cases misclassified as low-risk, as these false negatives can result in pregnant individuals not receiving necessary medical intervention. We recommend further research to improve the model's sensitivity to high-risk cases and better differentiate between medium and low-risk categories before it is ready to be put into production in clinical settings. Additional feature engineering or exploring ensemble methods may help reduce these critical misclassifications. 

The data set used in this project consists of health conditions of pregnant individuals from rural areas of Bangladesh, created by Marzia Ahmed at Daffodil International University. It was sourced from the UC Irvine Machine Learning Repossitory and can be found [here](https://archive.ics.uci.edu/dataset/863/maternal+health+risk). Each observation in the data set corresponds to a pregnant individual's health profile, comprising a risk intensity level (low, medium or high risk) and associated clinical measurements including demographic information (age) and vital signs (systolic blood pressure, diastolic blood pressure, blood glucose concentration, body temperature and resting heart rate). The data set was collected via IoT-based risk monitoring system from hospitals, community clinics and maternal health cares in rural Bangladesh. 
 
## Report

The report can be found [here](https://github.com/GloriaYi/Maternal-Health-Risk-Classifier/blob/main/notebooks/health_analysis.pdf).

## Usage

### Setup

> If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub repository.

1. Navigate to the root of this project on your computer using the
   command line and enter the following command:

``` 
make up
```

2. Browse to this url: http://localhost:8886/

3. To run the analysis, open a terminal and run the following commands:

```
python scripts/fit_maternal_health_risk_classifier.py \
    --training-data=data/processed/maternal_health_risk_train.csv \
    --pipeline-to=results/models \
    --plot-to=results/figures \
    --seed=522
```

### Clean up

1. To shut down the container and clean up the resources, enter the following command to stop the Docker compose services:

``` 
make stop
```

2. Enter the following command to stop the Docker compose services:

``` 
make remove
```

## Developer notes

### Developer dependencies

- [Docker](https://www.docker.com/) 

### Adding a new dependency

1. Add the dependency to the `environment.yml` file on a new branch. If you have installed the package into the environment, make sure you are in the environment `health_analysis_env` and run `conda env export --from-history | grep -v "^prefix" > environment.yml` to export the environment automatically.

2. Run `python utils/update_enviroment_yml.py --root_dir="." --env_name="health_analysis_env" --yml_name="environment.yml"` to append the version numbers of the packages.
   
3. Run `conda-lock -f environment.yml -p osx-arm64 -p osx-64 -p linux-64 -p win-64` to update the `conda-lock.yml` file.

4. Re-build the Docker image locally to ensure it builds and runs properly.

5. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

6. Update the `docker-compose.yml` file on your branch to use the new container image (make sure to update the tag specifically).

5. Send a pull request to merge the changes into the `main` branch. 

## License

The Maternal Health Risk Classifier report contained herein are licensed under the [Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [the license file](LICENSE) for more information. If you re-use this, please provide attribution and link to this webpage. The software code contained within this repository is licensed under the MIT license. See [the license file](LICENSE) for more information.  

## References

Ahmed, Marzia. "Maternal Health Risk." UCI Machine Learning Repository, 2020, https://doi.org/10.24432/C5DP5D.

Ahmed, Marzia et al. “Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT).” Lecture Notes in Electrical Engineering (2020): n. pag.
