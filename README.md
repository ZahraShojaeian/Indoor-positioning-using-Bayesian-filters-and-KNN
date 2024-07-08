# Indoor positioning using WiFi RSS signals
## Project Overview
This project aims to design an indoor localization system utilizing WiFi data for five distinct locations. The system uses the received signal strength (RSS) from three WiFi access points (A, B, and C) to determine the probability of being in each location.
## Objective
To Develop a program that, given a new scan <rssA, rssB, rssC>, outputs a probability vector <p1, p2, p3, p4, p5>, where pi represents the probability of being at location i.
## Data Preparation
### 1. Train and Test Split
Train Data:
NM points were filled with -100 dBm.
If NM points > 30% in a location, they were attributed to distance and replaced with -100.
Otherwise, considered as noise and removed from the training set.
Test Data:
All NM points were replaced with -100.
### 2. AP Tables and Likelihood Functions
Constructed AP tables for each WiFi sample in each location.
Fitted Gaussian distributions to use as Likelihood functions in the Bayesian filter.
For Wi-Fi 1 in location 5 (with only NM points), manually defined the Likelihood function.

## Bayesian Filter Model
### 1. Prediction with Naive Bayesian Estimator
Initial Step:
Defined a Prior probability for each location with a uniform distribution.
Iteration Process:
Used KNN classifier to filter most probable locations and updated Prior probability.
Calculated Posterior probability using Bayes rule from Prior and Likelihood functions.
Summed predictions of all WiFis for each location, normalized, and found the maximum.
Iterated until the predicted location probability exceeded 95%.
Updated Prior with Posterior in each iteration.
### 2. Test Phase
Split data into train and test sets (80% train, 20% test).
Conducted tests 90 times across 15 sample sets (3 per location).
Achieved an overall accuracy of 80.1%.

<img src="Confusion matrix.png" width="450">


### 3. Accuracy improvement:
The whole dataset in this project consisted of 15 data points from a real place and one can achieve a higher accuracy using a larger training set.


## Instructions to run the codes:



### Create a new Conda environment using the provided `my_environment.yaml` file: 

 
1.  ```bash
    conda env create -f my_environment.yaml -n <environment_name>
    ```
2. Activate the environment:

    ```bash
   conda activate <environment_name>
    ```

3. Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```
Or if it's a Conda environment:

conda create --name <environment_name> --file requirements.txt

### Run the codes
1. First run Data_process.py
2. Then run Baysian_estimator.py 

