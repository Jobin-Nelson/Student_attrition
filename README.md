# Student attrition

## [Project Description](https://github.com/Jobin-Nelson/Student_attrition/blob/master/input/Capstone%20Project.pdf)
***Clearwater State University*** offers a wide variety of degree programs, from online degrees to a doctorate in education. Programs are offered in the streams of the arts, education, business & nursing.
Some key strategic goals for the University are:
- Increase enrolment of students
- Improve retention, progression, and graduation rates
- Recruit better academically qualified undergraduate and graduate students
- Increase external funding and recognition

### [Objective](https://github.com/Jobin-Nelson/Student_attrition/blob/master/input/Capstone%20Project.pdf)
1. Identify key drivers of early student attrition
2. Build a predictive model to identify students with higher early attrition risk
3. Recommed appropriate interventions based on the analysis

### [Data](https://github.com/Jobin-Nelson/Student_attrition/blob/master/input/Student%20Applications%20%26%20Performance.xlsx)
Primarily, data will be of the following high-level categories:
- Student Application data (demographics, High school GPA)
- Financial indicators (financial background, eligibility, quantum of aid)
- Course Preferences
- Performance Record

## Overview
- [Basic Exploratory Data Analysis](https://github.com/Jobin-Nelson/Student_attrition/blob/master/src/Student_attrition.ipynb): EDA to understand student profiles, early attrition rates in various programs, how does attrition change across various combinations of student characteristics
- [Data Cleaning](https://github.com/Jobin-Nelson/Student_attrition/tree/master/src/cleaned_data): Raw data is not optimized for modeling, it contains considerable missing values which we need to treat before any form of modeling
- [Feature Engineering](https://github.com/Jobin-Nelson/Student_attrition/blob/master/src/Student_attrition.ipynb): Forming better features and removing reduntant information from the data to facilitate modeling stage
- [Modeling](https://github.com/Jobin-Nelson/Student_attrition/blob/master/src/Student_attrition.ipynb): Trying out different models to get a baseline performance
- [Hyperparameter Tuning](https://github.com/Jobin-Nelson/Student_attrition/tree/master/src/tuning): Performing hyperparameter tuning on promising models with Optuna and HyperOpt to optmize the results
- [Report Building](https://github.com/Jobin-Nelson/Student_attrition/blob/master/report/Attrition_report.pptx): Presenting the main takeaways and insights gained through the analysis in the form of Powerpoint report

*Tools: pandas, seaborn, scikit-learn, XGboost, Optuna*