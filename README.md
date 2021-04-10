### Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection implementation (unofficial)
Unofficial pytorch implementation of  
Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM)  
\- Guodong Wang, Shumin Han, Errui Ding, Di Huang  (2021)  
https://arxiv.org/abs/2103.04257v2 

Usage 
~~~
python train.py --dataset_path '...\mvtec_anomaly_detection\bottle' --project_path 'path\to\save\results'
~~~

MVTecAD pixel-level AUC-ROC score (mean of 3 trials)
| Category | Original paper | This code |
| :-----: | :-: | :-: |
| bottle | 0.988 | 0.984(1)|

Under testing.  

notice : This code is not verified yet. Depending on the test results, the code can be modified.