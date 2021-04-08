# Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection
Unofficial pytorch implementation of  
Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM)  
\- Guodong Wang, Shumin Han, Errui Ding, Di Huang  (2021)  
https://arxiv.org/abs/2103.04257v2 

Usage 
~~~
python train.py --dataset_path '...\mvtec_anomaly_detection\bottle' --project_path 'path\to\save\results'
~~~

MVTecAD pixel-level AUC-ROC score 
| Category | Original paper | This code |
| :-----: | :-: | :-: |
| Bottle | 0.988 | 0.984 | 

Under testing.