# SBlur: An Obfuscation Approach for Preserving Sensitive Attributes in Recommender System
This repository releases the Python implementation of our paper "SBlur: An Obfuscation Approach for Preserving Sensitive Attributes in Recommender System" where SBlur stands for Strategic Blurring Sensitive Attributes.

## Abstract
User interaction in the recommender system is treated as a way of expressing user preferences, which later serve as input to provide more accurate recommendations. However, such interaction data can be exploited to infer users' private attributes, including gender, age, and personality traits, posing significant privacy implications. Existing obfuscation-based approaches endeavor to mitigate these vulnerabilities by adding or removing interactions from user profiles before or during recommender algorithm training. Nevertheless, these methods often compromise recommendation accuracy while facing challenges such as the cold-start user problem and the 'rich get richer' effect, undermining recommendation diversity. To address these constraints, we propose SBlur, a strategic obfuscation approach designed to preserve users' attribute privacy while balancing the privacy-accuracy-fairness trade-off and enhancing diversity. SBlur conceals gender inference attacks by strategically adding and removing items, supported by a combined similarity measure that integrates rating-based and genre preference-based similarities. This combined similarity enables precise user profile personalization for obfuscation, particularly in cold-start scenarios. We evaluate SBlur using three popular datasets (ML100k, ML1M, and Yahoo!Movie) and three state-of-the-art recommendation algorithms (UserKNN, ALS, and BPRMF). Experimental results demonstrate that SBlur achieves a balanced trade-off between privacy, recommendation accuracy, and fairness while promoting recommendation diversity.

## Python packages to install
* Numpy
* Pandas
* sklearn
* scipy
## Dataset
For this study, three public datasets are used, which can be found in the following sources
* For MovieLens 100k and 1M dataset: https://grouplens.org/datasets/movielens/
* For Yahoo!Movie dataset: https://webscope.sandbox.yahoo.com/
