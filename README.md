# Tackling Domain-Specific Winograd Schemas with Knowledge-Based Reasoning and Machine Learning
This repository is related to the work of our academic paper that can be found as below:<br> 
https://arxiv.org/abs/2011.12081 

The detailed explanation of our work is given in this paper. Our work is mainly about tackling Winograd schemas in the specific domain defined by us - the *thanking* domain. We developed the advanced high-level knowledge-based reasoning method by modifying the method of Sharma (2019) which uses Answer Set Programming and K-Parser. In addition, we suggest an ensemble method of combining knowledge-based reasoning and machine learning, which gives a better performance in the thanking domain than each single method. This repository gives the codes to demonstrate our advanced high-level knowledge-based reasoning method described in Section 4 of our paper. If you want to cite our work, the following is the citation: 
```
@misc{hong2020tackling,
      title={Tackling Domain-Specific Winograd Schemas with Knowledge-Based Reasoning and Machine Learning}, 
      author={Suk Joon Hong and Brandon Bennett},
      year={2020},
      eprint={2011.12081},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## The data set for the experiments
The Winograd schema sentences used in the experiments are extracted from WinoGrande (Sakaguchi *et al.*, 2020). These sentences belong to the thanking domain defined by us.
1. krr_ws_both_paired.csv : the 80 sentences that can be paired (only this data set is used for the two experiments in our paper)
2. krr_ws_both_non_paired.csv : the 91 sentences that can 'not' be paired

In the reasoning process of our method, we also used an external sentiment lexicon dictionary (Hu and Liu, 2004). The dictionary is in "sentiment_lexicon" directory.

## How to run the source code of the advanced high-level knowledge-based reasoning method 
This code is written in python, and here is the python version requirement:
```
python 3.5.x or python 3.6.x or python 3.7.x 
```
The following four python packages are needed (using their latest version should be fine):
```
clyngor
numpy
pandas
xlrd
```
Here is the related link about clyngor: https://pypi.org/project/clyngor/ <br>
To run clyngor in python, the binary file of Clingo is needed which can be downloaded here (in our implementation, we used the version of 5.4.0): https://github.com/potassco/clingo/releases <br>
For example, if you use windows, you can simply download "clingo-5.4.0-win64.zip" file in the Assets section from the clingo releases website above. Then, the next step is to unzip the zip file, and "clingo.exe"(the binary file) file will be found in the unzipped directory. Please make sure that "clyngor.CLINGO_BIN_PATH" in the line 9 of main.py MUST BE the location of the file in your own local directory:
```
clyngor.CLINGO_BIN_PATH = 'YourDirectory/clingo.exe'
```
Finally, by typing the commands below, our method is run on either experiment 1 data set or experiment 2 data set described in our academic paper. If you use a command prompt in windows, you SHOULD RUN IT AS ADMINISTRATOR.

For experiment 1,
```
python main.py --experiment_type ex1
```
For experiment 2,
```
python main.py --experiment_type ex2
```
## References
Minqing Hu and Bing Liu. Mining and Summarizing Customer Reviews. In *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-2004)*, Aug 22-25, 2004, Seattle, Washington, USA. <br>
Keisuke Sakaguchi, Ronan Le Bras,Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In *AAAI-20*, 2020.<br>
Arpit Sharma. Using answer set programming for commonsense reasoning in the winograd schema challenge. *arXiv:1907.11112[cs.AI]*, 2019.
