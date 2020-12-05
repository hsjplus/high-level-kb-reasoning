# Copyright 2020 The authors of the paper (https://arxiv.org/abs/2011.12081) 
# The paper's title: "Tackling Domain-Specific Winograd Schemas with Knowledge-Based Reasoning and Machine Learning"

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from clyngor import ASP, solve
import clyngor

import argparse
import pandas as pd
import numpy as np
import re

clyngor.CLINGO_BIN_PATH = 'C:/Users/AERO 15X V9/Desktop/dev/clingo-5.4.0-win64/clingo.exe'
# clyngor.CLINGO_BIN_PATH = 'YourDirectory/clingo.exe'

configuration_dict = {'ex1' : 'ex1_table.csv', 'ex2' : 'ex2_table.csv'}

# experiment type is either 'ex1' (experiment 1) or 'ex2' (experiment 2)
def predict(experiment_type=None):
    ex_table = pd.read_csv(configuration_dict[experiment_type])
    
    def extract_train_pid():
        # check the duplicates between the train set and the test set
        duplicates_list=[]
        for pid in ex_table['train']: 
            if pid in ex_table['test'].tolist():
                duplicates_list.append(pid)
    
        if len(duplicates_list) == 0:
            print('No duplicates between the train set and the test set')
        else:
            print('Duplicates: ', str(duplicates_list))
        
        print('The number of sentences in the train set: ', str(len(ex_table['train'].unique())))
        print('The number of sentences in the test set: ', str(len(ex_table['test'].unique())))
    
        # train_pid_list has indexes of the sentences in the train set 
        train_pid_list = ex_table['train'].tolist()
        # add the general rules indexes to train_pid_list 
        train_pid_list.append('general')
        train_pid_list.append('general_semantic')
        
        return train_pid_list
    
    train_pid_list = extract_train_pid()
    
    with open('facts_check_person.lp', 'rt') as file:
        facts_check_person = file.read()

    with open('rules_reasoning.lp', 'rt') as file:
        rules_reasoning = file.read()

    # add the semantic role rules which can be derived from the train set only
    df = pd.read_excel('rules_semantic_roles.xlsx')
    rules_semantic = ''
    for i in range(len(df)):
        if df['pID'][i] in train_pid_list:
            rules_semantic = rules_semantic + df['rules'][i] +'\n'
    
    # add the background knowledge principles which can be derived from the train set only     
    bg_id_list = []
    for i in range(len(df)):
        if df['pID'][i] in train_pid_list:
            if df['bg'][i] is not np.nan:
                bg_id_list.append(df['bg'][i])

    bg_id_list = list(set(bg_id_list))
    print('The number of the derived background knowledge principles: ', str(len(bg_id_list)))
    print('Waiting for the predictions...')
    # prediction results are saved in pred_dict
    pred_dict = {} 
    
    for test_pid in ex_table['test']:
        print(test_pid)
        pred_dict[test_pid] = []

        with open('./sentences/'+test_pid+'.lp', 'rt') as file:
            test_representation = file.read()
        
        # compare one Winograd schema sentence with each background knowledge principle
        for bg_id in bg_id_list: 
            with open('./background_knowledge/high_level_'+bg_id+'.lp', 'rt') as file:
                bg_representation = file.read()

            whole_representation = ''.join([test_representation, bg_representation, facts_check_person, rules_semantic, rules_reasoning])
            answers = ASP(whole_representation)
            for answer in answers.by_predicate.first_arg_only:
                try:
                    pred_dict[test_pid].append(str(answer['ans']))
                    break
                except:
                    pass
                
    print(pred_dict)
                
    return pred_dict

def evaluate(experiment_type=None, pred_dict=None):
    ex_table = pd.read_csv(configuration_dict[experiment_type])
    pred_dict = pred_dict 
    pattern = re.compile('[a-zA-Z_]+')
    
    pred_list = []
    for test_pid in ex_table['test']:
        if len(pred_dict[test_pid]) == 1:
            pred_list.append(pattern.findall(pred_dict[test_pid][0][12:-3])[0][:-1])
        elif len(pred_dict[test_pid]) == 0:
            pred_list.append('no_answer')
        else: 
            pred_list.append('multiple_answers')
            
    ex_result = pd.DataFrame()
    ex_result['test_pid'] = ex_table['test']
    ex_result['test_label'] = ex_table['test_label']
    ex_result['test_pred'] = pred_list
    ex_result['correct'] = ex_result['test_label'] == ex_result['test_pred']
    
    
    # calculate accuracy
    print(experiment_type, ' accuracy: ', str(np.average(ex_result['correct'])))                                              
    
    # save the evaluation result 
    SAVE_PATH = './results/' + experiment_type + '_krr_result.csv'
    ex_result.to_csv(SAVE_PATH)
    print(experiment_type, ' results are saved to: ' + SAVE_PATH)
    print(ex_result)
                                              
    return ex_result

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--experiment_type', type=str, default=None, help='choose the type of the experiments (ex1 or ex2)', required=True)
    args = arg_parser.parse_args()    

    ex_pred = predict(experiment_type=args.experiment_type)
    evaluate(experiment_type=args.experiment_type, pred_dict = ex_pred)
        
if __name__ == '__main__':
    main()
