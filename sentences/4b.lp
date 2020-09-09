% Jennifer was being taken out Sarah out to a restaurant for [she] wanted to thank the other for helping them in a difficult time.
% for better parsing, 'for' was changed into 'because' and grammatical issues were solved by adding 'by' and removing redundant 'out' 
% the changed sentence: Jennifer was being taken out by Sarah to a restaurant because [she] wanted to thank the other for helping them in a difficult time.

% modified: has_s(taken_4,caused_by,wanted_13). -> has_s(taken_out_4,caused_by,wanted_13). 

has_s(taken_out_4,attachment,restaurant_10).
has_s(helping_19,is_participant_in,time_24).
has_s(taken_out_4,supporting_verb,was_2).
has_s(taken_out_4,agent,sarah_7).
has_s(wanted_13,agent,she_12).
has_s(helping_19,recipient,them_20).
has_s(taken_out_4,passive_supporting_verb,being_3).
has_s(time_24,trait,difficult_23).
has_s(wanted_13,objective,thank_15).
has_s(thank_15,objective,helping_19).
has_s(taken_out_4,recipient,jennifer_1).
has_s(thank_15,agent,she_12).
has_s(them_20,instance_of,them).
has_s(them,is_subclass_of,group).
has_s(helping_19,instance_of,help).
has_s(help,is_subclass_of,social).
has_s(being_3,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(sarah_7,instance_of,sarah).
has_s(sarah,is_subclass_of,person).
has_s(difficult_23,instance_of,difficult).
has_s(difficult,is_subclass_of,all).
has_s(time_24,instance_of,time).
has_s(time,is_subclass_of,event).
has_s(thank_15,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(other_17,instance_of,other).
has_s(other,is_subclass_of,indefinite).
has_s(jennifer_1,instance_of,jennifer).
has_s(jennifer,is_subclass_of,person).
has_s(was_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(restaurant_10,instance_of,restaurant).
has_s(restaurant,is_subclass_of,artifact).
has_s(she_12,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(wanted_13,instance_of,want).
has_s(want,is_subclass_of,emotion).
has_s(taken_out_4,instance_of,taken_out).
has_s(was_2,caused_by,wanted_13).
has_s(being_3,caused_by,wanted_13).
has_s(taken_out_4,caused_by,wanted_13). % modified
has_s(she_12,semantic_role,wanter).
has_s(them_20,semantic_role,project).
has_s(she_12,semantic_role,thanker).


pronoun(she_12).
ans_ch1(jennifer_1).
ans_ch2(sarah_7).
