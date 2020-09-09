% Jennifer was taking out Sarah out to a restaurant for [she] wanted to thank the other for helping them in a difficult time.
% for better parsing, 'for' was changed into 'because' and a grammatical issue was solved by removing redundant 'out' 
% changed sentence: Jennifer was taking out Sarah to a restaurant because [she] wanted to thank the other for helping them in a difficult time.

% modified: has_s(taking_3,caused_by,wanted_11). -> has_s(taking_out_3,caused_by,wanted_11).

has_s(helping_17,is_participant_in,time_22).
has_s(taking_out_3,attachment,restaurant_8).
has_s(thank_13,agent,she_10).
has_s(taking_out_3,agent,jennifer_1).
has_s(wanted_11,agent,she_10).
has_s(time_22,trait,difficult_21).
has_s(thank_13,objective,helping_17).
has_s(taking_out_3,supporting_verb,was_2).
has_s(taking_out_3,recipient,sarah_5).
has_s(helping_17,recipient,them_18).
has_s(wanted_11,objective,thank_13).
has_s(wanted_11,instance_of,want).
has_s(want,is_subclass_of,emotion).
has_s(taking_out_3,instance_of,taking_out).
has_s(difficult_21,instance_of,difficult).
has_s(difficult,is_subclass_of,all).
has_s(helping_17,instance_of,help).
has_s(help,is_subclass_of,social).
has_s(sarah_5,instance_of,sarah).
has_s(sarah,is_subclass_of,person).
has_s(restaurant_8,instance_of,restaurant).
has_s(restaurant,is_subclass_of,artifact).
has_s(time_22,instance_of,time).
has_s(time,is_subclass_of,event).
has_s(other_15,instance_of,other).
has_s(other,is_subclass_of,indefinite).
has_s(thank_13,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(jennifer_1,instance_of,jennifer).
has_s(jennifer,is_subclass_of,person).
has_s(was_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(them_18,instance_of,them).
has_s(them,is_subclass_of,group).
has_s(she_10,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(was_2,caused_by,wanted_11).
has_s(taking_out_3,caused_by,wanted_11). % modified
has_s(she_10,semantic_role,thanker).
has_s(she_10,semantic_role,wanter).
has_s(them_18,semantic_role,project).


pronoun(she_10).
ans_ch1(jennifer_1).
ans_ch2(sarah_5).