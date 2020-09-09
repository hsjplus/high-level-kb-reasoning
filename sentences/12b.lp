% Craig thanked Jason for helping them pay their monthly car bill because [he] was rich.
% modified: has_s(pay_7,caused_by,was_14). -> has_s(helping_5,caused_by,was_14).
% added: has_s(helping_5,agent,jason_3).

has_s(helping_5,agent,jason_3). % added
has_s(thanked_2,agent,craig_1).
has_s(was_14,agent,he_13).
has_s(bill_11,is_possessed_by,their_8).
has_s(bill_11,trait,monthly_9).
has_s(pay_7,recipient,bill_11).
has_s(bill_11,complement_word,car_10).
has_s(he_13,trait,rich_15).
has_s(thanked_2,objective,helping_5).
has_s(thanked_2,recipient,jason_3).
has_s(pay_7,agent,them_6).
has_s(car_10,instance_of,car).
has_s(car,is_subclass_of,artifact).
has_s(craig_1,instance_of,craig).
has_s(craig,is_subclass_of,person).
has_s(pay_7,instance_of,pay).
has_s(pay,is_subclass_of,possession).
has_s(their_8,instance_of,their).
has_s(their,is_subclass_of,group).
has_s(bill_11,instance_of,bill).
has_s(bill,is_subclass_of,artifact).
has_s(jason_3,instance_of,jason).
has_s(jason,is_subclass_of,person).
has_s(he_13,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(was_14,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(them_6,instance_of,them).
has_s(them,is_subclass_of,group).
has_s(rich_15,instance_of,rich).
has_s(rich,is_subclass_of,all).
has_s(monthly_9,instance_of,monthly).
has_s(monthly,is_subclass_of,time_descriptive).
has_s(thanked_2,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(helping_5,instance_of,help).
has_s(help,is_subclass_of,social).
has_s(helping_5,caused_by,was_14). % modified
has_s(craig_1,semantic_role,thanker).
has_s(bill_11,semantic_role,money).
has_s(jason_3,semantic_role,thanked).
has_s(them_6,semantic_role,payer).


pronoun(he_13).
ans_ch1(craig_1).
ans_ch2(jason_3).
