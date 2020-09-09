% Adam thanked Dennis for their life after [he] jumped in the water and saved them after almost drowning.
% "after" was changed into "because" in K-Parser
% changed sentence: Adam thanked Dennis for their life because [he] jumped in the water and saved them after almost drowning.

% modified: has_s(thanked_2,caused_by,jumped_9). -> has_s(thanked_2,caused_by,jumped_in_9).

has_s(jumped_in_9,is_participant_in,water_12).
has_s(thanked_2,destination,life_6).
has_s(thanked_2,agent,adam_1).
has_s(saved_14,agent,he_8).
has_s(saved_14,recipient,them_15).
has_s(life_6,is_related_to,their_5).
has_s(jumped_in_9,agent,he_8).
has_s(thanked_2,recipient,dennis_3).
has_s(adam_1,instance_of,adam).
has_s(adam,is_subclass_of,person).
has_s(water_12,instance_of,water).
has_s(water,is_subclass_of,substance).
has_s(their_5,instance_of,their).
has_s(their,is_subclass_of,group).
has_s(jumped_in_9,instance_of,jumped_in).
has_s(almost_17,instance_of,almost).
has_s(almost,is_subclass_of,all).
has_s(drowning_18,instance_of,drowning).
has_s(drowning,is_subclass_of,contact).
has_s(saved_14,instance_of,save).
has_s(save,is_subclass_of,social).
has_s(dennis_3,instance_of,dennis).
has_s(dennis,is_subclass_of,person).
has_s(he_8,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(life_6,instance_of,life).
has_s(life,is_subclass_of,person).
has_s(thanked_2,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(them_15,instance_of,them).
has_s(them,is_subclass_of,group).
has_s(thanked_2,caused_by,jumped_in_9). % modified
has_s(adam_1,semantic_role,thanker).
has_s(he_8,semantic_role,entity_saving_money).
has_s(them_15,semantic_role,money).
has_s(dennis_3,semantic_role,thanked).


pronoun(he_8).
ans_ch1(adam_1).
ans_ch2(dennis_3).