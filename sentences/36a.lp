% Justin braced themself as Benjamin put the needle through the ear for the piercing, then [he] gave thanks and left.
% modified: has_s(put_6,agent,justin_1). -> has_s(put_6,agent,benjamin_5).

has_s(put_6,destination,piercing_14).
has_s(put_6,recipient,needle_8).
has_s(left_21,agent,he_17).
has_s(braced_2,recipient,themself_3).
has_s(gave_18,agent,he_17).
has_s(gave_18,modifier,then_16).
has_s(put_6,agent,benjamin_5). % modified
has_s(gave_18,recipient,thanks_19).
has_s(left_21,instance_of,leave).
has_s(leave,is_subclass_of,location).
has_s(piercing_14,instance_of,piercing).
has_s(benjamin_5,instance_of,benjamin).
has_s(benjamin,is_subclass_of,substance).
has_s(ear_11,instance_of,ear).
has_s(ear,is_subclass_of,body).
has_s(braced_2,instance_of,brace).
has_s(brace,is_subclass_of,contact).
has_s(then_16,instance_of,then).
has_s(then,is_subclass_of,all).
has_s(needle_8,instance_of,needle).
has_s(needle,is_subclass_of,plant).
has_s(put_6,instance_of,put).
has_s(put,is_subclass_of,contact).
has_s(he_17,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(thanks_19,instance_of,thanks).
has_s(thanks,is_subclass_of,communication).
has_s(justin_1,instance_of,justin).
has_s(justin,is_subclass_of,person).
has_s(gave_18,instance_of,give).
has_s(give,is_subclass_of,possession).
has_s(themself_3,instance_of,themself).
has_s(braced_2,next_event,gave_18).
has_s(braced_2,next_event,left_21).
has_s(put_6,next_event,gave_18).
has_s(put_6,next_event,left_21).
has_s(braced_2,next_event,left_21).
has_s(put_6,next_event,left_21).
has_s(gave_18,next_event,left_21).
has_s(he_17,semantic_role,entity_leaving).
has_s(themself_3,semantic_role,patient).
has_s(he_17,semantic_role,giver).
has_s(thanks_19,semantic_role,thing_given).


pronoun(he_17).
ans_ch1(justin_1).
ans_ch2(benjamin_5).