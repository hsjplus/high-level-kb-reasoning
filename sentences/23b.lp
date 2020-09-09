% I like a thank you note more than a phone call, because the [thing] seems polite.
% added: has_s(like_2,recipient,thank_you_note_4). has_s(thing_12,trait,polite_14).

has_s(like_2,recipient,thank_you_note_4). % added
has_s(thing_12,trait,polite_14). % added
has_s(like_2,modifier,more_5).
has_s(seems_13,agent,thing_12).
has_s(more_5,instance_of,more).
has_s(more,is_subclass_of,all).
has_s(i_1,instance_of,i).
has_s(i,is_subclass_of,person).
has_s(polite_14,instance_of,polite).
has_s(polite,is_subclass_of,quality_descriptive).
has_s(like_2,instance_of,like).
has_s(phone_call_8,instance_of,phone_call).
has_s(phone_call,is_subclass_of,communication).
has_s(thing_12,instance_of,thing).
has_s(thing,is_subclass_of,communication).
has_s(seems_13,instance_of,seem).
has_s(seem,is_subclass_of,perception).
has_s(thank_you_note_4,instance_of,thank_you_note).
has_s(thank_you_note,is_subclass_of,communication).
has_s(like_2,caused_by,seems_13).


pronoun(thing_12).
ans_ch1(thank_you_note_4).
ans_ch2(phone_call_8).