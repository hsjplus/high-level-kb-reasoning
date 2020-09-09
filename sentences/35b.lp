% Craig wrote a thank you note to Dennis because [he] sent him a thoughtful gift to him.
% In K-Parser, the only change is that "thank you note" is changed into "thank-you note". 

has_s(gift_14,recipient,him_16).
has_s(wrote_2,attachment,dennis_7).
has_s(note_5,trait,thank_you_4).
has_s(wrote_2,recipient,note_5).
has_s(gift_14,trait,thoughtful_13).
has_s(wrote_2,agent,craig_1).
has_s(sent_10,recipient,gift_14).
has_s(sent_10,agent,he_9).
has_s(sent_10,recipient,him_11).
has_s(gift_14,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(thoughtful_13,instance_of,thoughtful).
has_s(thoughtful,is_subclass_of,quality_descriptive).
has_s(wrote_2,instance_of,write).
has_s(write,is_subclass_of,creation).
has_s(dennis_7,instance_of,dennis).
has_s(thank_you_4,instance_of,thank).
has_s(craig_1,instance_of,craig).
has_s(craig,is_subclass_of,person).
has_s(him_11,instance_of,him).
has_s(him,is_subclass_of,person).
has_s(him_16,instance_of,him).
has_s(him,is_subclass_of,person).
has_s(he_9,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(note_5,instance_of,note).
has_s(note,is_subclass_of,communication).
has_s(sent_10,instance_of,send).
has_s(send,is_subclass_of,motion).
has_s(wrote_2,caused_by,sent_10).
has_s(him_16,semantic_role,thing_given).
has_s(gift_14,semantic_role,sent).
has_s(he_9,semantic_role,sender).
has_s(him_11,semantic_role,sent).

pronoun(he_9).
ans_ch1(craig_1).
ans_ch2(dennis_7).