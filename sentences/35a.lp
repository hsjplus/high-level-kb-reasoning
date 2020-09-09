% Craig wrote a thank you note to Dennis because [he] received a thoughtful gift from him.
% In K-Parser, the only change is that "thank you note" was changed into "thank-you note"

has_s(received_10,raw_material,him_15).
has_s(wrote_2,attachment,dennis_7).
has_s(note_5,trait,thank_you_4).
has_s(received_10,recipient,gift_13).
has_s(wrote_2,recipient,note_5).
has_s(received_10,agent,he_9).
has_s(wrote_2,agent,craig_1).
has_s(gift_13,trait,thoughtful_12).
has_s(thoughtful_12,instance_of,thoughtful).
has_s(thoughtful,is_subclass_of,quality_descriptive).
has_s(wrote_2,instance_of,write).
has_s(write,is_subclass_of,creation).
has_s(dennis_7,instance_of,dennis).
has_s(thank_you_4,instance_of,thank).
has_s(received_10,instance_of,receive).
has_s(receive,is_subclass_of,possession).
has_s(craig_1,instance_of,craig).
has_s(craig,is_subclass_of,person).
has_s(him_15,instance_of,him).
has_s(him,is_subclass_of,person).
has_s(he_9,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(note_5,instance_of,note).
has_s(note,is_subclass_of,communication).
has_s(gift_13,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(wrote_2,caused_by,received_10).

pronoun(he_9).
ans_ch1(craig_1).
ans_ch2(dennis_7).