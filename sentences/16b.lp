% Because Monica gave a gift to Carrie, [she] wrote a thank you letter for the gift.
% modified: has_s(gave_3,modifier,monica_2). -> has_s(gave_3,agent,monica_2).

has_s(wrote_10,destination,gift_17).
has_s(wrote_10,agent,she_9).
has_s(letter_14,trait,thank_12).
has_s(gave_3,recipient,gift_5).
has_s(wrote_10,recipient,letter_14).
has_s(letter_14,complement_word,you_13).
has_s(gave_3,agent,monica_2). % modified
has_s(gift_5,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(gift_17,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(wrote_10,instance_of,write).
has_s(write,is_subclass_of,creation).
has_s(monica_2,instance_of,monica).
has_s(monica,is_subclass_of,person).
has_s(thank_12,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(carrie_7,instance_of,carrie).
has_s(carrie,is_subclass_of,person).
has_s(letter_14,instance_of,letter).
has_s(letter,is_subclass_of,communication).
has_s(she_9,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(you_13,instance_of,you).
has_s(you,is_subclass_of,person).
has_s(gave_3,instance_of,give).
has_s(give,is_subclass_of,possession).
has_s(wrote_10,caused_by,gave_3).
has_s(gift_5,semantic_role,thing_given).

pronoun(she_9).
ans_ch1(monica_2).
ans_ch2(carrie_7).