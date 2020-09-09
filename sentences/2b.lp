% Kayla bought a tablet for Rachel for their birthday last Sunday, and [she] thanked them for the thoughtful gift.
% added: has_s(rachel_6,instance_of,rachel).

has_s(rachel_6,instance_of,rachel). % added
has_s(bought_2,destination,birthday_9).
has_s(thanked_15,destination,gift_20).
has_s(bought_2,recipient,tablet_4).
has_s(sunday_11,trait,last_10).
has_s(thanked_15,recipient,them_16).
has_s(gift_20,trait,thoughtful_19).
has_s(bought_2,agent,kayla_1).
has_s(birthday_9,is_possessed_by,their_8).
has_s(bought_2,time,sunday_11).
has_s(thanked_15,agent,she_14).
has_s(she_14,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(birthday_9,instance_of,birthday).
has_s(birthday,is_subclass_of,time).
has_s(last_10,instance_of,last).
has_s(last,is_subclass_of,all).
has_s(their_8,instance_of,their).
has_s(their,is_subclass_of,group).
has_s(thanked_15,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(gift_20,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(bought_2,instance_of,buy).
has_s(buy,is_subclass_of,possession).
has_s(thoughtful_19,instance_of,thoughtful).
has_s(thoughtful,is_subclass_of,quality_descriptive).
has_s(tablet_4,instance_of,tablet).
has_s(tablet,is_subclass_of,artifact).
has_s(kayla_1,instance_of,kayla).
has_s(kayla,is_subclass_of,person).
has_s(sunday_11,instance_of,sunday).
has_s(sunday,is_subclass_of,date).
has_s(them_16,instance_of,them).
has_s(them,is_subclass_of,group).
has_s(bought_2,next_event,thanked_15).
has_s(tablet_4,semantic_role,thing_bought).
has_s(them_16,semantic_role,thanked).
has_s(kayla_1,semantic_role,buyer).
has_s(she_14,semantic_role,thanker).


pronoun(she_14).
ans_ch1(kayla_1).
ans_ch2(rachel_6).