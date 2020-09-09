% Tanya is thankful for Laura's generosity because [she] bought her a skirt to wear to the party.
% added: has_s(is_2,caused_by,bought_10). has_s(tanya_1,trait,thankful_3). has_s(is_2,agent,tanya_1).

has_s(tanya_1,trait,thankful_3). % added
has_s(is_2,caused_by,bought_10). % added
has_s(is_2,agent,tanya_1). % added
has_s(wear_15,attachment,party_18).
has_s(thankful_3,destination,generosity_7).
has_s(bought_10,recipient,her_11).
has_s(generosity_7,is_possessed_by,laura_5).
has_s(thankful_3,dependent,skirt_13).
has_s(bought_10,agent,she_9).
has_s(generosity_7,instance_of,generosity).
has_s(generosity,is_subclass_of,attribute).
has_s(bought_10,instance_of,buy).
has_s(buy,is_subclass_of,possession).
has_s(thankful_3,instance_of,thankful).
has_s(thankful,is_subclass_of,all).
has_s(party_18,instance_of,party).
has_s(party,is_subclass_of,person).
has_s(laura_5,instance_of,laura).
has_s(laura,is_subclass_of,person).
has_s(skirt_13,instance_of,skirt).
has_s(skirt,is_subclass_of,person).
has_s(wear_15,instance_of,wear).
has_s(wear,is_subclass_of,body).
has_s(she_9,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(her_11,instance_of,her).
has_s(her,is_subclass_of,person).
has_s(tanya_1,instance_of,tanya).
has_s(tanya,is_subclass_of,person).
has_s(is_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(her_11,semantic_role,thing_bought).
has_s(she_9,semantic_role,buyer).

pronoun(she_9).
ans_ch1(tanya_1).
ans_ch2(laura_5).