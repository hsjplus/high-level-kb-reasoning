% Kayla was much more grateful for the gifts than Angela because [she] didn't send out thank you notes.
% In K-Parser, "thank you notes" was changed into "thank-you notes".
% In ASP, as "not" has a special function, "not" is changed into "nt".

% added: has_s(grateful_5,extent,more_4).

has_s(grateful_5,extent,more_4). % added
has_s(grateful_5,destination,gifts_8).
has_s(notes_18,trait,thank_you_17).
has_s(send_out_15,agent,she_12).
has_s(send_out_15,supporting_verb,did_13).
has_s(kayla_1,trait,grateful_5).
has_s(was_2,agent,kayla_1).
has_s(send_out_15,recipient,notes_18).
has_s(send_out_15,negative,not_14).
has_s(send_out_15,instance_of,send_out).
has_s(gifts_8,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(thank_you_17,instance_of,thank).
has_s(angela_10,instance_of,angela).
has_s(angela,is_subclass_of,person).
has_s(not_14,instance_of,nt).
has_s(nt,is_subclass_of,all).
has_s(grateful_5,instance_of,grateful).
has_s(grateful,is_subclass_of,all).
has_s(did_13,instance_of,do).
has_s(do,is_subclass_of,social).
has_s(more_4,instance_of,more).
has_s(more,is_subclass_of,all).
has_s(kayla_1,instance_of,kayla).
has_s(kayla,is_subclass_of,person).
has_s(was_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(notes_18,instance_of,note).
has_s(note,is_subclass_of,communication).
has_s(she_12,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(was_2,caused_by,did_13).


pronoun(she_12).
ans_ch1(kayla_1).
ans_ch2(angela_10).