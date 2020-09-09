% Kayla was much more grateful for the gifts than Angela because [she] sent out thank you notes.
% In K-Parser, "thank you notes" is changed into "thank-you notes".
% added: has_s(grateful_5,extent,more_4).
% modified: has_s(sent_out_13,instance_of,sent_out). -> has_s(sent_out_13,instance_of,send_out).
%           has_s(was_2,caused_by,sent_13). -> has_s(was_2,caused_by,sent_out_13).

has_s(grateful_5,extent,more_4). % added
has_s(grateful_5,destination,gifts_8).
has_s(sent_out_13,agent,she_12).
has_s(notes_16,trait,thank_you_15).
has_s(sent_out_13,recipient,notes_16).
has_s(kayla_1,trait,grateful_5).
has_s(was_2,agent,kayla_1).
has_s(grateful_5,instance_of,grateful).
has_s(grateful,is_subclass_of,all).
has_s(gifts_8,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(more_4,instance_of,more).
has_s(more,is_subclass_of,all).
has_s(sent_out_13,instance_of,send_out). % modified
has_s(thank_you_15,instance_of,thank).
has_s(angela_10,instance_of,angela).
has_s(angela,is_subclass_of,person).
has_s(kayla_1,instance_of,kayla).
has_s(kayla,is_subclass_of,person).
has_s(was_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(notes_16,instance_of,note).
has_s(note,is_subclass_of,communication).
has_s(she_12,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(was_2,caused_by,sent_out_13). % modified


pronoun(she_12).
ans_ch1(kayla_1).
ans_ch2(angela_10).
