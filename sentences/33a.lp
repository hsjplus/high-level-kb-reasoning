% Amy never got around to writing a thank you note for the graduation gift Megan's got her because [she] inconsiderate.
% In K-parser, "Megan's got her because _ inconsiderate." was changed into "Megan has got her because she was inconsiderate."
% In K-parser, "thank you note" was changed into "thank-you note"
% changed sentence: Amy never got around to writing a thank-you note for the graduation gift Megan has got her because [she] was inconsiderate.

% added: has_s(writing_6,agent,amy_1). has_s(writing_6,recipient,note_9). has_s(got_16,recipient,her_17). 
% modified: has_s(got_3,agent,her_17). -> has_s(got_3,agent,amy_1).
%           has_s(got_16,agent,note_9). -> has_s(got_16,agent,megan_14).

has_s(writing_6,recipient,note_9).      % added
has_s(writing_6,agent,amy_1).           % added
has_s(note_9,destination,megan_14).
has_s(was_20,agent,she_19).
has_s(got_3,modifier,amy_1).
has_s(got_3,verb_complement,around_4).
has_s(got_16,agent,megan_14).           % modified
has_s(got_16,recipient,her_17).         % added   
has_s(got_16,supporting_verb,has_15).
has_s(megan_14,complement_word,graduation_12).
has_s(got_3,negative,never_2).
has_s(note_9,trait,thank_you_8).
has_s(got_3,dependent,inconsiderate_21).
has_s(megan_14,complement_word,gift_13).
has_s(got_3,agent,amy_1).               % modified
has_s(she_19,trait,inconsiderate_21).
has_s(she_19,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(never_2,instance_of,never).
has_s(never,is_subclass_of,all).
has_s(got_16,instance_of,get).
has_s(get,is_subclass_of,possession).
has_s(around_4,instance_of,around).
has_s(megan_14,instance_of,megan).
has_s(megan,is_subclass_of,person).
has_s(note_9,instance_of,note).
has_s(note,is_subclass_of,communication).
has_s(her_17,instance_of,her).
has_s(her,is_subclass_of,person).
has_s(inconsiderate_21,instance_of,inconsiderate).
has_s(inconsiderate,is_subclass_of,quality_descriptive).
has_s(got_3,instance_of,get).
has_s(get,is_subclass_of,possession).
has_s(thank_you_8,instance_of,thank).
has_s(amy_1,instance_of,amy).
has_s(amy,is_subclass_of,person).
has_s(has_15,instance_of,have).
has_s(have,is_subclass_of,possession).
has_s(writing_6,instance_of,write).
has_s(write,is_subclass_of,creation).
has_s(gift_13,instance_of,gift).
has_s(gift,is_subclass_of,act).
has_s(was_20,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(got_3,caused_by,was_20).
has_s(writing_6,caused_by,was_20).
has_s(has_15,caused_by,was_20).
has_s(got_16,caused_by,was_20).
has_s(note_9,semantic_role,receiver).
has_s(her_17,semantic_role,receiver).


pronoun(she_19).
ans_ch1(amy_1).
ans_ch2(megan_14).
