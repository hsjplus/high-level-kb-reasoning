% Patricia is known to be less gracious than Lindsey because [she] just isn't thankful for anything.

% In ASP, not has a special function, so "not" was changed to "nt"

% added: has_s(gracious_7, extent, less_6).
% modified: has_s(she_11,trait,gracious_7). -> has_s(patricia_1,trait,gracious_7).
%           has_s(be_5,agent,she_11). -> has_s(be_5,agent,patricia_1).

has_s(gracious_7, extent, less_6).         % added 
has_s(thankful_15,destination,anything_17).
has_s(known_3,passive_supporting_verb,is_2).
has_s(thankful_15,negative,not_14).
has_s(known_3,recipient,patricia_1).
has_s(she_11,trait,thankful_15).
has_s(is_13,agent,she_11).
has_s(be_5,agent,patricia_1).              % modified
has_s(patricia_1,trait,gracious_7).        % modified
has_s(be_5,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(just_12,instance_of,just).
has_s(just,is_subclass_of,all).
has_s(known_3,instance_of,know).
has_s(know,is_subclass_of,cognition).
has_s(thankful_15,instance_of,thankful).
has_s(thankful,is_subclass_of,all).
has_s(lindsey_9,instance_of,lindsey).
has_s(lindsey,is_subclass_of,person).
has_s(less_6,instance_of,less).
has_s(less,is_subclass_of,all).
has_s(patricia_1,instance_of,patricia).
has_s(patricia,is_subclass_of,person).
has_s(she_11,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(anything_17,instance_of,anything).
has_s(is_13,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(gracious_7,instance_of,gracious).
has_s(gracious,is_subclass_of,all).
has_s(not_14,instance_of,nt).
has_s(nt,is_subclass_of,all).
has_s(is_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(be_5,caused_by,is_13).
has_s(patricia_1,semantic_role,thing_known).


pronoun(she_11).
ans_ch1(patricia_1).
ans_ch2(lindsey_9).