% After slipping on the floor, Joel gratefully accepted Nelson's hand, as [he] provided help getting up.
% In K-Parser, "as" was changed into "because".

% modified: has_s(gratefully_8,complement_word,joel_7). -> has_s(accepted_9,modifier,gratefully_8).
%           has_s(accepted_9,agent,gratefully_8). -> has_s(accepted_9,agent,joel_7).
% added: has_s(accepted_9,caused_by,provided_16).

has_s(accepted_9,caused_by,provided_16).        % added
has_s(accepted_9,modifier,gratefully_8).      % modified
has_s(accepted_9,recipient,hand_12).
has_s(getting_18,verb_complement,up_19).
has_s(accepted_9,agent,joel_7).               % modified
has_s(hand_12,is_possessed_by,nelson_10).
has_s(provided_16,recipient,help_17).
has_s(accepted_9,previous_event,slipping_2).
has_s(provided_16,agent,he_15).
has_s(getting_18,instance_of,get).
has_s(get,is_subclass_of,possession).
has_s(provided_16,instance_of,provide).
has_s(provide,is_subclass_of,possession).
has_s(gratefully_8,instance_of,gratefully).
has_s(gratefully,is_subclass_of,all).
has_s(joel_7,instance_of,joel).
has_s(joel,is_subclass_of,person).
has_s(nelson_10,instance_of,nelson).
has_s(nelson,is_subclass_of,person).
has_s(slipping_2,instance_of,slip).
has_s(slip,is_subclass_of,motion).
has_s(he_15,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(floor_5,instance_of,floor).
has_s(floor,is_subclass_of,artifact).
has_s(hand_12,instance_of,hand).
has_s(hand,is_subclass_of,body).
has_s(help_17,instance_of,help).
has_s(help,is_subclass_of,act).
has_s(accepted_9,instance_of,accept).
has_s(accept,is_subclass_of,cognition).
has_s(up_19,instance_of,up).
has_s(slipping_2,next_event,provided_16).
has_s(slipping_2,next_event,getting_18).
has_s(accepted_9,next_event,provided_16).
has_s(accepted_9,next_event,getting_18).
has_s(hand_12,semantic_role,thing_accepted).
has_s(gratefully_8,semantic_role,acceptor).
has_s(help_17,semantic_role,thing_provided).
has_s(he_15,semantic_role,provider).


pronoun(he_15).
ans_ch1(joel_7).
ans_ch2(nelson_10).