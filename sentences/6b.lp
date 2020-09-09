% Eric was more thankful for the gift than Robert because the gift [he] received was less expensive.

% added: has_s(thankful_4,extent,more_3). has_s(expensive_17,extent,less_16). 
%        has_s(received_14,recipient,gift_12).has_s(gift_12,trait,expensive_17).

has_s(thankful_4,extent,more_3).      % added
has_s(expensive_17,extent,less_16).   % added
has_s(received_14,recipient,gift_12). % added
has_s(gift_12,trait,expensive_17).    % added
has_s(thankful_4,destination,gift_7).
has_s(eric_1,trait,thankful_4).
has_s(received_14,agent,he_13).
has_s(was_2,agent,eric_1).
has_s(eric_1,trait,expensive_17).
has_s(was_15,agent,eric_1).
has_s(thankful_4,instance_of,thankful).
has_s(thankful,is_subclass_of,all).
has_s(eric_1,instance_of,eric).
has_s(eric,is_subclass_of,person).
has_s(was_15,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(less_16,instance_of,less).
has_s(less,is_subclass_of,all).
has_s(he_13,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(expensive_17,instance_of,expensive).
has_s(expensive,is_subclass_of,all).
has_s(robert_9,instance_of,robert).
has_s(robert,is_subclass_of,person).
has_s(gift_7,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(received_14,instance_of,receive).
has_s(receive,is_subclass_of,possession).
has_s(more_3,instance_of,more).
has_s(more,is_subclass_of,all).
has_s(was_2,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(gift_12,instance_of,gift).
has_s(gift,is_subclass_of,possession).
has_s(was_2,caused_by,received_14).


pronoun(he_13).
ans_ch1(eric_1).
ans_ch2(robert_9).
