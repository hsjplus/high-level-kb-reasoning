% Tanya loved the washer that Mary bought for her, so _ received a thank you note.
% In K-Parser, "thank you note" was changed into "thank-you note"

has_s(bought_7,destination,her_9).
has_s(loved_2,recipient,washer_4).
has_s(received_13,agent,she_12).
has_s(loved_2,agent,tanya_1).
has_s(note_16,trait,thank_you_15).
has_s(loved_2,dependent,so_11).
has_s(bought_7,agent,mary_6).
has_s(received_13,recipient,note_16).
has_s(note_16,instance_of,note).
has_s(note,is_subclass_of,communication).
has_s(received_13,instance_of,receive).
has_s(receive,is_subclass_of,possession).
has_s(her_9,instance_of,her).
has_s(her,is_subclass_of,person).
has_s(loved_2,instance_of,love).
has_s(love,is_subclass_of,emotion).
has_s(thank_you_15,instance_of,thank).
has_s(mary_6,instance_of,mary).
has_s(mary,is_subclass_of,person).
has_s(tanya_1,instance_of,tanya).
has_s(tanya,is_subclass_of,person).
has_s(she_12,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(washer_4,instance_of,washer).
has_s(washer,is_subclass_of,person).
has_s(bought_7,instance_of,buy).
has_s(buy,is_subclass_of,possession).
has_s(loved_2,causes,received_13).
has_s(bought_7,causes,received_13).
has_s(washer_4,semantic_role,loved).
has_s(tanya_1,semantic_role,lover).
has_s(mary_6,semantic_role,buyer).

pronoun(she_12).
ans_ch1(tanya_1).
ans_ch2(mary_6).