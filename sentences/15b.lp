% Joseph helped to cure Kevin's disease, so [he] gave thanks for having a good doctor.
% added: has_s(gave_11,recipient,thanks_12).

has_s(gave_11,recipient,thanks_12). % added
has_s(helped_2,dependent,so_9).
has_s(cure_4,agent,joseph_1).
has_s(gave_11,agent,he_10).
has_s(gave_11,objective,having_14).
has_s(having_14,recipient,doctor_17).
has_s(helped_2,objective,cure_4).
has_s(doctor_17,trait,good_16).
has_s(helped_2,agent,joseph_1).
has_s(cure_4,recipient,disease_7).
has_s(disease_7,is_possessed_by,kevin_5).
has_s(thanks_12,instance_of,thanks).
has_s(thanks,is_subclass_of,act).
has_s(having_14,instance_of,have).
has_s(have,is_subclass_of,possession).
has_s(good_16,instance_of,good).
has_s(good,is_subclass_of,quality_descriptive).
has_s(joseph_1,instance_of,joseph).
has_s(joseph,is_subclass_of,person).
has_s(he_10,instance_of,he).
has_s(he,is_subclass_of,person).
has_s(disease_7,instance_of,disease).
has_s(disease,is_subclass_of,state).
has_s(gave_11,instance_of,give).
has_s(give,is_subclass_of,possession).
has_s(helped_2,instance_of,help).
has_s(help,is_subclass_of,social).
has_s(kevin_5,instance_of,kevin).
has_s(kevin,is_subclass_of,person).
has_s(cure_4,instance_of,cure).
has_s(cure,is_subclass_of,body).
has_s(doctor_17,instance_of,doctor).
has_s(doctor,is_subclass_of,person).
has_s(joseph_1,semantic_role,doctor).
has_s(he_10,semantic_role,giver).
has_s(joseph_1,semantic_role,helper).
has_s(disease_7,semantic_role,patient).


pronoun(he_10).
ans_ch1(joseph_1).
ans_ch2(kevin_5).