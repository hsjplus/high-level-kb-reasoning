% IF someone "gives_thanks_to" person1 because person2 "being_owed" THEN person1 is same as person2

has_k(person2_,relation,causer).

has_k(someone_,gives_thanks_to,person1_). 
has_k(person2_,semantic_role,being_owed).

has_k(person1_,instance_of,person1).
has_k(person1,is_subclass_of,person).
has_k(someone_,instance_of,someone).
has_k(someone,is_subclass_of,person).
has_k(person2_,instance_of,person2).
has_k(person2,is_subclass_of,person).

has_k(person1_,is_same_as,person2_).
has_k(person2_,is_same_as,person1_).
