% IF something "is preferred to" something1 because something2 is "bad" THEN something1 is same as something2

has_k(something2_,relation,causer).

has_k(something_,is_preferred_to,something1_). 
has_k(something2_,semantic_role,bad).

has_k(something1_,instance_of,something1).
has_k(something1,is_subclass_of,something).
has_k(something_,instance_of,something_instance).
has_k(something_instance,is_subclass_of,something).
has_k(something2_,instance_of,something2).
has_k(something2,is_subclass_of,something).

has_k(something1_,is_same_as,something2_).
has_k(something2_,is_same_as,something1_).
