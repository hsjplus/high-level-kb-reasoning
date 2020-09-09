% During the fire evacuation that was occurring at the office building, Maria thanked Christine when [she] escaped the building.
% In K-Parser, "when" was changed into "because"

has_s(occurring_7,site,building_11).
has_s(thanked_14,prep_during,evacuation_4).
has_s(escaped_18,recipient,building_20).
has_s(occurring_7,supporting_verb,was_6).
has_s(thanked_14,agent,maria_13).
has_s(evacuation_4,complement_word,fire_3).
has_s(escaped_18,agent,she_17).
has_s(thanked_14,recipient,christine_15).
has_s(occurring_7,agent,evacuation_4).
has_s(building_11,complement_word,office_10).
has_s(thanked_14,instance_of,thank).
has_s(thank,is_subclass_of,communication).
has_s(evacuation_4,instance_of,evacuation).
has_s(evacuation,is_subclass_of,act).
has_s(she_17,instance_of,she).
has_s(she,is_subclass_of,person).
has_s(christine_15,instance_of,christine).
has_s(christine,is_subclass_of,person).
has_s(escaped_18,instance_of,escape).
has_s(escape,is_subclass_of,motion).
has_s(fire_3,instance_of,fire).
has_s(fire,is_subclass_of,event).
has_s(building_11,instance_of,building).
has_s(building,is_subclass_of,artifact).
has_s(maria_13,instance_of,maria).
has_s(maria,is_subclass_of,person).
has_s(was_6,instance_of,be).
has_s(be,is_subclass_of,stative).
has_s(building_20,instance_of,building).
has_s(building,is_subclass_of,artifact).
has_s(occurring_7,instance_of,occur).
has_s(occur,is_subclass_of,change).
has_s(office_10,instance_of,office).
has_s(office,is_subclass_of,artifact).
has_s(was_6,caused_by,escaped_18).
has_s(occurring_7,caused_by,escaped_18).
has_s(thanked_14,caused_by,escaped_18).
has_s(building_20,semantic_role,place).
has_s(maria_13,semantic_role,thanker).
has_s(she_17,semantic_role,entity_escaping).
has_s(christine_15,semantic_role,thanked).

pronoun(she_17).
ans_ch1(maria_13).
ans_ch2(christine_15).