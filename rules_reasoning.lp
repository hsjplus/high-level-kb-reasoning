% The rules below are written by Arpit Sharma. We only made one modification and one addition noted in comments below.
% The paper related to these rules: 
% Arpit Sharma. Using answer set programming for commonsense reasoning in the winograd schema challenge. arXiv:1907.11112[cs.AI], 2019.

% encoding

% step 1
node_G_s( X ) :- has_s(X ,R , Y ) , R != instance_of .
node_G_s( Y ) :- has_s(X ,R , Y ) , R != instance_of .
% These three Added 
node_G_s( A1 ) :- ans_ch1( A1 ).
node_G_s( A2 ) :- ans_ch2( A2 ).
node_G_s( P ) :- pronoun( P ).

edge_G_s(X ,R , Y ) :- has_s(X ,R , Y ) , R != instance_of .

% step 2
node_G_k( X ) :- has_k(X ,R , Y ) , R != instance_of .
node_G_k( Y ) :- has_k(X ,R , Y ) , R != instance_of .
edge_G_k(X ,R , Y ) :- has_k(X ,R , Y ) , R != instance_of, R != is_same_as.

% step 3
{ matches(X , Y ) : node_G_s( X ) , node_G_k( Y ) }.
:- matches(X , Y ) , matches( X1 , Y ) , X != X1 .
:- matches(X , Y ) , matches(X , Y1 ) , Y != Y1 .

k_node_matches( Y ) :- matches(X , Y ) .
:- node_G_k( Y ), not k_node_matches( Y ) .
% modified this one line below 
:- matches(X , Y ), ans_ch1(X1), ans_ch2(X2), pronoun(X3), X != X1, X != X2, X != X3, has_k(Y , instance_of, C ), not has_s(X , instance_of , C ).
:- edge_G_k( X1 ,R , Y1 ) , matches(X , X1 ) , matches(Y , Y1 ) , not edge_G_s(X ,R , Y ) .
% added this one line below
:- matches(X , Y ), has_k(Y , semantic_role, C ), not has_s(X , semantic_role , C ).

% step 4
invalid_1 :- matches(P , N1 ), matches(X , N2 ) , ans_ch1( A ) ,
pronoun( P ) , A != X , N1 != N2 ,
has_k( N1 , is_same_as  , N2 ) .

invalid_2 :- matches(P , N1 ), matches(X , N2 ) , ans_ch2( A ) ,
pronoun( P ) , A != X , N1 != N2 , has_k( N1 , is_same_as  , N2 ) .

ans( A ) :- matches(P , N1 ) , matches(A , N2 ) , ans_ch1( A ) ,
not invalid_1 , pronoun( P ) ,
has_k( N1 , is_same_as , N2 ) .

ans( A ) :- matches(P , N1 ) , matches(A , N2 ) , ans_ch2( A ) ,
not invalid_2 , pronoun( P ),
has_k( N1 , is_same_as , N2 ) .
