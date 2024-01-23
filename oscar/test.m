% Given dimension d
d = 8; % You can change this value as needed

% % Calculate the number of parameters needed
% num_params = (d - 1) * d / 2;
% 
% % Create the symbolic parameters dynamically
% p = sym('p', [1 num_params]); % This creates p1, p2, ..., pN
% assume(p, 'real'); % Assume that the parameters are real
% 
% % Call the function to get symbolic vectors
% vectors = get_vectors_symbolic(p);
% disp(vectors)

% % Test for linear independence using row_reduce function
% rref_matrix = row_reduce(d);
% 
% % Display the row-reduced echelon form of the matrix
% disp(rref_matrix);

is_matrix_invertible(d)
