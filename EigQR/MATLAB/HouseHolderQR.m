%from lec notes of : https://cs.uwaterloo.ca/~y328yu/mycourses/475/lectures/lec11.pdf
function A = HouseHolderQR(A)
%output: R = upper triangle of A (w/ zeros below the main diagonal)
%
m=size(A,1); % # rows 
n=size(A,2); % # cols

    for j = 1:n
        [v,beta] = house(A(j:m,j));           
        %H=eye(m,m) - beta*vertcat(zeros(j-1,1),v)*vertcat(zeros(j-1,1),v)';
        A(j:m,j:n) = A(j:m,j:n) - beta*v*(v'*A(j:m,j:n)); %i.e. A_j = H_j*A_j-1, and on the last iteration, R=A_j
        A(j+1:m,j) = v(2:end); %for memory efficiency, this stores the non-trivial part of 'v' of this iteration in what would otherwise be zeros along the column 
        %disp(A)
    end
end

function [v,beta]=house(x)
    n = length(x);
    sig2 = norm(x(2:n))^2; %or: x(2:n)'*x(2:n)
    v=vertcat(1, x(2:n));
    x1=x(1); % A(j,j)
    
    if sig2==0
        beta=0;
    else
        if x1 <= 0 
            v1 = x1 - sqrt(x1^2 + sig2); 
        else
            v1 = - sig2/(x1 + sqrt(x1^2 + sig2) );
        end
        beta = 2*v1^2/(sig2 + v1^2);
        v(2:end) = v(2:end) / v1; %this is correct - there is a typo in waterloo lec notes
    end
end

function [v, beta] = houseInv(a)
end

%testing:
%A = [1 -4; 2 3; 2 2];
%A=[4 1 -2 2; 1 2 0 1; -2 0 3 -2; 2 1 -2 -1];