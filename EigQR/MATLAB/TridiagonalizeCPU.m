%An attempted CPU version of the tridiagonalization in Cosnuau 2014 paper
function [A,Q] = TridiagonalizeCPU(A)
%transforms symmetric matrix A into a tri-diagonal matrix
    n=size(A,1);

    %I=eye(n,n);
    Q=eye(n,n);
    a=zeros(n,1); %main diagonal
    b=zeros(n-1,1); %1st sub and super diagonal
   
    %Househodler tridiagonalization (Golub textbook: p. 415, alg 8.3.1)
    for k=1:n-2
        [u,sig]=house(A(k+1:n,k)); % householder vector        

%             % in full
%             H=eye(n,n) - sig*vertcat(zeros(k,1),u)*vertcat(zeros(k,1),u)'; % i.e. H = I - sig*u*u'; 
%             A = H'*A*H;
%             Q = Q*H;     
        
%         % in reduced form: (where u is n-k elements long) (note, with this by the last iteration only diag, 1st sub diag and 1st sup diag elements are assigned)
%         v = -sig*A(k+1:n,k+1:n)*u;
%         alpha = -(1/2)*sig*v'*u;
%         w = v + alpha*u;        
%         A(k+1,k)=norm(A(k+1:n,k)); %set sub-diagonal entry, & then copy to super diagonal entry
%         A(k,k+1)=A(k+1,k);
%         A(k+1:n, k+1:n) = A(k+1:n, k+1:n) + w*u' + u*w';
        
        % in reduced form: (where u is n elements long)
        u = vertcat(zeros(k,1),u);
        v = -sig*A*u;        
        alpha = -(1/2)*sig*v'*u;
        w = v + alpha*u;
        A = A + w*u' + u*w';
        a(k) = A(k,k); 
        b(k) = A(k,k+1);
        
%         %updating Q' (not Q, as indicated:)
        p = u'* Q;
        Q = Q - sig*u*p;
        
        %updates Q
%         p = Q*u;
%         Q = Q - sig*p*u';        
    end
    %assign the elements of the remaining 2x2 matrix of A
    a(n-1)=A(n-1,n-1);  %a(k+1)=A(k+1,k+1); %equivalent expressions
    a(n)=A(n,n);        %a(k+2)=A(k+2,k+2);
    b(n-1)=A(n-1,n);    %b(k+1)=A(k+1,k+2);
end

function [v,beta]=house(x) %computes the householder vector 'v'
    n = length(x);
    sig2 = norm(x(2:n))^2; %sum(x(2:n).^2); % %or: x(2:n)'*x(2:n)
    v=vertcat(1, x(2:n));
    x1=x(1); % A(j,j)
    
    if sig2==0
        disp('sig2 is zero')
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

%% Alternative implementation:
% function [W,H] = hessenberg(A)
% [m,n] = size(A);
% W = zeros(m,m);
% for k = 1:m-2
%  x = A(k+1:m,k);
%  v = x;
%  v(1) = sign(x(1))*norm(x) + v(1);
%  v = v/norm(v);
%  A(k+1:m,k:m) = A(k+1:m,k:m) - 2*v*(v'*A(k+1:m,k:m));
%  A(:,k+1:m) = A(:,k+1:m) - 2*(A(:,k+1:m)*v)*v';
%  W(k+1:m,k) = v;
% end
% H = A; 

%% Test cases:
%test1: A=[4 1 -2 2; 1 2 0 1; -2 0 3 -2; 2 1 -2 -1];
% Aout= [4 -3 0 0; -3 10/3 -5/3 0; 0 -5/3 -33/25 68/75; 0 0 68/75 149/75];
%test2: A=[3 1 -4 2; 1 4 3 -1; -4 3 -2 3; 2 -1 3 2];
% Aout= [3 -4.583 0 0 ; -4.583 -4.571 -1.218 0; 0 -1.218 5.039 -0.364; 0 0 -0.364 3.532];

%Sanity checks:
%Aout = Q*Ain*Q'; %tri-diagonal matrix R should result
%Ain = Q'*Aout*Q;
%I = Q*Q' = Q'*Q;
%Ain and Aout should have the same eigenvalues

%Relating eigenvectors back:
%[Vin,Din]=eig(Ain)
%[Vout,Dout]=eig(Aout)
    %Din and Dout will be the same
    %Eigenvectors of the original matrix can be found by:
    %Vin = Q'*Vout 
    
    
    