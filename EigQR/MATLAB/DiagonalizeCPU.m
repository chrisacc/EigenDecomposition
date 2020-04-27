function [A,Q] = DiagonalizeCPU(A,Q)
%transforms the tri-diagonalized matrix T into a diagonalized matrix 
%without 'shift', this method is slow to converge
n=size(A,1);
MAX_ITER=10000;

%%
Amat=A;
D=1;
qrTol=1e-6; %tolerance for norm of off diagonal elements
m=0;
while D > qrTol && m<MAX_ITER;  
    m=m+1;
    %% v1 - using matlab's qr
    [Qmat,Rmat]=qr(Amat); %matlab function
    Amat=Rmat*Qmat;
    if m==1
        Qmat_new=Qmat;
    else
        Qmat_new=Qmat_new*Qmat;
    end
    
    %% v2 - using handwritten qr for real tri-diag
    for k=1:n-1 % 1:n-2?  
        u = zeros(n,1);        
        %%%
        x = A(k:n,k);
        x1 = x(1);        
        nu = sign(x1)*sqrt(x(1)^2 + x(2)^2); %sign(x1)*sqrt(sum(x.^2)); sign(x1)*norm(x); %*** another spot that numerically made a huge difference
        sig =  (nu + x1)/nu; %1 + x1/nu; %wow, using 1+x1/nu made a really changed the result...      
        u(k) = 1.0; 
        u(k+1) = A(k,k+1)/(nu+x1);               
        %%%  
        
        % in reduced form: (where u is n elements long)
        %u = vertcat(zeros(k-1,1),u);
        v = -sig*A*u;
        alpha = -(1/2)*sig*v'*u;
        w = v + alpha*u;
        A = A + w*u' + u*w'; 
        
        %update matrix Q (
        if k==1 && m==1
            Q = eye(n,n) - sig*(u*u');            
        else
%             p = u'* Q';
%             Q = Q' - sig*u*p;

            p = Q*u;
            Q = Q - sig*p*u';

%             Q = Q*(eye(n,n) - sig*(u*u'));
        end
    end    
    D = norm(A - diag(diag(A)));
    %A(abs(A)<qrTol)=0; %***I don't think this needs to be done but it did speed up convergence
    
    disp(m)
    disp('A: My func output:'); disp(A)
    disp('Amat: Matlab output:'); disp(Amat)
    disp('Q: My func output:'); disp(Q)
    disp('Qmat_new: Matlab output:'); disp(Qmat_new)
    %pause();
end
disp(m)
disp('My func output:'); disp(A)
disp('Matlab output:'); disp(Amat)
end

function [v,beta]=house(x) %computes the householder vector 'v'
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

function [c,s]=givens(x,z) %computes the householder vector 'v'
   
end
%test with Tin = [4 -3 0 0; -3 10/3 -5/3 0; 0 -5/3 -33/25 68/75; 0 0 68/75 149/75];