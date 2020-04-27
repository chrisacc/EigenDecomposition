function [EigVal,Q] = DiagonalizeShiftedCPU(A,Q)
%Assumed that input matrix A is tridiagonal, Q is the accumulated
%Keep in mind that this was just written as a stepping stone towards implementing in CUDA (i.e. not optimized for Matlab)

n=size(A,1);
MAX_ITER=10000;
qrTol = 1e-9;eps; 

if nargin==1 %if the Q input is not provided, intitialize it to the identity matrix    
    Q=eye(n,n);
end

I = eye(n,n);
EigVal = zeros(n,n);

QRMethod=4; %1-uses matlabs QR function, 2-reduced form (matrix-vector mult), 3-HH method (no matrix - vec mult), 4-givens rotations (no matrix - vec mult), 
%Methods 1 and 2 are really just for understanding the algorithm
%Methods 3 and 4 are for efficient implementations (only requres input of 'a' (the diagonal of 'A') and 'b' (the 1st sub and super diagonal of 'A')

a=diag(A);
b=diag(A,1);

m=n; %initial problem size
iter = 0;
while m > 1  && iter<MAX_ITER;  
    iter = iter+1;
    %calculate the shift : %mu = A(m,m); is an easy choice but not efficient nor stable    
    
    if QRMethod==1
        %% v1 - using matlab's qr
        %Wilkinson shift:
        mu = WilkinsonShift(A(m-1,m-1), A(m,m), A(m,m-1));
        
        [Qmat,Rmat]=qr(A - mu*I); %i.e. with pseudo code notation: Am - mu*I = Qm*Rm;
        A=Rmat*Qmat + mu*I; %i.e. in psuedo code notation : A_(m+1) = Rm*Qm + mu*I; 
        %Q(:,1:m) = Q(:,1:m)*Qmat; %update Q
        Q(1:m,:) = Qmat'*Q(1:m,:); %update Qh (transpose of Q)
        
        if abs(A(m,m-1))<= qrTol*abs(A(m,m)) %deflate if true
            EigVal(m,m) = A(m,m);
            
            m=m-1;
            I = I(1:m,1:m);
            A = A(1:m,1:m);
            if m==1
                EigVal(m,m)=A(m,m);
            end
        end
        
    elseif QRMethod==2 %NB: for some reason this method is the least accurate (small qrTol may not converge)
                
        %Wilkinson shift:
        mu = WilkinsonShift(A(m-1,m-1), A(m,m), A(m,m-1));
                
        A = A - mu*I;
        for k=1:m-1;           
            u = zeros(m,1);
            x = A(k:m,k);
            x1 = x(1);            
            nu = sign(x1)*sqrt(x(1)^2 + x(2)^2);%nu = sign(x1)*sqrt(sum(x.^2));
            sig =  (nu + x1)/nu; 
            u(k) = 1.0;
            u(k+1) = A(k,k+1)/(nu+x1);
            
            v = -sig*A*u;
            alpha = -(1/2)*sig*v'*u;
            w = v + alpha*u;
            A = A + w*u' + u*w';
            
            % updating Q' (not Q, as indicated:)
            p = u'* Q(1:m,:);
            Q(1:m,:) = Q(1:m,:) - sig*u*p;
            
            %p = Q(:,1:m)*u;
            %Q(:,1:m)=Q(:,1:m) - sig*p*u';                       
        end
        A = A + mu*I; 
        
        if abs(A(m,m-1))<= qrTol*abs(A(m,m)) %deflate if true
            EigVal(m,m) = A(m,m);
            m=m-1;
            I = I(1:m,1:m);
            A = A(1:m,1:m);
            if m==1
                EigVal(m,m)=A(m,m);
            end
        end
        
    elseif QRMethod==3 % HH method with no matrix-vec mult (i.e. for 'single threaded' GPU implementation of diagonalization        
        if iter==1
            S=zeros(n,1);
        end
        ck=0; %every iter, the matrix is assumed to be tridiagonal
        
        %Wilkinson shift:        
        mu = WilkinsonShift(a(m-1), a(m), b(m-1));
        
        a = a - mu;        
        for k=1:m-1;                      
            x1=a(k);
            x2=b(k);
            
            nu = sign(x1)*sqrt(x1^2 + x2^2);
            sig = (nu + x1)/nu;
            u1 = b(k)/(nu+x1);
            
            if k==1
                u=[1 u1 0 0].';
                v=-sig*[a(k) + b(k)*u1;     b(k) + a(k+1)*u1;   b(k+1)*u1;  0];
            elseif k<m-1
                u=[0 1 u1 0].';
                v=-sig*[b(k-1) + ck*u1;     a(k) + b(k)*u1;     b(k) + a(k+1)*u1;   b(k+1)*u1];
            elseif k==m-1
                u=[0 0 1 u1].';
                v=-sig*[0;  b(k-1) + ck*u1;     a(k) + b(k)*u1;     b(k) + a(k+1)*u1];
            end
            alpha = -(1/2)*sig*(v(1)*u(1) + v(2)*u(2) + v(3)*u(3) + v(4)*u(4));
            w = v + alpha*u;
            
            if k==1
                a(k)=a(k) + w(1)*u(1) + u(1)*w(1); %leaving in this verbose form for ease of translation when the input is complex
                a(k+1)=a(k+1) + w(2)*u(2) + u(2)*w(2);
                b(k)=b(k) + w(2)*u(1) + u(2)*w(1);
                b(k+1) = b(k+1) + w(2)*u(3) + u(2)*w(3);
                ck = w(1)*u(3) + u(1)*w(3);
            elseif k<m-1
                a(k)=a(k) + w(2)*u(2) + u(2)*w(2);
                a(k+1)=a(k+1) + w(3)*u(3) + u(3)*w(3);
                b(k)=b(k) + w(3)*u(2) + u(3)*w(2);
                b(k+1) = b(k+1) + w(4)*u(3) + u(4)*w(3);
                b(k-1) = b(k-1) + w(2)*u(1) + u(2)*w(1);
                ck = w(4)*u(2) + u(4)*w(2);
            elseif k==m-1
                a(k)=a(k) + w(3)*u(3) + u(3)*w(3);
                a(k+1)=a(k+1) + w(4)*u(4) + u(4)*w(4);
                b(k)=b(k) + w(4)*u(3) + u(4)*w(3);
                b(k-1) = b(k-1) + w(3)*u(2) + u(3)*w(2);                
            end
            
            %update Q here in analogous way as on GPU:
            for iThread = 1:n
                %technically, the following is updating the conj transpose of Q, not Q
                S(iThread) = sig*(Q(k,iThread) + u1 * Q(k+1,iThread));
                Q(k, iThread) = Q(k, iThread) - S(iThread);
                Q(k+1, iThread) = Q(k+1, iThread) - u1*S(iThread);      
                
                %this updates Q
%                 S(iThread) = sig*(Q(iThread, k) + u1 * Q(iThread,k+1));
%                 Q(iThread,k) = Q(iThread, k) - S(iThread);
%                 Q(iThread,k+1) = Q(iThread, k+1) - u1*S(iThread);                
            end      
            
%             if n==m && k==1
%                 disp(Q(:,1));
%             end
        end        
        a = a + mu;
        
        if abs(b(m-1))<=qrTol*abs(a(m))
            EigVal(m,m) = a(m);
            m=m-1;
            if m==1
                EigVal(m,m) = a(m);
            end
        end
        
    elseif QRMethod==4
        %% Using givens - operating on the diagonal and subdiagonal
                
        %Wilkinson shift:
        mu = WilkinsonShift(a(m-1), a(m), b(m-1));  
       
        x= a(1) - mu;
        y= b(1); %this alg assumes that the subdiagonal 'b' is a vector of 'n-1' elements 
        for k=1:m-1;        
            if m>2
                %Gmat=givens(x,y); c=Gmat(1,1); s=Gmat(1,2);                
                [c,s]=myGivens(x,y);
            else               
                [c,s]=SymSchur(a(1), b(1), b(1), a(2));                 
            end
            w_giv = c*x - s*y;
            d = a(k) - a(k+1); 
            z = (2*c*b(k) + d*s)*s;
            a(k) = a(k) - z;
            a(k+1) = a(k+1) + z; 
            b(k) = d*c*s + (c^2 - s^2)*b(k);
            x = b(k);
            if k>1
                b(k-1)=w_giv;
            end
            if k<m-1
                y = -s*b(k+1);
                b(k+1) = c*b(k+1);
            end
            
            %Matix multiplication version
            %Q(k:k+1,1:n)=[c -s; s c]*Q(k:k+1,1:n); %update cong transpose of Q
            %Q(1:n,k:k+1)=Q(1:n,k:k+1)*[c s; -s c]; %update Q
            
            %Qupdate rewritten in preparation for GPU implementation:
            for iThread=1:n
                %technically, the following is updating the conj transpose of Q (i.e. Qh), not Q
                Qtemp1=Q(k,iThread);
                Qtemp2=Q(k+1,iThread);
                Q(k, iThread) = Qtemp1*c - Qtemp2*s;
                Q(k+1, iThread) = Qtemp1*s + Qtemp2*c;
                                
                %this updates Q
%                 Qtemp1=Q(iThread,k);
%                 Qtemp2=Q(iThread,k+1);
%                 Q(iThread,k) = Qtemp1*c - Qtemp2*s;
%                 Q(iThread,k+1) = Qtemp1*s + Qtemp2*c;                                
            end
        end
                
        %if abs(b(m-1))<= qrTol*(abs(a(m-1)) +abs(a(m)))
        if abs(b(m-1))<=qrTol*abs(a(m))
            EigVal(m,m) = a(m);
            m=m-1;           
            if m==1
                EigVal(m,m) = a(m);
            end
        end        
    
    end%method
end %iter

disp('number of global iterations required')
disp(iter);
end

function [c,s]=myGivens(a,b) %p. 216 Golub textbook
    if b==0
        c=1; s=0;
    else
        if abs(b)> abs(a)
            tau = -a/b; s=1/(sqrt(1+tau^2)); c=s*tau;
        else
            tau = -b/a; c=1/(sqrt(1+tau^2)); s=c*tau;
        end
    end
end

function [c,s]=SymSchur(app,apq,aqp,aqq)
%determine the c,s parameters that will diagonalize the input matrix [app apq; aqp app]
%i.e. [bpp bpq; bqp bqq] = [c s; -s c]'*[app apq; aqp aqq]*[c s; -s c]; % where the result is diagonal
    elem = apq;
    y = (aqq - app)*0.5;
    d = abs(y) + sqrt(elem*elem + y*y);
    r = sqrt(elem*elem + d*d);
    if (r<eps) %r<0;
        c=1.0;
        s=0;
    else
        if y~=0
            c = d/r; 
            s = y/abs(y) * elem/r;
        else
            c=1/sqrt(2);
            s=1/sqrt(2);
        end
    end
end

function mu = WilkinsonShift(a1,a2,b1)
%Inputs are the elements comprising the matrix:
%[a1 b1]
%[b1 a2]
%often written as:
%[a(n-1) b(n-1)]
%[b(n-1) a(n)  ]

%Output: the shift mu, nearest eigenvalue to d;
%mu is that eigenvalue of the tridiagonal matrix T's trailing 2-by-2 principal submatrix closer to t(n,n)

%Wilkinson shift:
d = (a1 - a2)/2;
if d==0
    mu = a2- abs(b1);
else
    mu = a2 - b1^2/(d + sign(d)*sqrt(d^2 + b1^2));
end
end
