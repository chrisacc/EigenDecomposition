function [e,E] = EigClassicalJacobi(S)
%Jacobi algorithm () for finding the eigenvalues (e) and eigenvectors (E) of matrix (S)
%Assumptions: S is a square, real and symmetric
%Warning: currently there will be a problem if input matrix is already diagonalized
%With this algorithm, the upper triangle of S is destroyed, lower triangle and the diagonal are unchanged
N=size(S,1);
MAX_ITER = 1e6;
ind0=1; %1 for matlab, 0 for C
NOffDiag=nchoosek(N,2); %number of iterations expected per sweep

%% Extraneous notes:
%on CPU, QR algorithm outperforms Jacobi, however, QR cannot be parellelized
%Basis of Jacobi algorithm:
%eigen decomposition of 2x2 symm, real matrix has exact solution
%take 2x2 blocks of overall matrix an

%% To dos:
%additional testing: DONE
%additional commenting: DONE
%output format specification (sorting of evalues option?)
%double check
%input tolerance

%% Do some initialization first
E=eye(N);
state=N; %initialize, algorithm is complete when state=0;
ind=zeros(N,1);
e=zeros(N,1);
changed=zeros(N,1);

for k=0+ind0:N-1+ind0
    ind(k) = maxind(k,S,N);
    e(k)=S(k,k);
    changed(k) = 1;
end

%% Main loop
num_iter=0;
while state ~= 0 && num_iter < MAX_ITER 
    %a 'sweep' refers to a sequence of jacobi rotations applied to all non-diagonal elements)
    %a number of iterations are required to achieve a sweep
    %Note: each iteration the upper diagonal elements belonging to rows k & l, and cols k & l will be updated
    m=0+ind0; %find index (k,l) of pivot p
    for k=1+ind0:N-1-1+ind0
        if abs(S(k,ind(k))) > abs(S(m,ind(m))) %???
            m=k;
        end
    end
    
    %determine the rotation matrix
    k=m;
    l=ind(m);
    p=S(k,l);
    y=(e(l)-e(k))/2;
    d=abs(y) + sqrt(p^2 +y^2);
    r=sqrt(p^2 + d^2);
    c=d/r;
    s=p/r;
    t=p^2/d;
    
    if y<0
        s=-s;
        t=-t;
    end
    
    S(k,l)=0.0; %k cannot equal l, the idea is to set off-diagonal elements to 0
    [e,changed,state] = JUpdate(k,-t, e, changed, state); %updates e(k)
    [e,changed,state] = JUpdate(l, t, e, changed, state); %updates e(l)
    
    %Rotate rows and cols k and l of S (in the next 3 for loops) 
%     fprintf('k,l: %d,%d \n',k,l);
    for i = 0+ind0:k-1 
%         fprintf('%d,%d \n', i,k)
%         fprintf('%d,%d \n', i,l)
        S=JRotate(i,k,i,l,c,s,S);
    end
        
    for i = k+1: l-1 %note, k and l already have ind0 offset taken into account
%         fprintf('%d,%d \n', k,i)
%         fprintf('%d,%d \n', i,l)
        S=JRotate(k,i,i,l,c,s,S);
    end
    
    for i= l+1: N-1+ind0
        S=JRotate(k,i,l,i,c,s,S);
    end
    
    %Rotate eigenvectors
    for i=0+ind0:N-1+ind0
        E=JRotate(i,k,i,l,c,s,E); %updates elements E(i,k) & E(i,l)
    end
    ind(k) = maxind(k,S,N);
    ind(l) = maxind(l,S,N);
       
    fprintf('iteration: %d \n', num_iter);
    %fprintf('k,l: %d,%d \n',k,l);
    %disp('End iter S matrix:'); disp(S);
    num_iter = num_iter+1;    
end %end while

approxsweeps=round(num_iter/NOffDiag);
fprintf('Convergence took approximately %d sweeps \n', approxsweeps)
end

%%%%%%%%%%%%%%%%%%
function [A]=JRotate(k,l,i,j,c,s,A)
    %***add to this function linear indexing 
    
    %alters elements A(k,l) and A(i,j) with the rotation matrix defined with parameters 'c' and 's'
    %[A_kl;A_ij] = [c -s; s c]*[A_kl; A_ij];
    
    %multi-dimensional indexing:
    temp1 = c*A(k,l) - s*A(i,j);
    temp2 = s*A(k,l) + c*A(i,j);
    A(k,l) = temp1;
    A(i,j) = temp2;
    
    %linear indexing version:    
end

function [e, changed, state]=JUpdate(k, t, e, changed, state)
%update e(k) and its status
    y = e(k); %prev_ek
    e(k) = y + t; %updated e(k)
    
    if (changed(k) == 1 && y == e(k))
        changed(k) = 0;
        state = state - 1;
    elseif (changed(k) ~= 1 && y ~= e(k))
        changed(k) = 1;
        state = state + 1;
    end    
end

function [m] = maxind(k,S,N)
%index of largest off-diagonal element in row k
    ind0=1;
    m = k+1;
    for i = k+2 : N-1 +ind0;    
        %disp(i)
        if abs(S(k,i)) > abs(S(k,m))
            m = i; 
        end
    end
end

%%% Test with matlab:
% MTM3=[4 -30 60 -35; -30 300 -675 420; 60 -675 1620 -1050; -35 420 -1050 700];
% [e,E]=EigClassicalJacobi(MTM3);
% [V,D]=eig(MTM3); %check 1
% E*diag(e)*E'; %check 2 - should return MTM3
% invMTM3=E*diag(1./e)*E'; %check 3 - should be inverse(MTM3)
% invMTM3*MTM3

%for i=1:numel(test3c); fprintf('%d,',test3c(i)); end



%     %Testing with experimental data:
%     %Read in binary matrix
%     filename='C:\Users\Ben\Desktop\DELETE\PractiseMVBF\test4Rs.bin'; %128x128x1332
%     %format Rs into row-major.. doesn't matter actually because it's symmetric
%     fileID = fopen(filename,'r');
%     
%     Rs=fread(fileID, 'single');
%     fclose(fileID);
%     bla=reshape(Rs,[128 128 1332]);
%     %single 128x128 matrix took ~20s 
%     %test how long to decompose 1000 matrices:
% tic; 
% for i=1:1000; 
%     [V,D]=eig(bla(:,:,i)); 
% end 
% toc;