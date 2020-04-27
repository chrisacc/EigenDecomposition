function [e,E] = EigParallelOrderJacobi(S)
%This version of the jacobi algorithm avoids the computationally expensive aspect of finding the max off diagonal entry on each iteration
%in the algorithm below, k,l vars are equivalent to the often used p,q vars
THRESHOLD = 1e-4;
MAX_SWEEPS = 30; 

N=size(S,1);
NOffDiag=nchoosek(N,2); %number of iterations expected per sweep
E=eye(N);
e=zeros(N,1);
X=zeros(size(S));
%Pre-calculate order of p,q pairs 

IterBlockToElem=ComputeAllChessParams(N);

sweepcnt=0;
offset=THRESHOLD+1;
while sweepcnt< MAX_SWEEPS && offset > THRESHOLD 
    iter = 0;
    while iter < N-1
        %disp(iter)
        
        for kcnt=0:N/2 -1; 
            
            %identify k,l pair
            k=IterBlockToElem(iter*N + kcnt*2 +1);
            l=IterBlockToElem(iter*N + kcnt*2 +1+1);
            %% calc cos and sin
            p=S(k,l);
            y=(S(l,l)-S(k,k))/2; %y=(e(l)-e(k))/2;
            d=abs(y) + sqrt(p^2 +y^2);
            r=sqrt(p^2 + d^2);
            c=d/r;
            s=p/r;
            t=p^2/d;
            
            if y<0
                s=-s;
                t=-t;
            end
        
%             % testing:
%             tau=(S(l,l)-S(k,k))/(2*S(k,l));
%             ttemp(1) = -tau + sqrt(1+tau^2);
%             ttemp(2) = -tau - sqrt(1+tau^2);
%             [~,minind] = min(abs(ttemp));
%             t = ttemp(minind);
%             c = 1/sqrt(1+t^2);
%             s = c*t;
            
            %% update rows and cols k,l                
            %Row update
            for j=1:N
                S=JRotate(k,j,l,j,c,s,S);%updates elements S(k,j) & S(l,j) 
            end
            
            %Col update
            for i=1:N
                S=JRotate(i,k,i,l,c,s,S);%updates elements S(i,k) & S(i,l)
                E=JRotate(i,k,i,l,c,s,E);%updates elements E(i,k) & E(i,l)
            end
        end        
        disp(S)        
        iter = iter +1; 
    end
    offset=CalcOffset(S);
    sweepcnt=sweepcnt+1;
    
    figure(1); plot(sweepcnt,offset,'*b'); xlabel('sweep'); ylabel('offset'); title('Convergence over sweeps'); hold on;
end
e=diag(S);

disp('#Sweeps')
disp(sweepcnt)

%disp('diagonalized matrix:')
%disp(S)

end %fcn

function IterBlockToElem=ComputeAllChessParams(N)
    %output has size 2*(N-1)*N/2
    IterBlockToElem=zeros(2*(N-1)*N/2,1);
    
%     %% in C++ counter notation
%     for j=0:N-1 -1%j is block ID in CUDA version
%         for i=0:N/2 -1%i is local ID in CUDA version           
%             index = (j-1)*N + (i-1)*2 + 1;  %index should be 1,3,5,7...
%             
%             index1 = mod(i + j, N-1); %second arg should be N
%             if i~=0
%                 index2 = mod(N - i + j - 1, N-1);               
%             else
%                 index2 = N-1; % this is correct
%             end            
%             row_pair(1)= nanmin(index1,index2);
%             row_pair(2)= nanmax(index1,index2);
%             row_pair
%             
% %            IterBlockToElem(index) = row_pair(1);
% %            IterBlockToElem(index + 1) = row_pair(2);
%         end
%     end
    
    %% 
    for j=0:N-1 - 1%j is block ID in CUDA version
        for i=0:N/2 -1 %i is local ID in CUDA version           
            index = (j)*N + (i)*2 + 1;  %index should be 1,3,5,7...
            
            index1 = mod(i + j, N-1); %second arg should be N
            if i~=0
                index2 = mod(N - i + j - 1, N-1);               
            else
                index2 = N-1; % this is correct
            end            
            row_pair(1)= nanmin(index1,index2);
            row_pair(2)= nanmax(index1,index2);            
            
           IterBlockToElem(index) = row_pair(1)+1;
           IterBlockToElem(index + 1) = row_pair(2)+1;
                      
        end
    end    
    
%     %% testing 
%     initPQset=1:N; 
%     for i=1:N-1;
%         if i==1
%             newset=initPQset;
%         elseif i<N-1
%             newset=newset(
%         elseif i==N-1
%             newset=
%         end
%         IterBlockToElem(1:N)=newset;
%     end
    
end

function offset=CalcOffset(S)
    offset = sum(abs(S(:))) - trace(abs(S));
end

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


%testcases:
%test8x8=[14764,3928,5720,9388,7596,11096,12888,2220,3928,11692,10412,7768,9048,6572,5292,12888,5720,10412,9644,8024,8792,7340,6572,11096,9388,7768,8024,8620,8364,8792,9048,7596,7596,9048,8792,8364,8620,8024,7768,9388,11096,6572,7340,8792,8024,9644,10412,5720,12888,5292,6572,9048,7768,10412,11692,3928,2220,12888,11096,7596,9388,5720,3928,14764];
%test8x8_2=[1.920188e+04,2.058504e+04,2.241607e+04,2.293878e+04,1.088616e+04,1.268967e+04,1.543893e+04,1.634972e+04,2.058504e+04,2.836570e+04,2.732219e+04,2.892768e+04,1.331679e+04,1.554238e+04,2.046872e+04,1.986829e+04,2.241607e+04,2.732219e+04,3.021303e+04,2.993872e+04,1.197496e+04,1.546113e+04,1.779099e+04,1.928698e+04,2.293878e+04,2.892768e+04,2.993872e+04,3.312610e+04,1.503402e+04,1.853382e+04,2.061623e+04,2.120199e+04,1.088616e+04,1.331679e+04,1.197496e+04,1.503402e+04,1.557441e+04,9.832584e+03,1.837476e+04,1.463367e+04,1.268967e+04,1.554238e+04,1.546113e+04,1.853382e+04,9.832584e+03,1.741195e+04,1.090660e+04,1.485545e+04,1.543893e+04,2.046872e+04,1.779099e+04,2.061623e+04,1.837476e+04,1.090660e+04,2.455812e+04,1.826495e+04,1.634972e+04,1.986829e+04,1.928698e+04,2.120199e+04,1.463367e+04,1.485545e+04,1.826495e+04,2.055110e+04];