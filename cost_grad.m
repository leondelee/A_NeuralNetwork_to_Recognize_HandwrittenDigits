function [Cost,Grad_Theta]=cost_grad(theta,input_units,hidden_units,hidden_layers,...
    output_units,X,y,lambda);
X=double(X);   %convert type
y=double(y);
Cost=0;
Theta=cell(hidden_layers+1,1);
for i=1:hidden_layers+1     %to extract the Theta(i,j) from the long vector of Theta
    if hidden_layers>1
        if i==1
            Theta{i}=reshape(theta(1:hidden_units*(input_units+1)),hidden_units,input_units+1);
        elseif i<=hidden_layers
            Theta{i}=reshape(theta(hidden_units*(input_units+1)+(i-2)*hidden_units*(hidden_units+1)+1:...
                hidden_units*(input_units+1)+(i-1)*hidden_units*(hidden_units+1)),hidden_units,hidden_units+1);
        else
            Theta{i}=reshape(theta(hidden_units*(input_units+1)+(hidden_layers-1)*hidden_units*(hidden_units+1)+1:...
                end),output_units,hidden_units+1);
        end
    else
        Theta{1}=reshape(theta(1:hidden_units*(input_units+1)),hidden_units,input_units+1);
        Theta{2}=reshape(theta(hidden_units*(input_units+1)+1:end),output_units,hidden_units+1);
    end
end
%forward propagation for calculating a 

[m,n]=size(X);
%celldisp(a);
%celldisp(z);
a{1}=X;
a{1}=[ones(m,1) a{1}];
for i=1:hidden_layers+1
    if i~=hidden_layers+1
        z{i+1}=a{i}*Theta{i}';      %input values of (i+1)th layer
        a{i+1}=sigmoid(z{i+1});
        a{i+1}=[ones(size(a{i+1},1),1) a{i+1}];
    else 
        z{i+1}=a{i}*Theta{i}';       %input values of (i+1)th layer
        a{i+1}=sigmoid(z{i+1});
    end
end
fprintf('a3')
%generate the y matrix
y_mat=zeros(m,output_units);
for i=1:output_units
    if i==10
        y_mat(:,i)=(y==10);
    else
        y_mat(:,i)=(y==i);
    end
end
%cost function
for i=1:m
    Cost_tem(i)=-1/m*(y_mat(i,:)*log(a{hidden_layers+2}(i,:))'+(1-y_mat(i,:))*log(1-a{hidden_layers+2}(i,:))');
end
Cost=sum(Cost_tem);
%regularization for cost function
for l=1:hidden_layers+1
    for i=1:size(Theta{l},1)
        for j=2:size(Theta{l},2)
            Cost=Cost+lambda/(2*m)*Theta{l}(i,j)^2;
        end
    end
end
%back propagation for calculating T
T=cell(hidden_layers+1,1);
Grad_Theta=cell(size(Theta));
%celldisp(T);
%celldisp(Grad_Theta);
theta_tem=Theta;
for k=2:hidden_layers+1
    theta_tem{k}(:,1)=[];
end
for k=1:size(Theta,1)
    Grad_Theta{k}=zeros(size(Theta{k}));
end
for i=1:m
    for l=0:hidden_layers
        if l==0
            T{hidden_layers+2-l}=a{hidden_layers+2-l}(i,:)'-y_mat(i,:)';
            Grad_Theta{1+hidden_layers-l}=Grad_Theta{1+hidden_layers-l}+T{hidden_layers+2-l}*a{1+hidden_layers-l}(i,:);
        else
            %size(theta_tem{hidden_layers+2-l}),size(T{hidden_layers+3-l}),size(z{hidden_layers+2-l})
            T{hidden_layers+2-l}=theta_tem{hidden_layers+2-l}'*T{hidden_layers+3-l}.*sigmGrad(z{hidden_layers+2-l}(i,:)');
            Grad_Theta{1+hidden_layers-l}=Grad_Theta{1+hidden_layers-l}+T{hidden_layers+2-l}*a{1+hidden_layers-l}(i,:);
        end
        
    end
            
end
sigmGrad(z{hidden_layers+2-l}(i,:));
z{hidden_layers+2-l}(1,:);
%calculate the partial derivatives of the cost function 
tem=[];
%regularization for Grad_Theta
for l=1:hidden_layers+1
    for i=1:size(Theta{l},1)
        for j=2:size(Theta{l},2)
            Grad_Theta{l}(i,j)=Grad_Theta{l}(i,j)+lambda/m*Theta{l}(i,j);
        end
    end
end
for i=1:size(Grad_Theta,1)
    tem=[tem;Grad_Theta{i}(:)];
end
Grad_Theta=1/m*tem;
end


    
    
    
    
    %for k=1:size(Grad_Theta{2+hidden_layers-l},1)
     %       for j=1:size(Grad_Theta{2+hidden_layers-l},2)
      %          Grad_Theta{2+hidden_layers-l}(i,j)
            
