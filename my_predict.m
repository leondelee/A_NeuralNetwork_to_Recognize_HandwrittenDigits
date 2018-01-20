function rate=my_predict(theta,X,y,input_units,hidden_units,hidden_layers,output_units);
Theta=cell(hidden_layers+1,1);
X=double(X);
y=double(y);
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
[y_value,y_pre]=max(a{hidden_layers+2},[],2);
rate=0;
flag=0;
for i=1:m
    if y_pre(i)==y(i)
        flag=flag+1;
    end
end
%[y_pre y]
rate=flag/length(y)*100;
end

