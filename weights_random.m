function w=weights_random(input_units,hidden_units,hidden_layers,output_units);
epsl=sqrt(6)/sqrt(input_units+output_units);
w=[];
for i =1:hidden_layers+1
    if i==1
        temp=-epsl*rand(hidden_units,input_units+1)+2*epsl;
        w=[w;temp(:)];
    elseif i<=hidden_layers
        temp=-epsl*rand(hidden_units,hidden_units+1)+2*epsl;
        w=[w;temp(:)];
    else
        temp=-epsl*rand(output_units,hidden_units+1)+2*epsl;
        w=[w;temp(:)];
    end
end
end
    