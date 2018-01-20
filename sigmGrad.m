function res=sigmGrad(z);
res=sigmoid(z).*(1-sigmoid(z));
end
