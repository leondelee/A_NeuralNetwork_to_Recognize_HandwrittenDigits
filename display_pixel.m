function display_pixel(X,pixel_width)
[m,n]=size(X);
if ~exist('pixel_width','var')
    if sqrt(n)~=round(sqrt(n))
        fprintf('Please input the number of pixel of each digit');
    else
        pixel_width=round(sqrt(n));
    end
end
colormap(gray);
num_eg_each_cls=round(m/10);
show_mat=[];
for i =1:10
    show_mat_temp=[];
    for j=1:num_eg_each_cls
        new_sqr=reshape(X((i-1)*num_eg_each_cls+j,:),pixel_width,pixel_width);
        show_mat_temp=[show_mat_temp new_sqr];
    end
    show_mat=[show_mat;show_mat_temp];
end
h=imagesc(show_mat);
end
    
        
     
        
