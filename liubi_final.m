



clc;clear;close all;
files = gunzip('E:\Data_Aneurysm\all_data\0001\mask_resample_NearestNeighbor_nb_0.3.nii.gz');

nii = load_untouch_nii(files{1});

img = nii.img;
% save img.mat;

spacingx = 0.3;
spacingy = 0.3;
spacingz = 0.3;
B2 = [];
[n1,n2,n3] = size(img);
B = [];

for m = 1:n3
a2 = img(:,:,m);

contour2 = bwperim(a2);
    for i = 1:n1
        for j = 1:n2
            if contour2(i,j) == 1
%                 plot3(i*0.3,j*0.3,m*0.3,'r.');
                %scatter3(j*spacingx,i*spacingy,m*spacingz,'filled')
%                 hold on
                B=[B;[i*0.3,j*0.3,m*0.3]];
            end
        end
    end
end

dlmwrite('a.txt',B,'delimiter',' ');
xlswrite('a.xlsx',B);


x=B(:,1);
y=B(:,2);
z=B(:,3);



% do the fitting
[ center, radii, evecs, v, chi2 ] = ellipsoid_fit( [ x y z ], '' );
fprintf( 'Ellipsoid center: %.5g %.5g %.5g\n', center );




%
for m = 1:n3
a2 = img(:,:,m);
    for i = 1:n1
        for j = 1:n2
            if a2(i,j) == 1
%                 plot3(i*spacingx,j*spacingy,m*spacingz,'r.');
%                 hold on
                B2=[B2;[i*spacingx,j*spacingy,m*spacingz]];
            end
        end
    end
end

xlswrite('b.xlsx',B2); %
x=B2(:,1)-center(1,1);
y=B2(:,2)-center(2,1);
z=B2(:,3)-center(3,1);
B3=[x y z];  %

%plot3(x,y,z,'r.');
[theta,fai,rho] = cart2sph(x,y,z);
B4=[theta,fai,rho]; 


angle1=B4(:,1)*180/pi;
angle2=B4(:,2)*180/pi;
angle3=B4(:,3);
B5=[angle1 angle2 angle3];  
B6=[];

for i = 1:size(B5,1)
    d = B5(i,1);
    if d==180
        B6(i,1)=188;
    elseif d<0
        B6(i,1) = (sign(d/36)*2+1)*18;
    else B6(i,1) = (fix(d/36)*2+1)*18;
    end
end

for i = 1:size(B5,1)
    d = B5(i,2);
    if d==90
        B6(i,2)=89;
    elseif d<0
        B6(i,2) = (sign(d/18)*2+1)*9;
    else B6(i,2) = (fix(d/18)*2+1)*9;
    end
end
%B7(:,1)=B6(:,1);
B6(:,3)=B4(:,3);   

B7 = sortrows(B6);

if (B7(1,1)==B7(2,1)) && (B7(1,2)==B7(2,2))

else B7(1,:)=[];
end
j = 2;
while j ~= size(B7,1)
    if((B7(j,1) == B7(j-1,1) )&&(B7(j,2) == B7(j-1,2)))||((B7(j,1) == B7(j+1,1) )&&(B7(j,2) == B7(j+1,2)))
    j=j+1;  
    else B7(j,:) = []; 
    end
    
end
if (B7(size(B7,1)-1,1)==B7(size(B7,1),1)) && (B7(size(B7,1)-1,2)==B7(size(B7,1),2))

else B7(size(B7,1),:)=[];
end
xlswrite('c1.xlsx',B7);

% fid1=importdata('c1.xlsx');
% t=fid1;
x=B7(:,1);
y=B7(:,2);
z=B7(:,3);
n=length(x);
j=1;
c(1,1)=x(1,1);
c(1,2)=y(1,1);
for i=1:n
    if i+1<=n
        if x(i,1)==x(i+1,1)&&y(i,1)==y(i+1,1)
            c(i+1,1)=x(i+1,1);
            c(i+1,2)=y(i+1,1);
            j=j+1;
        else
            mmm=z(i,1)-z(i-j+1,1);
            c(i-j+1,3)=mmm;
            c(i+1,1)=x(i+1,1);
            c(i+1,2)=y(i+1,1);
            j=1;
            mmm=0;
        end
    end
end
if j~=0
    mmm=z(i,1)-z(i-j+1,1);
    c(i-j+1,3)=mmm;
    j=1;
    mmm=0;
end
q=1;
for i=1:n
    if c(i,3)~=0
        D(q,1)=c(i,1);
        D(q,2)=c(i,2);
        D(q,3)=c(i,3);
        q=q+1;
    end
end


%prism
xlswrite('c2.xlsx',D);

%note
ex=1;
j=1;
while ex~=size(D,1)
    if D(ex,3)>20
        while j~=size(B7,1)
        if (B7(j,1) == D(ex,1))&&(B7(i,2) == D(ex,2))
            B7(j,:)=[];
            B3(j,:)=[];
        end
        j=j+1;
        end
        D(ex,:)=[];
        
    else ex=ex+1;
    end
    j=1;
end

%note


x = 1:size(D,1);
y = D(:,3);
maxthickness=max(D(:,3));
minthickness=min(D(:,3));
meanthickness=mean(D(:,3));
median=median(D(:,3));
chazhi=maxthickness-minthickness
ratio=maxthickness/minthickness
msd=std(D(:,3),0,1);

%

subplot(2,2,1);
scatter(x,y,10,'filled');
set(gca,'xlim',[1,size(D,1)],'xtick',[0:4:size(D,1)])
title('Scatter plot of thickness values');
ylabel('thickness/mm')
xlabel('data point')
subplot(2,2,2);    
for i=1:size(B5,1)
    for j=1:size(D,1)
        if (B6(i,1) == D(j,1))&&(B6(i,2) == D(j,2))
            plot3(B3(i,1),B3(i,2),B3(i,3),'.','Color',[0.01 D(j,3)/8 D(j,3)/8]);
            hold on
        end
    end
end

title('Color block diagram of tumor wall thickness distribution');

subplot(2,2,3);
x = 1:1:size(D,1);
y = D(:,3);
for i=1:length(D(:,3))
    h = bar(x(i),y(i));
    cdata = get(h,'YData'); 
    set(h,'FaceColor',[0.1 D(i,3)/8 D(i,3)/8],'BarWidth',1,'EdgeColor','k') 
    hold on
end
set(gca,'xlim',[1,size(D,1)],'xtick',[0:4:size(D,1)])
title('Thickness value histogram');
ylabel('thickness/mm')
xlabel('data point')
subplot(2,2,4);
histogram(D(:,3),40);
title('Thickness value histogram');
xlabel('thickness/mm')
ylabel('number')

fprintf( 'T_range: %.5g\n', chazhi );
fprintf( 'T_median: %.5f\n', median);
fprintf( 'T_max: %.5g\n' ,maxthickness);
fprintf( 'T_min: %.5f\n', minthickness);
fprintf( 'T_mean: %.5f\n', meanthickness);
fprintf( 'T_std: %.5f\n', msd);
fprintf( '\n' );
