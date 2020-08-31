

x0 =load('lena.txt');
grad = @(x)cat(3, x-x([end 1:end-1],:), x-x(:,[end 1:end-1])); %gradient by first order difference
v = grad(x0);                                                  %gradient of image x0

div = @(v)v([2:end 1],:,1)-v(:,:,1) + v(:,[2:end 1],2)-v(:,:,2);
del  = div(v);

%noise

sigma = 2;
y = x0 + randn(128)*sigma;  %noisy signal                                     %noisy image

figure(5),
imagesc(x0);
colormap("gray");
title('original image');
imwrite(mat2gray(x0),'original_l.png','png');

figure,
imagesc(y);
colormap("gray");
title('noisy image');
imwrite(mat2gray(y,'noisy_l.png','png');

lambda = 1;
epsilon = 1e-3;
tau = 1.8/( 1 + lambda*8/epsilon );
tau = tau*4;

niter = 3300;
x = y;
E = [];

for i=1:niter
NormE = sqrt(epsilon^2 + sum(grad(x).^2,3));              %norm of gradient of x

Jreg = sum(sum(NormE));
f = 1/2*norm(x-y,2)^2 + lambda*Jreg;                      %cost function

Normalize = grad(x)./repmat(NormE, [1 1 2]);              %∇Jε(x)i=G∗(u)whereui=(Gx)i/||(Gx)i||ε
GradJ = -div( Normalize );
Gradf = x-y+lambda*GradJ;

E(end+1) = f;
x = x - tau*Gradf;                                       %update restored image
end

SNR = 10*log10(mean(x0.^2,'all')/mean((x-x0).^2,'all'))
SNRnoisy = 10*log10(mean(x0.^2,'all')/mean((y-x0).^2,'all'))

figure,
imagesc(x);
colormap("gray");
title('denoised image');
imwrite(mat2gray(x),'denoised_l.png','png');

figure,
h = plot(E);
title('cost func');


