load(fullfile('./matlab.mat')); 
showExtrinsics(stereoParams);
imageDir = fullfile('./test');
images = imageDatastore(imageDir);
I1 = readimage(images, 1);
I2 = readimage(images, 2);
[frameLeftRect, frameRightRect] = rectifyStereoImages(I1, I2, stereoParams);

imwrite(frameLeftRect,'left.jpeg');
imwrite(frameRightRect,'Right.jpeg');