#include <iostream>
#include <opencv2/opencv.hpp>
// Images exist in 3 dimensions RGB Format
// This appears to be a kernel to just handle grayscale image blurring
__global__
void imageBlur(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int BLUR_SIZE) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height){
        int pixVal = 0;
        int pixels = 0
        for i = -BLUR_SIZE; i <= BLUR_SIZE; i++){
            for j = -BLUR_SIZE; j <= BLUR_SIZE; j++){
                int newCol = col + i;
                int newRow = row + j;
                if (newCol >= 0 && newCol < width && newRow >= 0 && newRow < height){
                    pixVal += inputImage[newRow * width + newCol];
                    pixels++;
                }
            }
        }
        outputImage[row * width + col] = pixVal / pixels;
    }


}