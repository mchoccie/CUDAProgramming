//blockIdx.x refers to the block index in the grid
//blockDim.x refers to the number of threads in the block
//memcpy arguments in order: destination, source, size, direction
#include <iostream>
#include <opencv2/opencv.hpp>
__global__
void colortoGrayScaleConversion(unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    // Calculate the row and column index of the pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within the image bounds
    if (col < width && row < height) {
        // Calculate the index of the pixel in the input image
        int idx = (row * width + col) * 3; // 3 channels (RGB)

        // Get the RGB values
        unsigned char r = inputImage[idx];
        unsigned char g = inputImage[idx + 1];
        unsigned char b = inputImage[idx + 2];

        // Convert to grayscale using the luminosity method
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        // Set the output pixel to grayscale
        outputImage[row * width + col] = gray;
    }
}
 int main(){
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // Allocate memory for input and output images
    unsigned char *d_input, *d_output;
    size_t rgbSize = width * height * 3 * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, rgbSize);
    cudaMalloc(&d_output, graySize);

    // Copy data to device
    cudaMemcpy(d_input, image.data, rgbSize, cudaMemcpyHostToDevice);

    // Set up block/grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Call the kernel
    colortoGrayScaleConversion<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    std::vector<unsigned char> hostGray(width * height);
    cudaMemcpy(hostGray.data(), d_output, graySize, cudaMemcpyDeviceToHost);

    // Convert to OpenCV grayscale image and save
    cv::Mat grayImage(height, width, CV_8UC1, hostGray.data());
    cv::imwrite("output_gray.jpg", grayImage);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Grayscale image saved as output_gray.jpg" << std::endl;
    return 0;
 }