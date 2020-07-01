/*
Alexis Rodríguez Casañas.
Versión CUDA de un algoritmo de desenfoque muy simple.
Compilar con:
nvcc filter.cu `pkg-config --libs opencv` --expt-relaxed-constexpr
Ejecutar con:
./a.out imagen.jpg

*/


#include <cstdio>
#include <memory>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>

#define KERNEL_SIZE 5



__global__ void applyFilter(uchar* source, uchar* result, int rows, int columns)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int numberOfColumns = columns;
    int numberOfRows = rows;

    if (i < numberOfRows)
        for (int j = 0; j < numberOfColumns; j++) 
        {
            float val = 0, sum = 0;
            for (int t = i - KERNEL_SIZE; t < i + KERNEL_SIZE + 1; t++)
                for (int s = j - KERNEL_SIZE; s < j + KERNEL_SIZE + 1; s++) {
                    if ((s >= 0) && (t >= 0)) {
                        int x = cv::min(numberOfColumns - 1, cv::max(0, s));
                        int y = cv::min(numberOfRows - 1, cv::max(0, t));
                        float weight = 1;
                        
                        val += source[y * numberOfColumns + x] * weight;
                        sum += weight;
                    }
                }
                result[i * numberOfColumns + j] = round(val / sum);
        }
}



int main(int argc, char** argv)
{
    std::shared_ptr<cv::Mat> image(new cv::Mat(cv::imread(argv[1], cv::IMREAD_GRAYSCALE)));
    std::shared_ptr<cv::Mat> result(new cv::Mat(image->rows, image->cols, CV_8UC1));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //Necesito variables auxiliares simples porque el kernel de CUDA no entiende de OpenCV y el tipo de dato cv::mat
    //No obstante, este tipo de dato es lo que hay debajo de un cv::mat
    uchar* src;
    uchar* uresult;

    //Reservamos la memoria en el dispositivo
    cudaMalloc(&src, image->total() * sizeof(uchar));
    cudaMalloc(&uresult, image->total() * sizeof(uchar));


    //Preparamos las variables del kernel
    int blocks = 32;
    dim3 threadsPerBlock(image->rows / blocks);

    /*
    Con todo listo, nos llevamos los datos al dispositivo, computamos y los traemos de vuelta.
    En mi opinión hay que contar en la medición de tiempo el cómputo pero también el traslado de datos en ambas direcciones, ya que es una penalización
    que hay que tener muy presente a la hora de decidir si computar en la GPU y es una consecuencia única de esta decisión, por lo que no sería justo
    no sumar alguna de estas medidas.
    */
    auto startTime = std::chrono::high_resolution_clock::now();
    cudaMemcpy(src, image->data, image->total() * sizeof(uchar), cudaMemcpyHostToDevice);
    applyFilter<<<blocks, threadsPerBlock>>>(src, uresult, image->rows, image->cols);
    cudaMemcpy(result->data, uresult, image->total(), cudaMemcpyDeviceToHost);
    auto endTime = std::chrono::high_resolution_clock::now();

    cv::imwrite("result.png", *result);

    cudaFree(src);
    cudaFree(uresult);

    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Elapsed time: " << (double) elapsedTime / 1000.0 << "s" << std::endl;
}