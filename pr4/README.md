# Práctica 4: CUDA
Alexis Rodríguez Casañas

### 1. Desarrolla una versión en CUDA del código que has desarrollado en la práctica 3 para el procesamiento de imágenes.
```
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
```

### 2. Analiza el rendimiento de las tres versiones paralelas que has desarrollado variando el número de procesadores y el tamaño de la imagen.

### 3. Conclusión
Esta práctica ha sido muy interesante y las conclusiones son parecidas nuevamente a las obtenidas en la práctica anterior. Si bien es cierto que el rendimiento
con CUDA es impresionante, la programación no es sencilla, por lo que antes de lanzarnos a implementar algo debemos valorar si nos compensa
el tiempo y esfuerzo invertidos. Desarrollar cuesta tiempo y dinero y sobre todo en un entorno corporativo, es muy difícil justificar que se necesitan una o dos semanas
para desarrollar algo que se tardaría un día, sobre todo si ganamos, por ejemplo, solo unos segundos en tiempo de ejecución. Al igual que MPI, 
no es una tecnología para programar por defecto durante todo un proyecto, sino que debe aplicarse con mucho criterio, en un cuello de botella puntual que sepamos 
que nos beneficiará enormemente.
