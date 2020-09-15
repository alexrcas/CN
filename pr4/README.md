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

A continuación se detalla una tabla que recoge los resultados para la ejecución de las cuatro versiones del algoritmo con una imagen JPEG de alta resolución (1920 x 1080 píxeles)


|            | 1    | 2    | 3    | 4    |
|------------|------|------|------|------|
| Secuencial | 4    |      |      |      |
| MPI        | 3    | 1.36 | 0.90 | 0.75 |
| OpenMP     | 3    | 2    | 1    | 0.9  |
| GPU        | 0.34 |      |      |      |

![](https://i.ibb.co/Qd4KS3G/image.png)

La aceleración obtenida viene determinada por la siguiente tabla

|            | 1    | 2    | 3    | 4    |
|------------|------|------|------|------|
| MPI        | 1.3    | 2.94 | 4.44 | 5.33 |
| OpenMP     | 1.3    | 2    | 4    | 4.44  |
| GPU        | 11 |      |      |      |


A continuación se ha utilizado una imagen significativamente mayor (3000x3000 píxeles), obteniendo los siguientes resultados

|            | 1    | 2    | 3    | 4    |
|------------|------|------|------|------|
| Secuencial | 7    |      |      |      |
| MPI        | 6    | 2.21 | 1.52 | 1.35 |
| OpenMP     |  6    | 4    | 3    | 2 |
| GPU        | 0.5 |      |      |      |

![](https://i.ibb.co/LCZ09gS/image.png)

La aceleración obtenida viene determinada por la siguiente tabla

|            | 1    | 2    | 3    | 4    |
|------------|------|------|------|------|
| MPI        | 1.16    | 3,16 | 4.6 | 5.18 |
| OpenMP     | 1.16    | 1.75    | 2.33    | 3.5  |
| GPU        | 14 |      |      |      |

Como se puede observar, la mayor eficiencia se obtiene con la GPU. Esto no es una sorpresa ya que este dispositivo cuenta con unas características muy especiales tales como se vio en esta asignatura, que lo hacen tremendamente eficiente para este tipo de operaciones.

### 3. Conclusión
Esta práctica ha sido muy interesante y las conclusiones son parecidas nuevamente a las obtenidas en la práctica anterior. Si bien es cierto que el rendimiento
con CUDA es impresionante, la programación no es sencilla, por lo que antes de lanzarnos a implementar algo debemos valorar si nos compensa
el tiempo y esfuerzo invertidos. Desarrollar cuesta tiempo y dinero y sobre todo en un entorno corporativo, es muy difícil justificar que se necesitan una o dos semanas
para desarrollar algo que se tardaría un día, sobre todo si ganamos, por ejemplo, solo unos segundos en tiempo de ejecución. Al igual que MPI, 
no es una tecnología para programar por defecto durante todo un proyecto, sino que debe aplicarse con mucho criterio, en un cuello de botella puntual que sepamos 
que nos beneficiará enormemente.
