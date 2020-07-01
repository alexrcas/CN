
# Computación en la nube: Práctica 3

#### Alexis Rodríguez Casañas

Esta práctica consiste en desarrollar un algoritmo de tratamiento de imágenes y desarrollar tres versiones del mismo (estándar, openmp y mpi) comparando y analizando los diferentes resultados obtenidos.

## 1. Desarrolla un algoritmo que desarrolle un algoritmo de tratamiento de imágenes/vídeo.
Se desarrolla un algoritmo de filtrado muy simple que consiste en una ventana o kernel que se desplaza por la imagen promediando el valor de los píxeles adyacentes, consiguiendo así una imagen suavizada. Para una mayor simplicidad, el algoritmo trabaja únicamente en un canal, es decir, en escala de grises. Esto quiere decir que si se le proporciona una imagen a color la imagen resultante estará en escala de grises, pero este procesamiento es previo y no forma parte del algoritmo ni tomado en cuenta para el rendimiento.

El algoritmo ha sido encontrado en un repositorio de *github* y se ha modificado para simplificarlo y centrar así la atención en el concepto de la práctica que nos ocupa y no tanto en aspectos de algoritmia. También se han sustituido los punteros por **smart pointers** para evitar preocuparnos por la gestión de la memoria. Esto último se ha hecho a lo largo de todo el código fuente.

```
void  applyFilter(std::shared_ptr<cv::Mat> source, std::shared_ptr<cv::Mat> result) {
	int numberOfColumns = source->cols;
	int numberOfRows = source->rows;
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++) {
		
			float val = 0, sum = 0;
			for (int t = i - KERNEL_SIZE; t < i + KERNEL_SIZE + 1; t++)
				for (int s = j - KERNEL_SIZE; s < j + KERNEL_SIZE + 1; s++) {
					if ((s >= 0) && (t >= 0)) {
						int x = cv::min(numberOfColumns - 1, cv::max(0, s));
						int y = cv::min(numberOfRows - 1, cv::max(0, t));
						float weight = 1;
						val += source->data[y * numberOfColumns + x] * weight;
						sum += weight;
					}
				}
				result->data[i * numberOfColumns + j] = round(val / sum);
		}
}
```
El programa principal es muy simple:
```
int  main(int  argc, char**  argv) {
	std::shared_ptr<cv::Mat> image(new  cv::Mat(cv::imread(argv[1], cv::IMREAD_GRAYSCALE)));
	std::shared_ptr<cv::Mat> result(new  cv::Mat(image->rows, image->cols, CV_8UC1));
	
	auto startTime = std::chrono::high_resolution_clock::now();
	gaussBlur(image, result);
	auto endTime = std::chrono::high_resolution_clock::now();
	
	cv::imwrite("result.png", *result);
	
	auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
	std::cout << "Elapsed time: " << elapsedTime << "s" << std::endl;
}
```
## 2. Implementa una versión MPI para este algoritmo.
La dificultad de la versión *MPI* radica en el programa principal, ya que se deben trocear la matriz original y repartirla entre los distintos procesadores. La implementación del algoritmo de filtrado no cambia. Para simplificar la lectura, se omiten las inicializaciones de las variables típicas de *MPI* con las que empieza todo programa principal.
```
int  main(int  argc, char  **argv) {

	int itemsPerCore;
	std::shared_ptr<cv::Mat> image;
	std::shared_ptr<cv::Mat> result;
	std::shared_ptr< std::vector<uchar> > subImage;
	std::shared_ptr< std::vector<uchar> > subResult;
	
	if (rank == 0) {
		image.reset(new  cv::Mat(cv::imread(argv[1], cv::IMREAD_GRAYSCALE)));
		result.reset(new  cv::Mat(image->rows, image->cols, CV_8UC1));
		itemsPerCore = image->total() / size;
		
		for (int i = 1; i < size; i++ )
			rc = MPI_Send(&itemsPerCore, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
	}
	else
		rc = MPI_Recv(&itemsPerCore, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		
	subImage.reset(new  std::vector<uchar>(itemsPerCore));
	subResult.reset(new  std::vector<uchar>(itemsPerCore));
	
	startWtime = MPI_Wtime();
	
	if ( rank == 0)
		MPI_Scatter(image->data, itemsPerCore, MPI_BYTE, subImage->data(), itemsPerCore, MPI_BYTE, 0, MPI_COMM_WORLD);
	else
		MPI_Scatter(NULL, 0, NULL, subImage->data(), itemsPerCore, MPI_BYTE, 0, MPI_COMM_WORLD);  
		
	std::shared_ptr<cv::Mat> aux_image(new  cv::Mat(*subImage));
	std::shared_ptr<cv::Mat> aux_result(new  cv::Mat(*subResult));
	applyFilter(aux_image, aux_result);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if ( rank == 0)
		MPI_Gather(aux_result->data, itemsPerCore, MPI_BYTE, result->data, itemsPerCore, MPI_BYTE, 0, MPI_COMM_WORLD);
	else
		MPI_Gather(aux_result->data, itemsPerCore, MPI_BYTE, NULL, 0, NULL, 0, MPI_COMM_WORLD);
	endWtime = MPI_Wtime();
	
	if (rank == 0) {
		cv::imwrite("result.png", *result);
		std::cout << "Elapsed time: " << endWtime - startWtime << "s" << std::endl;
	}  
	
	rc = MPI_Finalize();
	return  0;
}
```

## 3. Desarrolla una versión OpenMP para este algoritmo.
La versión *OpenMP* es muy simple, ya que es exactamente igual que la versión estándar pero añadiendo la directiva *parallel for* en la implementación del filtro al inicio del bucle anidado, quedando como se muestra a continuación:
```
void  applyFilter(std::shared_ptr<cv::Mat> source, std::shared_ptr<cv::Mat> result) {
	int numberOfColumns = source->cols;
	int numberOfRows = source->rows;
	
#pragma  omp  parallel  for  shared(source, result, numberOfColumns, numberOfRows)
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++) {
		
			float val = 0, sum = 0;
			for (int t = i - KERNEL_SIZE; t < i + KERNEL_SIZE + 1; t++)
				for (int s = j - KERNEL_SIZE; s < j + KERNEL_SIZE + 1; s++) {
					if ((s >= 0) && (t >= 0)) {
						int x = cv::min(numberOfColumns - 1, cv::max(0, s));
						int y = cv::min(numberOfRows - 1, cv::max(0, t));
						float weight = 1;
						val += source->data[y * numberOfColumns + x] * weight;
						sum += weight;
					}
				}
				result->data[i * numberOfColumns + j] = round(val / sum);
		}
}
```

## 4. Compara ambas versiones.
Para todos los programas se ha proporcionado la misma entrada. Se trata de una imagen en formato *PNG* de tamaño 8.000 x 6.000 píxeles.

![](https://i.ibb.co/zHR3Zv4/before.png)

El resultado arrojado por el filtro implementado es el siguiente:

![](https://i.ibb.co/NtvRT6p/after.png)


Las especificaciones del equipo donde se han realizado las pruebas son las siguientes:
* Procesador: Intel dual core (2 núcleos a 2.40GHz)
* Memoria 8GB DDR3

### Versión estándar
![](https://i.ibb.co/brtK2d6/1-standard.png)
*Se observa el uso de un solo procesador a su máxima capacidad mientras el otro está prácticamente dormido*

![](https://i.ibb.co/Yp57WTW/2-standard.png)
*Tiempo de ejecución: 52 segundos*

### Versión MPI
![](https://i.ibb.co/3Nyp2rs/1-mpi.png)
*Se observa el uso de los dos núcleos*

![](https://i.ibb.co/b2rN6Dx/2-mpi.png)
*El tiempo de ejecución se reduce notablemente hasta los 19 segundos, lo cual es aproximadamente un 68% más rápido.*

### Versión OpenMP
![](https://i.ibb.co/Gp6yTFx/1-openmp.png)
*Al igual que en la versión MPI, se observa todos los núcleos disponibles trabajando.*

![](https://i.ibb.co/x8dcm5B/2-openmp.png)
*Aunque no tanto como en la versión MPI, el tiempo se reduce notablemente; prácticamente un 50%.*

## Conclusión
Ha sido muy interesante comprobar como normalmente no se utiliza toda la capacidad del hardware y el poder del procesamiento en paralelo. Las cifras son impresionantes y hablan por sí solas, más aún teniendo en cuenta que las pruebas se realizaron en un viejo equipo con un *dual core*, cuando hoy en día lo normal es contar con al menos 4 procesadores.

Como punto en contra de *MPI*, cabe destacar la relativa dificultad de la implementación. Si bien es realizable, no es algo que vayamos a hacer como norma cada vez que desarrollemos cualquier cosa, sino que debemos pensar con antelación y aplicar a un problema crítico concreto, ya que si no tenemos buenos conocimientos y práctica, seguramente tardemos mucho más e implementar la versión paralela que el propio problema en sí.

Por otro lado, me ha sorprendido enormemente *OpenMP* en cuanto a la relación coste-resultado y creo que a diferencia de *MPI* sí podría ser algo que debería venirnos automáticamente a la mente cada vez que nos enfrentemos a un fragmento de código que se preste a ello.
