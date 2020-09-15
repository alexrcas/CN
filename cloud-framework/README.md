# Cloudy: framework para ejecución de programas paralelos en la nube
* Alexis Rodríguez Casañas

## Descripción de la práctica:
El objetivo del siguiente trabajo es lograr un pequeño framework que permita ejecutar programas en la nube, es decir, el cliente no debe utilizar sus recursos para la computación sino que esta debe ocurrir de forma remota y devolver el resultado a este.

## Requisitos
* El framework debe incorporar mecanismos de seguridad que permitan que solo sea utilizado por clientes registrados.
* Debe proporcionar una API que permita a cualquier desarrollador crear su propio servicio.
* Debe poder utilizarse de forma genérica

## Arquitectura y tecnologías
### Arquitectura
Para esta práctica, se ha optado por una arquitectura basada en microservicios, donde se han dividido las diferentes funciones en distintos servicios en ejecución. El servicio *File System* es el encargado de ejecutar los programas y devolver el resultado de su ejecución a través de la API. Esta API puede ser llamada desde dos servicios distintos:
* Web server: es un servidor web común que ofrece un front-end para utilizar el servicio. Solo usuarios registrados pueden acceder a él mediante una pantalla de *Log in*.
* External API: desde este servicio es posible utilizar perfectamente Cloudy solamente a base de llamadas REST. En realidad, es una API exactamente igual que la que posee el *web server*, solo que ha decidido separarse por motivos de limpieza y seguridad. Esto permite que esté totalmente desacoplada y que sea posible añadir o modificar funcionalidades sin que haya que modificar código en el resto de servicios.
![](https://i.ibb.co/KbnYrBf/image.png)

### Tecnologías
#### Back-end
El back-end está hecho en *Python* utilizando el framework *Flask*. Es un framework injustamente poco conocido pero que ha ido ganando adeptos con el paso del tiempo, ya que permite crear servicios REST con mucha agilidad.
#### Persistencia de datos
Como base de datos, se ha utilizado una de tipo NoSQL (MongoDB) que se encuentra en la nube (Atlas). 
#### Front-end
Además del necesario HTML, JS y CSS, se han utilizado las siguientes tecnologías:
* Jinja2 como motor de plantillas.
* Boostrap como framework de CSS

El uso de plantillas ha permitido un código del lado del cliente muy limpio, modular y sin necesidad prácticamente de Javascript.

## ¿Cómo funciona?
El usuario debe subir un programa ejecutable que ha creado, acompañado de un fichero de configuración *json* con unas reglas adecuadas. Este fichero *json* será utilizado por el sistema para crear dinámicamente el formulario de uso del programa. Por ejemplo, un fichero *json* como el siguiente:
```
{
	"name": filter,
	"inputs": [
		{"name": "imagen de entrada", "type": "file"},
		{"name": "tamaño de filtro", "type": "text"}
	]
}
```
generaría un formulario con dos campos, uno para subir un fichero y otro para introducir un número que indicaría el tamaño del filtro.

## Análisis de costes en Amazon
Para desplegar el servicio anterior en la nube, necesitaríamos:
* Una máquina para el servicio de API.
* Una máquina para el servidor web.
* Una máquina para el servicio Filesystem.
* Una máquina para la base de datos.

En realidad, lo correcto sería almacenar los programas, ficheros y demás archivos estáticos de los usuarios en un servidor FTP, por lo que también sería adecuado contar con una máquina destinada al almacenamiento.

### Base de datos
Como se puede ver en las siguientes imágenes, el coste de una base de datos viene determinado por horas de instancia activa (primera imagen) y operaciones de E/S (segunda imagen). En nuestro caso, nuestra base de datos hará relativamente pocas operaciones y de complejidad reducida, ya que se limitará a autenticar a los usuarios y almacenar las horas de cómputo que han empleado o los créditos restantes disponibles. Esto nos permite elegir una opción muy económica como podría ser la primera o segunda de la tabla. En cualquier caso, si nos equivocamos en la estimación, el servicio puede reescalarse rápidamente, lo cual es precisamente una de las ventajas de delegar nuestra infraestructura en la nube.

![](https://i.ibb.co/PMcvxjH/image.png)

![](https://i.ibb.co/B3VFbhn/image.png)

### Almacenamiento
Para el almacenamiento, debería ser suficiente con la primera opción, ya que los usuarios descargarán las imágenes o resultados arrojados a su equipo local y la infraestructura únicamente necesita un pequeño espacio temporal.
![](https://i.ibb.co/TtrQdF8/image.png)

### Computación
Para el cómputo, podría elegirse la segunda opción para el servidor web y la API, y la cuarta opción para el servidor de cómputo, ya que es quien necesitará una cantidad significativa de núcleos para realizar su trabajo.
![](https://i.ibb.co/Yjx81xh/image.png)

A continuación se desgolsa una estimación mensual del coste calculando a la baja:
* Servicio de base de datos: 194,4€
* Almacenamiento (50GB): 1,15€
* Servidor web y API (se les supone 12h de CPU diarias): 36,72€

## Conclusiones
Personalmente, he encontrado la práctica muy interesante para conocer los principios básicos de computación en la nube. Cuando se implementa un servicio de estas características es necesario tener en cuenta muchas variables relativas a la seguridad y la administración de sistemas que normalmente se pasan por alto al no ser necesarias en otro tipo de sistemas y queda patente que un servicio de estas características no es un programa informático tradicional.
Aún así, el potencial de estos servicios es enorme como hemos visto durante la asignatura y como estamos viendo en el mundo real, donde este tipo de sistemas están siendo la tendencia y sin duda, parece que han venido para quedarse.


