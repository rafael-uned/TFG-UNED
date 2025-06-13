# Trabajo de Fin de Grado - UNED

Repositorio asociado al Trabajo de Fin de Grado titulado "Diseño y modelado de haces de luz espacialmente estructurados mediante perfiles de fase" realizado por Rafael de la Fuente Herrezuelo.

## Instalación

Clona o descarga el repositorio. El código usa el lenguaje de programación Python, que se puede descargar en el siguiente enlace: 
https://www.python.org/downloads/

Es necesario instalar la librería [JAX](https://docs.jax.dev/en/latest/installation.html), que se usa para muchas operaciones matemáticas basadas en arreglos. Para ello, ejecuta en la línea de comandos usando pip: 

```
pip install -U jax
```

La herramienta pip se instala por defecto al instalar Python.

## Contenidos

Las funciones definidas en el Apéndice A del trabajo se encuentran en el archivo `funciones.py`. Gran parte de las mismas han sido adaptadas de [Diffractsim](https://github.com/rafael-fuente/diffractsim), una librería de código abierto implementada y publicada por el estudiante para cálculos numéricos de difracción.

El resto de respositorio consiste en los Jupyter Notebooks conteniendo el código para reproducir los resultados del trabajo y los archivos con los datos usados, como el logo USAF.
Un [Jupyter Notebook](https://jupyter.org/) es una aplicación web de código abierto que permite a crear y compartir documentos que incluyen código en vivo, ecuaciones y sus resultados.