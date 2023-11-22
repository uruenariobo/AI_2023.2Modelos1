# Utiliza la imagen oficial de Python como base
FROM python:3.10.9

# Variables de entorno para evitar la salida de texto descriptivo
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear y configurar el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar archivos y directorios necesarios al contenedor
COPY . /app

# Instalar las dependencias especificadas en requirements.txt
RUN pip install -r requirements.txt

# Expone el puerto 5000 (puerto predeterminado de Flask)
EXPOSE 5000

# Comando para ejecutar la aplicación REST
CMD ["python", "apirest.py"]
