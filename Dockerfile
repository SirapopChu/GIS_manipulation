FROM thinkwhere/gdal-python:latest
ADD GIS_manipulation/main.py .
CMD ["python", "main.py"]
