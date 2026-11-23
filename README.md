# Tarea 3: Funciones, modelos personalizados y ecuaciones diferenciales

Repositorio: https://github.com/ErickMeza0102/Tareaa3

El código de los cuatro problemas se subió en un primer commit y, a partir de ahí,
cada problema tiene su propio commit en `main` con su resultado y su gráfica.
No se usaron ramas ni repositorios separados.

1. `problema1_grayscale.py`: capa personalizada que convierte RGB a escala de grises.
2. `problema2_funciones.py`: aproximación de `3·sin(πx)` y `1+2x+4x³` en `[-1,1]`.
3. `problema3_polinomio.py`: capa entrenable de polinomio cúbico ajustada a `cos(2x)`.
4. `problema4_edo.py`: soluciones numéricas de dos EDOs en `[-5,5]` vs solución analítica.

Las gráficas están en `plots/` y los reportes en `reporte.pdf` y `reporte_ia.pdf`.

## Ejecución

```bash
pip install -r requirements.txt
python problema1_grayscale.py
python problema2_funciones.py
python problema3_polinomio.py
python problema4_edo.py
```
