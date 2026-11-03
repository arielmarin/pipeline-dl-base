# Pre-entrega 1: Pipeline de entrenamiento, validación y clasificador base

## Configuración del experimento
- **Versión de PyTorch:** `2.15.0+cpu`
- **Dispositivo utilizado:** `cpu`
- **Dataset:** Iris (split 80/20 estratificado, escalado solo con train)
- **Optimizador:** Adam
- **Learning rate elegido:** `0.01`
- **Arquitectura:** MLP (`4 -> 16 -> 16 -> 3`) con activaciones `ReLU`
- **Épocas:** 100

## Resultados y lectura de la curva de pérdida
- **Pérdida inicial (época 1):** Train `0.9609` | Validación `0.7992`
- **Pérdida final (época 100):** Train `0.0372` | Validación `0.0507`
- **Accuracy final en validación:** `96.67%`

![Curvas de entrenamiento](loss_curve.png)

### Interpretación del experimento
La pérdida bajó de forma consistente: en entrenamiento pasó de 0.9609 a 0.0372 y en validación de 0.7992 a 0.0507. La mayor parte del descenso ocurre en las primeras ~15 épocas; después ambas curvas se estabilizan en una meseta (~0.04 en train y ~0.05 en validación), por lo que el ciclo de entrenamiento (forward, `zero_grad`, `backward`, `step`) funciona correctamente y el modelo converge rápido en este dataset.

La pérdida de validación acompaña a la de entrenamiento durante todo el entrenamiento, con una brecha muy chica y sin tendencia ascendente, lo que indica que no hay overfitting y que el modelo generaliza a datos no vistos. El accuracy final en validación es de 96.67 % (29 de 30 muestras).

En la curva de accuracy, la validación oscila entre 96.7 % y 100 % a partir de la época ~15. Como el conjunto de validación tiene solo 30 muestras, cada ejemplo equivale a ~3.3 puntos, así que esas variaciones se deben al tamaño reducido del conjunto y no a inestabilidad del modelo. Dado que la mejora se estanca alrededor de la época 20-30, 100 épocas resultan más de las necesarias para este problema; en los próximos checkpoints se podrían explorar otros learning rates y early stopping.
