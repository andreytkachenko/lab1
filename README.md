## Задание
1. Скачать тестовые изображения и лейблы датасета MNIST с сайта http://yann.lecun.com/exdb/mnist/
2. Распаковать скаченные архивы
3. Установить зависимости `sudo pip install mlxtend` и `sudo pip install numpy`
4. Загружаем датасет:

```python
from mlxtend.data import loadlocal_mnist

X, y = loadlocal_mnist(
        images_path="<путь до>/t10k-images-idx3-ubyte", 
        labels_path="<путь до>/t10k-labels-idx1-ubyte")

# В `X` находятся изображения а в `y` значения соответственно
# `X.shape` == (10000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `y.shape` == (10000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 

```
5. Нормализуем данные изображений: значения для пикселей заданы в диапазоне 0 .. 255, а на вход сети нужно чтобы значения были 0 .. 1  
6. Загружаем сохраненное состояние (для каждого слоя из архива `params.zip`) с помощью `pickle` (https://pythonworld.ru/moduli/modul-pickle.html)
   Внимание! Данные находятся в формате списка python нужно преобразовать этот список в numpy array и сделать reshape!
   
   Формы тензоров в архиве:
   ```
   W1: (784, 16)
   W2: (16, 10)
   ```
7. Нужно написать код прямого прохождения двуслойной сети:
   ```
   - Слой 1:
       weights: `W1.pickle` из архива
       biases: `b1.pickle` из архива
       тип: линейный
       кол-во нейронов: 16
   - Слой 2:
       тип: сигмоида
   - Слой 3:
       weights: `W2.pickle` из архива
       biases: `b2.pickle` из архива
       тип: линейный
       кол-во нейронов: 10
   - Слой 4: сигмоида
   ```
8. Каждое изображение нужно пропустить через эту сеть и получить вектор с 10 предсказаниями для каждой цифры соотвественно. 
   Выбираем максимальное значение и сравниваем индекс со значением из `y`. 
   Например взяли изображение с индесом 12
   пропустили через сеть и получили вектор:
   `[0.02, 0.12, 0.69, 0.08, 0.21, 0.13, 0.0, 0.01, 0.22, 0.16]`

   находим наибольшее значение - в данном случае 0.69 находится по индексу 2
   и сравниваем со значением из `y` - `y[12] == 2` 

   2 == 2 - Угадали!

   Далее суммируем все и делим на общее кол-во (10000) и получаем точность в % (домножаем на 100)
9. Говорим мне это число до след пятницы.

