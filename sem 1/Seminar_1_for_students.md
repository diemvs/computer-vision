### Теоретический блок: Библиотеки OpenCV и PIL (Pillow) в Python

#### **1. Введение в библиотеки OpenCV и PIL**

**OpenCV (Open Source Computer Vision Library):**
OpenCV — это библиотека с открытым исходным кодом, предназначенная для обработки изображений и видео, анализа движения, распознавания лиц и выполнения других задач компьютерного зрения. Она поддерживает множество языков программирования, включая Python, и предоставляет обширный набор функций для работы с изображениями и видео.

**PIL (Python Imaging Library) / Pillow:**
PIL изначально была одной из основных библиотек для обработки изображений в Python. Однако она больше не поддерживается. Вместо нее активно используется Pillow — форк PIL, который поддерживается и обновляется сообществом. Pillow предоставляет удобные функции для открытия, обработки и сохранения различных форматов изображений.

#### **2. Основные возможности OpenCV и Pillow**

| **Функциональность**         | **OpenCV**                                                                                                                                          | **Pillow**                                                                                                                                          |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Загрузка изображений**     | `cv2.imread()`                                                                                                                                       | `Image.open()`                                                                                                                                      |
| **Отображение изображений**  | `cv2.imshow()`, `cv2.waitKey()`, `cv2.destroyAllWindows()`                                                                                        | `image.show()`                                                                                                                                      |
| **Сохранение изображений**   | `cv2.imwrite()`                                                                                                                                       | `image.save()`                                                                                                                                      |
| **Преобразование цветов**    | `cv2.cvtColor()`                                                                                                                                      | `image.convert()`                                                                                                                                   |
| **Изменение размера**        | `cv2.resize()`                                                                                                                                         | `image.resize()`                                                                                                                                     |
| **Обрезка (Cropping)**       | Использование срезов NumPy-массивов                                                                                                                  | Использование метода `image.crop()`                                                                                                                  |
| **Поворот и трансформация**  | `cv2.rotate()`, `cv2.getRotationMatrix2D()`, `cv2.warpAffine()`                                                                                      | `image.rotate()`                                                                                                                                     |
| **Фильтрация изображений**   | Различные фильтры, такие как `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.bilateralFilter()`                                                         | Встроенные методы и возможности расширения через Pillow-фильтры                                                                                        |
| **Рисование на изображениях**| Функции рисования, такие как `cv2.line()`, `cv2.circle()`, `cv2.rectangle()`, `cv2.putText()`                                                           | Методы `ImageDraw` для рисования на изображениях                                                                                                     |
| **Работа с видео**           | Чтение и запись видеофайлов, обработка кадров в реальном времени                                                                                     | Ограниченные возможности работы с видео (основной фокус на статических изображениях)                                                                 |
| **Обнаружение объектов**     | Поддержка алгоритмов обнаружения и распознавания объектов, включая каскадные классификаторы, HOG, SIFT, SURF, ORB и др.                                | Отсутствует встроенная поддержка обнаружения объектов                                                                                                 |

#### **3. Установка и настройка среды разработки**

Для выполнения семинарских заданий рекомендуется использовать **Visual Studio Code (VS Code)** в сочетании с виртуальным окружением `venv`. Ниже приведены шаги по настройке среды:

##### **a. Установка Python**

1. **Скачайте и установите Python** версии 3.8 или выше с официального сайта: [python.org](https://www.python.org/downloads/).
2. **Проверьте установку**:
   ```bash
   python --version
   ```
   Должна отобразиться версия Python, например, `Python 3.8.10`.

##### **b. Установка Visual Studio Code**

1. **Скачайте и установите VS Code** с официального сайта: [code.visualstudio.com](https://code.visualstudio.com/).
2. **Установите расширение Python** для VS Code:
   - Откройте VS Code.
   - Перейдите в раздел расширений (`Ctrl + Shift + X`).
   - Найдите и установите расширение **Python** от Microsoft.

##### **c. Создание и активация виртуального окружения**

1. **Создайте папку для проекта:**
   ```bash
   mkdir cv_seminar1
   cd cv_seminar1
   ```
2. **Создайте виртуальное окружение:**
   ```bash
   python -m venv venv
   ```
3. **Активируйте виртуальное окружение:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
4. **Установите необходимые библиотеки:**
   ```bash
   pip install opencv-python pillow
   ```
5. **Настройте VS Code для использования виртуального окружения:**
   - Откройте папку проекта в VS Code (`File` -> `Open Folder` -> выберите `cv_seminar1`).
   - В нижней части окна VS Code нажмите на текущий интерпретатор Python и выберите интерпретатор из виртуального окружения `venv`.

#### **4. Основные операции с изображениями**

##### **4.1. Загрузка изображений**

**OpenCV:**
```python
import cv2

# Загрузка цветного изображения
image_cv = cv2.imread('example.jpg')

# Загрузка изображения в градациях серого
gray_image_cv = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка успешности загрузки
if image_cv is None:
    print("Ошибка: изображение не найдено или не удалось загрузить.")
else:
    print("Изображение успешно загружено.")
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')
print("Изображение успешно загружено.")
```

##### **4.2. Отображение изображений**

**OpenCV:**
```python
import cv2

# Отображение изображения
cv2.imshow('OpenCV Image', image_cv)
cv2.waitKey(0)  # Ожидание нажатия клавиши
cv2.destroyAllWindows()
```

**Pillow:**
```python
from PIL import Image

# Отображение изображения
image_pil.show()
```

##### **4.3. Сохранение изображений**

**OpenCV:**
```python
import cv2

# Сохранение цветного изображения
cv2.imwrite('saved_opencv.jpg', image_cv)

# Сохранение изображения в градациях серого
cv2.imwrite('saved_gray_opencv.jpg', gray_image_cv)
```

**Pillow:**
```python
from PIL import Image

# Сохранение изображения в формате PNG
image_pil.save('saved_pillow.png')

# Преобразование в градации серого и сохранение
gray_image_pil = image_pil.convert('L')
gray_image_pil.save('saved_gray_pillow.png')
```

##### **4.4. Изменение размера изображения**

**OpenCV:**
```python
import cv2

# Изменение размера до 256x256 пикселей
resized_cv = cv2.resize(image_cv, (256, 256))
cv2.imwrite('resized_cv.png', resized_cv)
print("Измененный размер сохранен как 'resized_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Изменение размера до 256x256 пикселей
resized_pil = image_pil.resize((256, 256))
resized_pil.save('resized_pil.png')
print("Измененный размер сохранен как 'resized_pil.png'")
```

##### **4.5. Обрезка (Cropping) изображения**

**OpenCV:**
```python
import cv2

# Обрезка области с координатами (x, y) = (50, 50) до (200, 200)
cropped_cv = image_cv[50:200, 50:200]
cv2.imwrite('cropped_cv.png', cropped_cv)
print("Центральная область сохранена как 'cropped_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Обрезка области с координатами (left, upper, right, lower) = (50, 50, 200, 200)
cropped_pil = image_pil.crop((50, 50, 200, 200))
cropped_pil.save('cropped_pil.png')
print("Центральная область сохранена как 'cropped_pil.png'")
```

##### **4.6. Поворот изображения**

**OpenCV:**
```python
import cv2

# Поворот на 90 градусов по часовой стрелке
rotated_cv = cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite('rotated_cv.png', rotated_cv)
print("Поворот сохранен как 'rotated_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Поворот на 90 градусов по часовой стрелке
rotated_pil = image_pil.rotate(-90, expand=True)  # Отрицательные значения для поворота по часовой стрелке
rotated_pil.save('rotated_pil.png')
print("Поворот сохранен как 'rotated_pil.png'")
```

##### **4.7. Изменение контраста и яркости**

**OpenCV:**
```python
import cv2

# Настройка параметров
alpha = 1.5  # Контраст (1.0-3.0)
beta = 30    # Яркость (0-100)

# Изменение контраста и яркости
adjusted_cv = cv2.convertScaleAbs(image_cv, alpha=alpha, beta=beta)
cv2.imwrite('adjusted_cv.png', adjusted_cv)
print("Измененный контраст и яркость сохранены как 'adjusted_cv.png'")
```

**Pillow:**
```python
from PIL import Image, ImageEnhance

# Изменение контраста
enhancer_contrast = ImageEnhance.Contrast(image_pil)
contrast_pil = enhancer_contrast.enhance(1.5)  # Увеличение контраста на 50%
contrast_pil.save('contrast_pil.png')
print("Измененный контраст сохранен как 'contrast_pil.png'")

# Изменение яркости
enhancer_brightness = ImageEnhance.Brightness(image_pil)
brightness_pil = enhancer_brightness.enhance(1.3)  # Увеличение яркости на 30%
brightness_pil.save('brightness_pil.png')
print("Измененная яркость сохранена как 'brightness_pil.png'")
```

##### **4.8. Работа с каналами изображения**

**Описание:**
Работа с каналами позволяет разделить изображение на отдельные цветовые компоненты (например, R, G, B) и выполнять операции на каждом канале отдельно.

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Разделение на каналы
b_channel, g_channel, r_channel = cv2.split(image_cv)

# Сохранение отдельных каналов
cv2.imwrite('blue_channel_cv.png', b_channel)
cv2.imwrite('green_channel_cv.png', g_channel)
cv2.imwrite('red_channel_cv.png', r_channel)
print("Каналы сохранены как 'blue_channel_cv.png', 'green_channel_cv.png', 'red_channel_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Разделение на каналы
r, g, b = image_pil.split()

# Сохранение отдельных каналов
r.save('red_channel_pil.png')
g.save('green_channel_pil.png')
b.save('blue_channel_pil.png')
print("Каналы сохранены как 'red_channel_pil.png', 'green_channel_pil.png', 'blue_channel_pil.png'")
```

##### **4.9. Рисование на изображениях**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Рисование линии
cv2.line(image_cv, (50, 50), (200, 200), (0, 255, 0), 3)

# Рисование прямоугольника
cv2.rectangle(image_cv, (300, 50), (450, 200), (255, 0, 0), 2)

# Рисование круга
cv2.circle(image_cv, (150, 300), 50, (0, 0, 255), -1)  # Закрашенный круг

# Добавление текста
cv2.putText(image_cv, 'OpenCV', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Сохранение измененного изображения
cv2.imwrite('drawn_opencv.jpg', image_cv)
print("Изображение с нарисованными фигурами сохранено как 'drawn_opencv.jpg'")
```

**Pillow:**
```python
from PIL import Image, ImageDraw, ImageFont

# Загрузка изображения
image_pil = Image.open('example.jpg')
draw = ImageDraw.Draw(image_pil)

# Рисование линии
draw.line((50, 50, 200, 200), fill=(0, 255, 0), width=3)

# Рисование прямоугольника
draw.rectangle([300, 50, 450, 200], outline=(255, 0, 0), width=2)

# Рисование круга (эллипса)
draw.ellipse([100, 250, 200, 350], fill=(0, 0, 255), outline=(0, 0, 0))

# Добавление текста
font = ImageFont.load_default()
draw.text((50, 400), "Pillow", fill=(255, 255, 255), font=font)

# Сохранение измененного изображения
image_pil.save('drawn_pillow.png')
print("Изображение с нарисованными фигурами сохранено как 'drawn_pillow.png'")
```

##### **4.10. Конвертация между форматами OpenCV и Pillow**

**Описание:**
Иногда необходимо конвертировать изображения между форматами OpenCV и Pillow для использования функционала обеих библиотек.

```python
import cv2
from PIL import Image
import numpy as np

# 1. Загрузка изображения с помощью OpenCV
image_cv = cv2.imread('example.jpg')

# 2. Преобразование в формат Pillow (RGB)
rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_image)

# Изменение цветового пространства в Pillow (градации серого)
gray_pil = pil_image.convert('L')
gray_pil.save('converted_gray_pil.png')
print("Изображение преобразовано и сохранено как 'converted_gray_pil.png'")

# 3. Загрузка изображения с помощью Pillow
image_pil_loaded = Image.open('converted_gray_pil.png')

# 4. Преобразование в формат OpenCV (BGR)
image_np = np.array(image_pil_loaded)
# Если изображение в градациях серого, нужно добавить канал
if len(image_np.shape) == 2:
    image_cv_converted = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
else:
    image_cv_converted = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Изменение цветового пространства в OpenCV (градации серого уже, можно преобразовать в HSV)
hsv_cv = cv2.cvtColor(image_cv_converted, cv2.COLOR_BGR2HSV)
cv2.imwrite('converted_hsv_cv.png', hsv_cv)
print("Изображение преобразовано и сохранено как 'converted_hsv_cv.png'")
```

---

### Задания для Семинара 1: Работа с изображениями

**Цель:** Закрепить теоретические знания о библиотеках OpenCV и Pillow через практические задания, освоить базовые операции загрузки, отображения и сохранения изображений.

#### **Задание 1: Установка и настройка среды**

**Описание:**
1. Установите Python версии 3.8 или выше, если он еще не установлен.
2. Установите Visual Studio Code (VS Code).
3. Создайте новую папку для семинара, например, `cv_seminar1`.
4. Откройте эту папку в VS Code.
5. Создайте виртуальное окружение `venv` и активируйте его.
6. Установите библиотеки OpenCV и Pillow внутри виртуального окружения.

**Решение:**

1. **Установка Python:**
   - Скачайте установщик с [python.org](https://www.python.org/downloads/) и следуйте инструкциям.
   - Проверьте установку:
     ```bash
     python --version
     ```
     Должна отобразиться версия Python, например, `Python 3.8.10`.

2. **Установка VS Code:**
   - Скачайте установщик с [code.visualstudio.com](https://code.visualstudio.com/) и установите.

3. **Создание папки и открытие в VS Code:**
   ```bash
   mkdir cv_seminar1
   cd cv_seminar1
   code .
   ```

4. **Создание и активация виртуального окружения:**
   - Откройте терминал в VS Code (`Ctrl + ``).
   - Выполните команды:
     ```bash
     python -m venv venv
     ```
   - **Активировать виртуальное окружение:**
     - **Windows:**
       ```bash
       venv\Scripts\activate
       ```
     - **macOS/Linux:**
       ```bash
       source venv/bin/activate
       ```

5. **Установка библиотек:**
   ```bash
   pip install opencv-python pillow
   ```

6. **Проверка установки:**
   ```bash
   pip list
   ```
   Убедитесь, что `opencv-python` и `Pillow` установлены.

---

#### **Задание 2: Загрузка и отображение изображений**

**Описание:**
1. Скачайте изображение `example.jpg` и поместите его в папку проекта `cv_seminar1`.
2. Напишите скрипт на Python для загрузки и отображения изображения с помощью OpenCV.
3. Напишите аналогичный скрипт для загрузки и отображения изображения с помощью Pillow.
4. Сравните отображение изображения в OpenCV и Pillow.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Проверка загрузки
if image_cv is None:
    print("Ошибка: изображение не найдено.")
else:
    # Отображение изображения
    cv2.imshow('OpenCV Image', image_cv)
    cv2.waitKey(0)  # Ожидание нажатия клавиши
    cv2.destroyAllWindows()
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Отображение изображения
image_pil.show()
```

**Сравнение:**
- **OpenCV** отображает изображение в окне, созданном с помощью своей встроенной функции `imshow`. Цвета могут отображаться некорректно, так как OpenCV использует формат BGR по умолчанию.
- **Pillow** использует стандартные методы операционной системы для отображения изображения, что обычно приводит к правильному отображению цветов (RGB).

---

#### **Задание 3: Сохранение изображений в разных форматах**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV и сохраните его в формате PNG.
2. Загрузите изображение `example.jpg` с помощью Pillow и сохраните его в формате JPEG.
3. Проверьте сохраненные файлы.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Сохранение в формате PNG
cv2.imwrite('saved_opencv.png', image_cv)
print("Изображение сохранено как 'saved_opencv.png'")
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Сохранение в формате JPEG
image_pil.save('saved_pillow.jpg', 'JPEG')
print("Изображение сохранено как 'saved_pillow.jpg'")
```

---

#### **Задание 4: Работа с каналами изображения**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV и Pillow.
2. Разделите изображение на отдельные цветовые каналы (R, G, B).
3. Сохраните каждый канал как отдельное изображение.
4. Объедините каналы обратно в одно изображение и сохраните результат.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Разделение на каналы
b_channel, g_channel, r_channel = cv2.split(image_cv)

# Сохранение отдельных каналов
cv2.imwrite('blue_channel_cv.png', b_channel)
cv2.imwrite('green_channel_cv.png', g_channel)
cv2.imwrite('red_channel_cv.png', r_channel)
print("Каналы сохранены как 'blue_channel_cv.png', 'green_channel_cv.png', 'red_channel_cv.png'")

# Объединение каналов обратно
merged_cv = cv2.merge((b_channel, g_channel, r_channel))
cv2.imwrite('merged_cv.png', merged_cv)
print("Объединенное изображение сохранено как 'merged_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Разделение на каналы
r, g, b = image_pil.split()

# Сохранение отдельных каналов
r.save('red_channel_pil.png')
g.save('green_channel_pil.png')
b.save('blue_channel_pil.png')
print("Каналы сохранены как 'red_channel_pil.png', 'green_channel_pil.png', 'blue_channel_pil.png'")

# Объединение каналов обратно
merged_pil = Image.merge("RGB", (r, g, b))
merged_pil.save('merged_pil.png')
print("Объединенное изображение сохранено как 'merged_pil.png'")
```

---

#### **Задание 5: Изменение размера и обрезка изображений**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV и Pillow.
2. Измените размер изображения до 300x300 пикселей.
3. Обрежьте центральную область размером 200x200 пикселей.
4. Сохраните полученные результаты.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Изменение размера
resized_cv = cv2.resize(image_cv, (300, 300))
cv2.imwrite('resized_cv.png', resized_cv)
print("Измененный размер сохранен как 'resized_cv.png'")

# Обрезка центральной области
height, width = resized_cv.shape[:2]
start_x = width//2 - 100
start_y = height//2 - 100
cropped_cv = resized_cv[start_y:start_y+200, start_x:start_x+200]
cv2.imwrite('cropped_cv.png', cropped_cv)
print("Центральная область сохранена как 'cropped_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Изменение размера
resized_pil = image_pil.resize((300, 300))
resized_pil.save('resized_pil.png')
print("Измененный размер сохранен как 'resized_pil.png'")

# Обрезка центральной области
width, height = resized_pil.size
left = width//2 - 100
upper = height//2 - 100
right = width//2 + 100
lower = height//2 + 100
cropped_pil = resized_pil.crop((left, upper, right, lower))
cropped_pil.save('cropped_pil.png')
print("Центральная область сохранена как 'cropped_pil.png'")
```

---

#### **Задание 6: Поворот изображений на произвольный угол**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV и Pillow.
2. Поверните изображение на 45 градусов по часовой стрелке.
3. Поверните изображение на 45 градусов против часовой стрелки.
4. Сохраните результаты.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Получение центра изображения
center = (image_cv.shape[1]//2, image_cv.shape[0]//2)

# Поворот на 45 градусов по часовой стрелке
matrix_clockwise = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated_clockwise_cv = cv2.warpAffine(image_cv, matrix_clockwise, (image_cv.shape[1], image_cv.shape[0]))
cv2.imwrite('rotated_clockwise_cv.png', rotated_clockwise_cv)
print("Поворот по часовой стрелке сохранен как 'rotated_clockwise_cv.png'")

# Поворот на 45 градусов против часовой стрелки
matrix_counterclockwise = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_counterclockwise_cv = cv2.warpAffine(image_cv, matrix_counterclockwise, (image_cv.shape[1], image_cv.shape[0]))
cv2.imwrite('rotated_counterclockwise_cv.png', rotated_counterclockwise_cv)
print("Поворот против часовой стрелки сохранен как 'rotated_counterclockwise_cv.png'")
```

**Pillow:**
```python
from PIL import Image

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Поворот на 45 градусов по часовой стрелке
rotated_clockwise_pil = image_pil.rotate(-45, expand=True)
rotated_clockwise_pil.save('rotated_clockwise_pil.png')
print("Поворот по часовой стрелке сохранен как 'rotated_clockwise_pil.png'")

# Поворот на 45 градусов против часовой стрелки
rotated_counterclockwise_pil = image_pil.rotate(45, expand=True)
rotated_counterclockwise_pil.save('rotated_counterclockwise_pil.png')
print("Поворот против часовой стрелки сохранен как 'rotated_counterclockwise_pil.png'")
```

---

#### **Задание 7: Изменение контраста и яркости**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV и Pillow.
2. Увеличьте контраст изображения на 50% и яркость на 30 единиц.
3. Сохраните измененные изображения.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Настройка параметров
alpha = 1.5  # Контраст (1.0-3.0)
beta = 30    # Яркость (0-100)

# Изменение контраста и яркости
adjusted_cv = cv2.convertScaleAbs(image_cv, alpha=alpha, beta=beta)
cv2.imwrite('adjusted_cv.png', adjusted_cv)
print("Измененный контраст и яркость сохранены как 'adjusted_cv.png'")
```

**Pillow:**
```python
from PIL import Image, ImageEnhance

# Загрузка изображения
image_pil = Image.open('example.jpg')

# Изменение контраста
enhancer_contrast = ImageEnhance.Contrast(image_pil)
contrast_pil = enhancer_contrast.enhance(1.5)  # Увеличение контраста на 50%
contrast_pil.save('contrast_pil.png')
print("Измененный контраст сохранен как 'contrast_pil.png'")

# Изменение яркости
enhancer_brightness = ImageEnhance.Brightness(image_pil)
brightness_pil = enhancer_brightness.enhance(1.3)  # Увеличение яркости на 30%
brightness_pil.save('brightness_pil.png')
print("Измененная яркость сохранена как 'brightness_pil.png'")
```

---

#### **Задание 8: Гистограмма изображения и её визуализация**

**Описание:**
1. Загрузите изображение `example.jpg` в градациях серого с помощью OpenCV и Pillow.
2. Вычислите гистограмму интенсивности пикселей.
3. Визуализируйте гистограмму.
4. Сохраните результаты.

**Решение:**

**OpenCV:**
```python
import cv2
import matplotlib.pyplot as plt

# Загрузка изображения в градациях серого
gray_cv = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка загрузки
if gray_cv is None:
    print("Ошибка: изображение не найдено или не удалось загрузить.")
else:
    # Вычисление гистограммы
    hist = cv2.calcHist([gray_cv], [0], None, [256], [0, 256])

    # Визуализация гистограммы
    plt.figure(figsize=(10,5))
    plt.title("Гистограмма OpenCV")
    plt.xlabel("Интенсивность пикселя")
    plt.ylabel("Количество пикселей")
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.savefig('histogram_cv.png')
    plt.show()
    print("Гистограмма сохранена как 'histogram_cv.png'")
```

**Pillow:**
```python
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка изображения в градациях серого
image_pil = Image.open('example.jpg').convert('L')

# Вычисление гистограммы
hist = image_pil.histogram()

# Визуализация гистограммы
plt.figure(figsize=(10,5))
plt.title("Гистограмма Pillow")
plt.xlabel("Интенсивность пикселя")
plt.ylabel("Количество пикселей")
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.savefig('histogram_pil.png')
plt.show()
print("Гистограмма сохранена как 'histogram_pil.png'")
```

---

#### **Задание 9: Рисование фигур и текста на изображениях**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV и Pillow.
2. Нарисуйте на изображении линию, прямоугольник, круг и добавьте текст.
3. Сохраните измененные изображения.

**Решение:**

**OpenCV:**
```python
import cv2

# Загрузка изображения
image_cv = cv2.imread('example.jpg')

# Рисование линии
cv2.line(image_cv, (50, 50), (200, 200), (0, 255, 0), 3)

# Рисование прямоугольника
cv2.rectangle(image_cv, (300, 50), (450, 200), (255, 0, 0), 2)

# Рисование круга
cv2.circle(image_cv, (150, 300), 50, (0, 0, 255), -1)  # Закрашенный круг

# Добавление текста
cv2.putText(image_cv, 'OpenCV', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Сохранение измененного изображения
cv2.imwrite('drawn_opencv.jpg', image_cv)
print("Изображение с нарисованными фигурами сохранено как 'drawn_opencv.jpg'")
```

**Pillow:**
```python
from PIL import Image, ImageDraw, ImageFont

# Загрузка изображения
image_pil = Image.open('example.jpg')
draw = ImageDraw.Draw(image_pil)

# Рисование линии
draw.line((50, 50, 200, 200), fill=(0, 255, 0), width=3)

# Рисование прямоугольника
draw.rectangle([300, 50, 450, 200], outline=(255, 0, 0), width=2)

# Рисование круга (эллипса)
draw.ellipse([100, 250, 200, 350], fill=(0, 0, 255), outline=(0, 0, 0))

# Добавление текста
font = ImageFont.load_default()
draw.text((50, 400), "Pillow", fill=(255, 255, 255), font=font)

# Сохранение измененного изображения
image_pil.save('drawn_pillow.png')
print("Изображение с нарисованными фигурами сохранено как 'drawn_pillow.png'")
```

---

#### **Задание 10: Конвертация между форматами OpenCV и Pillow**

**Описание:**
1. Загрузите изображение `example.jpg` с помощью OpenCV.
2. Преобразуйте его в формат Pillow, измените цветовое пространство и сохраните.
3. Загрузите изображение `saved_gray_pil.png` с помощью Pillow.
4. Преобразуйте его обратно в формат OpenCV, измените цветовое пространство и сохраните.

**Решение:**

```python
import cv2
from PIL import Image
import numpy as np

# 1. Загрузка изображения с помощью OpenCV
image_cv = cv2.imread('example.jpg')

# 2. Преобразование в формат Pillow (RGB)
rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_image)

# Изменение цветового пространства в Pillow (градации серого)
gray_pil = pil_image.convert('L')
gray_pil.save('converted_gray_pil.png')
print("Изображение преобразовано и сохранено как 'converted_gray_pil.png'")

# 3. Загрузка изображения с помощью Pillow
image_pil_loaded = Image.open('converted_gray_pil.png')

# 4. Преобразование в формат OpenCV (BGR)
image_np = np.array(image_pil_loaded)
# Если изображение в градациях серого, нужно добавить канал
if len(image_np.shape) == 2:
    image_cv_converted = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
else:
    image_cv_converted = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Изменение цветового пространства в OpenCV (градации серого уже, можно преобразовать в HSV)
hsv_cv = cv2.cvtColor(image_cv_converted, cv2.COLOR_BGR2HSV)
cv2.imwrite('converted_hsv_cv.png', hsv_cv)
print("Изображение преобразовано и сохранено как 'converted_hsv_cv.png'")
```

### Итог семинара

После выполнения всех заданий студенты должны:

- Уметь настраивать рабочую среду разработки с использованием VS Code и виртуального окружения `venv`.
- Загружать изображения с помощью OpenCV и Pillow.
- Отображать изображения, используя функции обеих библиотек.
- Сохранять изображения в различных форматах.
- Изменять размер, обрезать и поворачивать изображения.
- Изменять контраст и яркость изображений.
- Работать с отдельными каналами изображений.
- Визуализировать гистограммы интенсивности пикселей.
- Применять различные фильтры для улучшения или изменения изображений.
- Рисовать фигуры и добавлять текст на изображения.
- Конвертировать изображения между форматами OpenCV и Pillow, понимая различия между ними.

---

### Рекомендации и ресурсы

- **Документация OpenCV:** [https://docs.opencv.org/](https://docs.opencv.org/)
- **Документация Pillow:** [https://pillow.readthedocs.io/en/stable/](https://pillow.readthedocs.io/en/stable/)
- **Учебники и примеры:**
  - [OpenCV-Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
  - [Pillow Tutorial](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html)
- **Книги:**
  - "Learning OpenCV 3" — Adrian Kaehler, Gary Bradski
  - "Python Imaging Library Handbook" — Fredrik Lundh

---

### Обратная связь и вопросы

В конце семинара выделите время для обсуждения возникших вопросов и трудностей. Поощряйте студентов делиться своими результатами и обсуждать проблемы, с которыми они столкнулись при выполнении заданий.

---

**Успешного обучения и продуктивной работы!**