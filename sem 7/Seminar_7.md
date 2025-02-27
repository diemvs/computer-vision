### **Задание 1. Предобработка изображений и извлечение признаков**  
1. Сформируйте набор данных, используя один из стандартных датасетов (например, MNIST или CIFAR-10).  
2. Реализуйте этапы предобработки: изменение размера изображений, нормализацию значений пикселей, устранение шумов (например, с использованием гауссового фильтра).  
3. Извлеките признаки с помощью методов HOG и LBP.  
4. Визуализируйте исходные изображения и соответствующие карты признаков.  
5. Проанализируйте, как изменения параметров (размер ячеек, количество ориентаций для HOG; радиус и количество соседей для LBP) влияют на полученные признаки.

---

### **Задание 2. Классификация изображений с использованием метода k-ближайших соседей (kNN)**  
1. Используя векторы признаков, полученные на предыдущем задании, разделите данные на тренировочную и тестовую выборки.  
2. Реализуйте алгоритм kNN с выбором оптимального значения параметра k.  
3. Оцените точность модели на тестовой выборке и постройте график зависимости точности от значения k.  
4. Обсудите, как выбор метрики расстояния (евклидова, манхэттенская) влияет на результаты классификации.

---

### **Задание 3. Классификация изображений с использованием опорных векторов (SVM)**  
1. На основе извлечённых признаков обучите модель SVM для задачи классификации изображений.  
2. Используйте как линейное, так и RBF ядро, сравните результаты.  
3. Проведите Grid Search для подбора оптимальных значений гиперпараметров \( C \) и \( \gamma \).  
4. Оцените эффективность модели с использованием метрик точности, precision, recall и F1-score.

---

### **Задание 4. Применение логистической регрессии для классификации изображений**  
1. Обучите модель логистической регрессии на выбранном наборе признаков и размеченных данных.  
2. Реализуйте регуляризацию (L1 и L2) и сравните, как она влияет на качество модели и разреженность весов.  
3. Оцените модель с помощью ROC AUC и постройте кривую ROC для тестовой выборки.  
4. Обсудите преимущества и ограничения логистической регрессии в контексте задач компьютерного зрения.

---

### **Задание 5. Снижение размерности с использованием метода главных компонент (PCA)**  
1. Примените PCA для снижения размерности векторов признаков, полученных из изображений.  
2. Исследуйте, сколько компонент необходимо сохранить для сохранения 95% дисперсии данных.  
3. Визуализируйте данные в новом пространстве (например, с использованием двух- или трёхмерной проекции).  
4. Обсудите, как снижение размерности влияет на качество классификации, сравнив результаты до и после применения PCA.

---

### **Задание 6. Ансамблевые методы: случайный лес и градиентный бустинг**  
1. Обучите модели случайного леса и градиентного бустинга на тех же данных, что использовались в предыдущих заданиях.  
2. Проведите сравнительный анализ моделей по точности, стабильности и вычислительной сложности.  
3. Постройте таблицу или график, демонстрирующий влияние числа деревьев (или слабых моделей) на качество предсказания.  
4. Обсудите, какие методы ансамблирования оказались наиболее эффективными для вашего набора данных.

---

### **Задание 7. Кросс-валидация и подбор гиперпараметров**  
1. Реализуйте кросс-валидацию для одной из выбранных моделей (например, SVM или логистической регрессии).  
2. С использованием Grid Search или Random Search найдите оптимальные значения гиперпараметров для выбранной модели.  
3. Проанализируйте стабильность модели, сравнив результаты на разных фолдах кросс-валидации.  
4. Обсудите, как изменение гиперпараметров влияет на риск переобучения и обобщающую способность модели.

---

### **Задание 8. Визуализация результатов и анализ ошибок**  
1. Для одной из обученных моделей (например, SVM или случайного леса) постройте матрицу ошибок (confusion matrix) и визуализируйте её.  
2. Проанализируйте, какие классы наиболее часто путаются и предположите причины этих ошибок (низкая контрастность, похожие текстуры, недостаточное количество обучающих примеров).  
3. Реализуйте визуализацию распределения признаков с использованием t-SNE или PCA, чтобы оценить разделимость классов в пространстве признаков.  
4. Обсудите, какие дополнительные шаги можно предпринять для улучшения разделимости классов.

---

### **Задание 9. Интеграция моделей в систему и оценка вычислительной эффективности**  
1. Разработайте конвейер обработки, включающий предобработку, извлечение признаков, снижение размерности и обучение выбранной модели.  
2. Оцените время обработки одного изображения или кадра, используя выбранный конвейер, с измерениями на CPU и (если возможно) на GPU.  
3. Проведите сравнительный анализ вычислительной сложности различных этапов конвейера и предложите способы оптимизации (например, параллельные вычисления или использование ускоренных библиотек).  
4. Обсудите применимость разработанного решения для систем реального времени, таких как видеонаблюдение или мобильные приложения.

---

### **Задание 10. Гибридные методы и интерпретируемость моделей**  
1. Разработайте гибридный конвейер, где предварительно извлечённые классическими методами признаки (например, HOG или LBP) объединяются с признаками, извлекаемыми с помощью предварительно обученной сверточной нейронной сети.  
2. Обучите классификатор (например, SVM) на объединённом векторе признаков.  
3. Проведите анализ вклада отдельных групп признаков в итоговое решение модели, используя методы интерпретации (например, анализ весовых коэффициентов или LIME).  
4. Обсудите, как гибридный подход влияет на точность и интерпретируемость модели, а также как можно использовать полученные знания для дальнейшей оптимизации системы.

--- 

Каждое из заданий направлено на практическое освоение ключевых этапов построения моделей машинного обучения в компьютерном зрении. Рекомендуется выполнить задания последовательно, документируя результаты и проводя сравнительный анализ для получения полного представления о возможностях и ограничениях различных методов.