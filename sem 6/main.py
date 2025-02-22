import cv2
import numpy as np

video_path = "/Users/diemvs/Documents/GitHub/computer-vision/sem 6/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Ошибка: не удалось открыть видеофайл {video_path}")
else:
    print(f"Видео успешно загружено: {video_path}")

cap.release()

# --- Настройки алгоритма Shi-Tomasi для детекции ключевых точек ---
feature_params = dict(
    maxCorners=100,  # Количество точек для отслеживания
    qualityLevel=0.3,  # Минимальное качество углов
    minDistance=7,  # Минимальное расстояние между углами
    blockSize=7  # Размер блока для вычисления градиента
)

# --- Параметры Лукаса-Канаде для вычисления оптического потока ---
lk_params = dict(
    winSize=(15, 15),  # Размер окна поиска
    maxLevel=2,  # Число уровней пирамиды
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Критерий остановки
)

# --- Открываем видео ---
cap = cv2.VideoCapture("/Users/diemvs/Documents/GitHub/computer-vision/sem 6/video.mp4")  # Укажите свой файл или 0 для веб-камеры

# Считываем первый кадр и конвертируем его в градации серого
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Определяем ключевые точки Shi-Tomasi
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Создаем маску для отрисовки траекторий
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Выход, если видео закончилось

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Вычисляем оптический поток методом Лукаса-Канаде ---
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Отбираем точки, которые удалось отследить
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # --- Отрисовка траекторий ---
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)  # Линии движения
        cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)  # Текущие точки

    # Объединяем кадр с маской
    output = cv2.add(frame, mask)

    # --- Отображаем результат ---
    cv2.imshow("Optical Flow - Lucas-Kanade", output)

    # --- Выход по клавише ESC ---
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Обновляем предыдущий кадр и точки
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()