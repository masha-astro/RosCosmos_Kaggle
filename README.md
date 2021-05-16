# Implementation of [publication](https://ieeexplore.ieee.org/document/6946667) for detecting changes on satellite images

# Made for solving [Kaggle problem](https://www.kaggle.com/c/roscosmos-rucode/overview)

Суть эксперимента - показать как работает алгоритм классической обработки изображений [SIFT](https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%81%D1%88%D1%82%D0%B0%D0%B1%D0%BD%D0%BE-%D0%B8%D0%BD%D0%B2%D0%B0%D1%80%D0%B8%D0%B0%D0%BD%D1%82%D0%BD%D0%B0%D1%8F_%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D1%8F_%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2), но в адаптированном к спутниковым изображениям случае. В частности, имплиментирована публикация **F. Dellinger, J. Delon, Y. Gousseau, J. Michel and F. Tupin, "Change detection for high resolution satellite images, based on SIFT descriptors and an a contrario approach," 2014 IEEE Geoscience and Remote Sensing Symposium, 2014, pp. 1281-1284, doi: 10.1109/IGARSS.2014.6946667** [можно посмотреть текст статьи](https://hal.archives-ouvertes.fr/hal-01059366/document). 

Принцип Гельмгольца: человеческие глаза замечают структуру только в том случае, если она не появляется случайным образом. Этот подход основан на двух основных концепциях: фоновой модели, описывающей конфигурации, в которых не должны быть обнаружены структуры (H0 hypothesis), и измерении обнаруживаемых структур. Затем каждому измерению присваивается мера показательности, называемая N F A (Количество ложных тревог), количественно оценивающая маловероятность данной структуры в фоновой модели H0. Затем N F A определяется из вероятности P(X≥x),X -случайная величина из биномиального распределения в фоновой модели. Пороговое значение N F A ≤ ε, при этом ε мало, позволяет нам обнаружить изменение на изображении.

# **Постановка задачи**:

Найти изменения на двух спутниковых изображениях одного и того-же участка Земли. Основные сложности - солнечные блики, облака, атмосферные явления, тени. 

# **Базовое решение**:

Берётся разность изображений и порог интенсивности (я брала нижний и верхний порог, так как есть тени и блики дающие наибольшие значения разности, а также есть облака и атмосферные явления дающие небольшое на вид значение разности). Данное решение также мной реализовано и можно посмотреть на его результаты.

# **Алгоритм решения с помощью SIFT**:

1. Берётся два изображения одной и той же местности, сделанные в разные моменты времени. Находятся ключевые точки (key points) на каждом изображении и их дескрипторы, с помощью алгоритма масштабно-инвариантной трансформации признаков (SIFT).
2. Далее берётся алгоритм KNN с k=2 ближайшими соседями, с помощью которого находятся совпадения ключевых точек на двух изображениях.
3. Используется [Lowe's ratio test](https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work) для отбора достаточно достоверных совпадений.
4. Вокруг каждой ключевой точки берётся окрестность с заданным радиусом, находится количество совпадений в этой окрестности и количество ключевых точек. И смотрим на втором изображении больше совпадений вокруг заданной точки, или же новых ключевых точек по сравнению с первым  изображением, считаем количество ложных тревог N F A для каждой ключевой точки.
5. Если количество ложных тревог N F A меньше заданного нами трешхолда, в итоговую маску изменений (1 - изменение есть, 0 - нет) заносятся 1 по всему радиусу вокруг проверенной ключевой точки.

# **Проверка результатов**:

В Kaggle соревновании были даны 5 валидационных пар изображений, у которых есть итоговые двоичные маски с таргетным наличием изменений. Данные для проверки результатов эксперимента можно скачать по [ссылке](https://www.kaggle.com/c/roscosmos-rucode/data).

# Requirements:

Требуется версия python < 3.9
Для того, чтобы установить все зависимости, требуется выполнить команду в терминале:

``` pip3 install -r requirements.txt ```

# Запуск скрипта на своей машине:

Пример запуска на моей локальной машине, где переменные:

radius - радиус окрестности ключевой точки, в которой рассматриваются совпадения и другие ключевые точки и далее считается NFA - int

eps - трешхолд для NFA (если полученное значение NFA < eps, мы заполняем окрестность заданного радиуса единицами) - float

path2valdata - путь к валидационным маскам - str

path2data - путь ко всем изображениям в 8 каналах - str

Ниже можно ознакомиться с какими параметрами я запускала для решения этой задачи (параметры radius, eps были подобраны варьированием и данные параметры на датасете соревнования дают наилучший результат)

``` python3 script.py --radius 5 --eps 1e-20 --path2valdata '/home/masha/Kaggle/roscosmos-rucode/mask/mask/' --path2data '/home/masha/Kaggle/roscosmos-rucode/Images_composit/Images_composit/8_ch/' ```

Ниже приведена команда, которую нужно скопировать и применить в терминале, предварительно вставив на места <INSERT ...> ваши параметры скрипта. Прежде чем запускать скрипт в терминале, убедитесь, что Вы скачали данные сореванования по [ссылке](https://www.kaggle.com/c/roscosmos-rucode/data), а также проверьте что корректно задаёте полный путь к скачанным данным.

``` python3 script.py --radius <INSERT YOUR RADIUS INT VALUE> --eps <INSERT YOUR EPS FLOAT VALUE> --path2valdata <INSERT THE PATH TO THE VALIDATION MASKS> --path2data <INSERT THE PATH TO ALL THE DATA SET IMAGES IN 8 CHANNELS> ```

# OUTPUT:

Замечание: Сохранение происходит в папку, из которой запускается скрипт.

На выходе сохраняется всего 15 графиков - 5 валидационных масок, 5 масок полученных из базового решения, 5 масок полученных с помощью алгоритма из публикации в файлы с названиями соответственно "target_i.png", "difference_with_threhold_i.png", "sift_i.png" - где i от 0 до 4.

А также сохраняется в submission_for_Kaggle.csv маска в формате, в котором она должна была подаваться в качестве решения на Kaggle для 41 пары фотографий из тестовой выборки.

# Результаты:

[34 место](https://www.kaggle.com/c/roscosmos-rucode/leaderboard) в соревновании из 69 команд
