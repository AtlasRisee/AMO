# АМО, Практическое задание №3
### Выполнили: Родин Григорий Юрьевич, Манаков Егор Валерьевич

Этот репозиторий содержит проект, генерирующий данные о количестве электроэнергии, потребляемой каждый час в течение суток в течение года. Сгенерированные данные сохраняются в виде DataFrame с помощью библиотеки pandas и разделяются на обучающую и тестовую выборки с помощью функции train_test_split из библиотеки sklearn.model_selection. Обучающая и тестовая выборки сохраняются в виде отдельных CSV-файлов в соответствующих директориях.

Этот проект может быть использован для обучения моделей машинного обучения для предсказания потребления электроэнергии.

В репозитории содержатся следующие файлы и директории:

* `data_creation.py` - скрипт для генерации данных и разделения их на обучающую и тестовую выборки
* `data/` - директория, содержащая обучающую и тестовую выборки в виде CSV-файлов

Для использования проекта необходимо клонировать репозиторий и запустить скрипт `data_creation.py`. Сгенерированные данные будут сохранены в директории `data/`.