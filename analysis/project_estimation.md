## Оценка проекта до взятия в работу

### 1) Потенциал проекта
**Проблема:** Когда пользователи размещают объявление на Авито, требуется написать заголовок, загрузить фото и указать параметры товара, а потом еще и придумать описание. Многие пользователи не знают, что в нём написать, и в итоге там оказывается что-то не очень информативное. 
Генерация описания объявления может помочь улучшить пользовательский опыт:
- продавцов:
  - экономия времени и сил, затрачиваемых на создание описания объявления
  - повышение доверия покупателей к объявлениям 
- покупателей
  - получение более информативных и полных описаний объявлений
 
Согласно [исследованию](https://www.ashmanov.com/static/files/AIP_CJM_Research_2023.pdf), отсутствие описания товара или бренда отталкивает 19% покупателей, а 28% покупателей считают описание важнейшим типом информации при выборе товара. 

По [опросу](https://auto.ru/mag/article/yandexgpt-pomozhet-sozdat-obyavlenie-na-avtoru-i-bystree-prodat-avtomobil/) журнала Авто.ру, примерно 10% пользователей ничего не пишут в описании объявления. Вероятно, не все готовы тратить на составление текста 7–8 минут — столько в среднем уходит на это у 60% пользователей.

Внедрение генерации в некоторых категориях объявлений Авито с дальнейшим A/B-тестом и опросом пользователей [показало](https://habr.com/ru/companies/avito/articles/852958/) следующие результаты при использовании генерации:
- рост заказов с доставкой на 1.7%
- 60% продавцов отметили, что им понравилось описание

Таким образом, стоит попробовать расширить этот функционал на все категории объявлений.


### 2) Простое решение
Простое решение - подстановка значений параметров в заранее подготовленный шаблон описания.

Такое решение обладает сразу несколькими недостатками:
- однообразие получающихся описаний и ухудшение читаемости для покупателей
- невозможность добавления в описание информации с фотографий
- негибкость решения в случае добавления новых параметров у объявлений (придется переписывать шаблон)


### 3) Реалистичность решения проблемы с помощью машинного обучения
Проблема может быть решена с использованием генеративных моделей. Для использования информации с изображений требуется интеграция с моделями `image captioning`.

Имеются [примеры](https://github.com/Aka-Gulu/IP-Auto-Generate-Product-Description-Using-Gen-AI?tab=readme-ov-file) решения таких задач на англоязычных данных. Также эта задача [решалась](https://habr.com/ru/companies/avito/articles/852958/) в Авито, но без использования фотографий товаров.

Возможные ограничения решения:
- Похожесть сгенерированных объявлений друг на друга
- Галлюцинации генеративной модели

Влияние этих проблем может быть уменьшено с помощью настройки гиперпараметров (например, температуры модели) и заданием промпта, регулирующего специфику ответа.


### Технические требования к задаче
Если использовать сервис будет примерно 10% продавцов на подаче объявлений, то можно ориентироваться на rpm 100 и 90 перцентиль ответа 10 сек.
