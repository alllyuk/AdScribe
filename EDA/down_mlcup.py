'''
Скрипт для скачивания и распаковки датасета Avito MLCup 2025
Подготовливает данные к задаче генерации описаний к объявлениям

Время выполнения: ~ минут
'''

import os
import pandas as pd
import requests
import time
import json
import zipfile
from tqdm import tqdm


PARQUET_DIR = '/workspace/AAA_project/data/mlcup_dataset/parquet/'
OUT_CSV = '/workspace/AAA_project/data/mlcup_dataset/meta_info.csv'
IMAGES_DIR = '/workspace/AAA_project/data/mlcup_dataset/images'
TMP_DIR = '/workspace/AAA_project/data/mlcup_dataset/tmp'

IMAGE_ZIP_LINKS = [
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0001.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0002.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0003.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0004.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0005.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0006.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0007.zip',

    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0001.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0002.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0003.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0004.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0005.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0006.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0007.zip',

    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0001.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0002.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0003.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0004.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0005.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0006.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0007.zip',

    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0004-chunk_0001.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0004-chunk_0002.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0004-chunk_0003.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0004-chunk_0004.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0004-chunk_0005.zip',

    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0001-chunk_0008.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0002-chunk_0008.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0003-chunk_0008.zip',
    'https://storage.yandexcloud.net/avitotechmlchallenge2025-2/train_title_images_decr/train_images_part_0004-chunk_0006.zip'
]

PARQUET_LINKS = [
    'https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_2/train_part_0001.snappy.parquet',
    'https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_2/train_part_0002.snappy.parquet',
    'https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_2/train_part_0003.snappy.parquet',
    'https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_2/train_part_0004.snappy.parquet',
]


def safe_mkdir(path: str) -> None:
    '''
    Создаёт папку, если её нет

    Parameters
    ----------
    path : str
        Путь к папке, которую нужно создать
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def download_file(url: str, dest: str) -> None:
    '''
    Скачивает файл по ссылке

    Parameters
    ----------
    url : str
        Ссылка на файл
    dest : str
        Путь, куда сохранить файл
    '''
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'Файл {dest} уже существует, пропускаем скачивание')
        return

    print(f'Скачивание {url} -> {dest}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)


def process_parquet_files(parquet_files: list[str]) -> pd.DataFrame:
    '''
    Обрабатывает все parquet-файлы, преобразует их в нужную структуру

    Parameters
    ----------
    parquet_files : list[str]
        Список путей к parquet-файлам

    Returns
    -------
    pd.DataFrame
        Датафрейм с итоговыми данными без дубликатов
    '''
    all_rows: list[dict[str, object]] = []

    for parquet_path in tqdm(parquet_files, desc='Обработка паркетов'):
        chunk = os.path.basename(parquet_path).split('_')[-1].split('.')[0]
        assert chunk in ['0001', '0002', '0003', '0004'], f'Неверный chunk: {chunk}'

        df = pd.read_parquet(parquet_path)
        df['source_chunk'] = chunk

        for idx, row in df.iterrows():
            if row['is_double'] == True:
                all_rows.append({
                    'item_id': str(row['base_item_id']),
                    'title': row['base_title'],
                    'description': row['base_description'],
                    'attrs': row['base_json_params'],
                    'category': row['base_category_name'],
                    'images': json.dumps([str(row['base_title_image'])]) if pd.notnull(row['base_title_image']) else json.dumps([]),
                    'source_chunk': chunk,
                })
            else:
                all_rows.append({
                    'item_id': str(row['base_item_id']),
                    'title':  row['base_title'],
                    'description': row['base_description'],
                    'attrs': row['base_json_params'],
                    'category': row['base_category_name'],
                    'images': json.dumps([str(row['base_title_image'])]) if pd.notnull(row['base_title_image']) else json.dumps([]),
                    'source_chunk': chunk,
                })
                all_rows.append({
                    'item_id': str(row['cand_item_id']),
                    'title': row['cand_title'],
                    'description': row['cand_description'],
                    'attrs': row['cand_json_params'],
                    'category': row['cand_category_name'],
                    'images': json.dumps([str(row['cand_title_image'])]) if pd.notnull(row['cand_title_image']) else json.dumps([]),
                    'source_chunk': chunk,
                })


    df_out = pd.DataFrame(all_rows)
    df_out = df_out.drop_duplicates(subset=['item_id'])
    print(f'Финальный размер: {df_out.shape}')

    df_out['attrs'] = df_out['attrs'].astype(str)
    df_out['description'] = df_out['description'].astype(str)
    df_out['title'] = df_out['title'].astype(str)
    df_out['category'] = df_out['category'].astype(str)

    return df_out


def download_and_extract_images(image_ids: set[str],
                                image_zip_links: list[str]) -> None:
    '''
    Скачивает и распаковывает только нужные картинки по id
    После распаковки удаляет архив и удаляет все ненужные картинки

    Parameters
    ----------
    image_ids : set[str]
        Множество нужных id изображений (без расширения)
    image_zip_links : list[str]
        Ссылки на zip-архивы с картинками
    images_dir : str
        Куда складывать картинки
    tmp_dir : str
        Куда временно скачивать архивы
    '''
    safe_mkdir(IMAGES_DIR)
    safe_mkdir(TMP_DIR)

    needed_files = set(f'{img_id}.jpg' for img_id in image_ids)

    for url in tqdm(image_zip_links,
                    desc='Качаем и распаковываем архивы с картинками'):
        zip_name = os.path.basename(url)
        zip_path = os.path.join(TMP_DIR, zip_name)

        download_file(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            files_in_zip = set(zf.namelist())
            extract_files = needed_files & files_in_zip
            for fname in tqdm(extract_files, desc=f'Распаковка {zip_name}', leave=False):
                zf.extract(fname, IMAGES_DIR)
        os.remove(zip_path)

    existing_imgs = set(os.listdir(IMAGES_DIR))
    needed_imgs = set(f"{img_id}.jpg" for img_id in image_ids)
    for fname in tqdm(existing_imgs, desc='Удаляем неиспользуемые картинки'):
        if fname not in needed_imgs:
            try:
                os.remove(os.path.join(IMAGES_DIR, fname))
            except Exception:
                pass


def create_df() -> None:
    '''
    Основная функция для создания датафрейма
    '''
    safe_mkdir(PARQUET_DIR)
    for url in PARQUET_LINKS:
        fname = url.split('/')[-1]
        dest = os.path.join(PARQUET_DIR, fname)
        download_file(url, dest)

    parquet_files = [
        os.path.join(PARQUET_DIR, fname)
        for fname in sorted(os.listdir(PARQUET_DIR))
        if fname.endswith('.parquet')
    ]
    print('Найденные файлы:', parquet_files)

    df_out = process_parquet_files(parquet_files)

    df_out.to_csv(OUT_CSV, index=False)
    print(f'Ура: {OUT_CSV}')


if __name__ == '__main__':

    print('=== Создание датафрейма из паркетов ===')
    time_start = time.time()
    # create_df() # 8 минут
    time_end = time.time()
    print(f'Время выполнения: {(time_end - time_start) / 60:.2f} минут')

    print('=== Скачивание и распаковка картинок ===')
    time_start = time.time()

    all_image_ids: set[str] = set()
    no_image_count = 0
    df_out = pd.read_csv(OUT_CSV)

    for imgs_json in df_out['images'].astype(str):
        if imgs_json in ('[]', '', 'nan', 'NaN', 'None'):
            no_image_count += 1
            continue
        try:
            ids = json.loads(imgs_json)
            all_image_ids.update(ids)
        except Exception as e:
            print(f'Ошибка: {e}')
            print(f'В all_image_ids добавили imgs_json={imgs_json}')
            all_image_ids.add(imgs_json)

            print(f'''
                Ошибок: {no_image_count},
                Верных: {len(all_image_ids)},
                Всего планировалось: {df_out.shape[0]}
                ''')

    download_and_extract_images(
        image_ids=all_image_ids,
        image_zip_links=IMAGE_ZIP_LINKS,
    )
    time_end = time.time()
    print(f'Время выполнения: {(time_end - time_start) / 60:.2f} минут')

    print('=== Готово! ===')
