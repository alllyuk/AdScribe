'''
Скрипт для скачивания и распаковки основного датасета

Время выполнения: ~5-20 минут
'''

import os
import re
import zipfile
import tarfile
import gdown
import shutil
import time


MAIN_DATASET_URL = 'https://drive.google.com/uc?export=download&id=1_3SVFUpacIAH5LZQkIMotlkRbsHVb744'
MAIN_DATASET_DIR = '/workspace/AAA_project/data/main_dataset'
IMAGES_DIR = os.path.join(MAIN_DATASET_DIR, 'images')


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


def extract_if_archive(file_path: str) -> None:
    '''
    Распаковывает архив и организует файлы:
    - изображения -> папка images/
    - csv и другие файлы -> корневая папка

    Parameters
    ----------
    file_path : str
        Путь к архиву, который нужно распаковать
    '''
    extract_dir = os.path.dirname(file_path)
    temp_extract_dir = os.path.join(extract_dir, 'temp_extract')
    
    try:
        safe_mkdir(temp_extract_dir)
        safe_mkdir(IMAGES_DIR)
        
        print(f'Распаковка {os.path.basename(file_path)}...')
        
        if file_path.lower().endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
        elif file_path.lower().endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(temp_extract_dir)
        
        print('Организация файлов датасета...')
        
        # Перебираем все файлы в распакованном архиве
        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                src_path = os.path.join(root, file)
                
                # Определяем путь назначения для каждого файла
                # Изображения в images/, остальное в корень
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    dst_path = os.path.join(IMAGES_DIR, file)
                else:
                    dst_path = os.path.join(extract_dir, file)
                
                shutil.move(src_path, dst_path)
        
        shutil.rmtree(temp_extract_dir)
        os.remove(file_path)
        
        print(f'Архив {os.path.basename(file_path)} распакован и удален')
        
    except Exception as e:
        print(f'Ошибка при распаковке {os.path.basename(file_path)}: {e}')
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)


def download_from_gdrive() -> None:
    '''
    Скачивает датасет с Google Drive
    '''
    safe_mkdir(MAIN_DATASET_DIR)
    
    file_id_match = re.search(r'id=([^&]+)', MAIN_DATASET_URL)
    if not file_id_match:
        print('Ошибка: не удалось извлечь ID файла из URL Google Drive')
        return
    
    file_id = file_id_match.group(1)
    file_path = os.path.join(MAIN_DATASET_DIR, 'main_dataset.tar.gz')
    
    if os.path.exists(file_path):
        print(f'Файл {os.path.basename(file_path)} уже существует, пропускаем')
        return
    
    print(f'Скачивание основного датасета с Google Drive...')
    try:
        '''
        gdown.download(f'https://drive.google.com/uc?id={file_id}',
                       file_path,
                       quiet=False)
        '''
        gdown.download(
            url=f'https://drive.google.com/uc?id={file_id}',
            output=file_path,
            quiet=False,
            use_cookies=False,
            fuzzy=True
        )
        extract_if_archive(file_path)
    except Exception as e:
        print(f'Ошибка при скачивании с Google Drive: {e}')


def main() -> None:
    start_time = time.time()
    
    print('=== Скачивание и распаковка основного датасета ===')
    download_from_gdrive()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f'\nГотово! Датасет находится в {MAIN_DATASET_DIR}')
    print(f'CSV файлы в корневой папке, изображения в подпапке images/')
    print(f'Время выполнения: {int(minutes)} мин {int(seconds)} сек')


if __name__ == '__main__':
    main()