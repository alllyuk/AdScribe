import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import re
import ast
import warnings
warnings.filterwarnings('ignore')

class NeuralTextLabeler:
    def __init__(self, device='cpu'):
        """
        Инициализация нейросетевого классификатора на основе эмбеддингов
        """
        self.device = device
        
        # Загружаем предобученные модели
        print("Загружаем предобученные модели...")
        
        # Модель для русского языка - sentence transformers
        self.sentence_model = SentenceTransformer('cointegrated/rubert-tiny2')
        
        # Модель для анализа тональности и семантики
        self.tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.bert_model = AutoModel.from_pretrained('cointegrated/rubert-tiny2')
        
        # Эталонные тексты для каждого класса
        self.reference_texts = {
            'single_item': [
                "Продаю ботинки для девочки в отличном состоянии",
                "Туфли школьные черные размер 32",
                "Кроссовки детские Adidas практически новые",
                "Сандалии летние для мальчика удобные",
                "Сапоги зимние теплые с мехом"
            ],
            'multiple_items': [
                "Продаю два комплекта обуви за одну цену",
                "Отдам обе пары туфель вместе",
                "Набор детской обуви кроссовки и сандалии",
                "Продаю две пары ботинок по отдельности или вместе",
                "Комплект обуви для детского сада сменка и уличная"
            ],
            'unclear': [
                "Состояние не очень хорошее возможно подойдет",
                "Не уверен в размере может быть больше",
                "Сложно сказать точное состояние обуви",
                "Возможно есть небольшие дефекты не очень видно",
                "Может быть нужно почистить или отремонтировать"
            ]
        }
        
        # Создаем эталонные эмбеддинги
        self._create_reference_embeddings()
        
        # Инициализируем дополнительные компоненты
        self.scaler = StandardScaler()
        self.meta_classifier = None
        self._setup_numerical_features()

    def _create_reference_embeddings(self):
        """Создает эталонные эмбеддинги для каждого класса"""
        print("Создаем эталонные эмбеддинги...")
        
        self.reference_embeddings = {}
        
        for class_name, texts in self.reference_texts.items():
            embeddings = self.sentence_model.encode(texts)
            # Используем центроид как представление класса
            self.reference_embeddings[class_name] = np.mean(embeddings, axis=0)
    
    def _setup_numerical_features(self):
        """Настройка извлечения численных признаков"""
        self.quantity_patterns = [
            (r'(\d+)\s*пар[аи]?', 'exact_pairs'),
            (r'две\s*пары?', 'two_pairs'),
            (r'обе\s*пары?', 'both_pairs'),
            (r'несколько\s*пар', 'several_pairs'),
            (r'комплект', 'set'),
            (r'набор', 'collection'),
            (r'вместе', 'together'),
            (r'отдельно', 'separately'),
            (r'за\s*все', 'for_all'),
            (r'один\s*(раз|предмет)', 'single_use'),
            (r'единственный', 'unique'),
            (r'может\s*быть', 'maybe'),
            (r'возможно', 'possibly'),
            (r'не\s*уверен', 'unsure'),
            (r'сложно\s*сказать', 'hard_to_say')
        ]

    def extract_numerical_features(self, text: str) -> np.ndarray:
        """Извлекает численные признаки из текста"""
        text_lower = text.lower()
        features = []
        
        # Паттерны количества
        for pattern, feature_name in self.quantity_patterns:
            matches = len(re.findall(pattern, text_lower))
            features.append(matches)
        
        # Длина текста
        features.append(len(text))
        features.append(len(text.split()))
        
        # Количество цифр
        features.append(len(re.findall(r'\d+', text)))
        
        # Количество вопросительных и восклицательных знаков
        features.append(text.count('?'))
        features.append(text.count('!'))
        
        # Количество изображений (если есть)
        try:
            if '[' in text and ']' in text:
                images_part = re.findall(r'\[.*?\]', text)
                if images_part:
                    images_list = ast.literal_eval(images_part[0])
                    features.append(len(images_list))
                else:
                    features.append(1)
            else:
                features.append(1)
        except:
            features.append(1)
        
        return np.array(features)

    def get_text_embeddings(self, texts: list) -> np.ndarray:
        """Получает эмбеддинги для списка текстов"""
        return self.sentence_model.encode(texts)

    def calculate_semantic_distances(self, text_embedding: np.ndarray) -> dict:
        """Вычисляет семантические расстояния до эталонных классов"""
        distances = {}
        
        for class_name, ref_embedding in self.reference_embeddings.items():
            # Косинусное сходство
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1), 
                ref_embedding.reshape(1, -1)
            )[0][0]
            
            # Преобразуем в расстояние
            distance = 1 - similarity
            distances[class_name] = distance
        
        return distances

    def extract_bert_features(self, text: str) -> np.ndarray:
        """Извлекает признаки из BERT модели"""
        # Токенизируем текст
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        # Получаем эмбеддинги
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Используем [CLS] токен как представление всего текста
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return cls_embedding.flatten()

    def cluster_analysis(self, texts: list, n_clusters: int = 3) -> dict:
        """Выполняет кластерный анализ текстов"""
        embeddings = self.get_text_embeddings(texts)
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Анализ кластеров
        cluster_info = {}
        for i in range(n_clusters):
            cluster_texts = [texts[j] for j in range(len(texts)) if cluster_labels[j] == i]
            cluster_center = kmeans.cluster_centers_[i]
            
            # Определяем ближайший эталонный класс
            distances_to_refs = {}
            for class_name, ref_embedding in self.reference_embeddings.items():
                distance = np.linalg.norm(cluster_center - ref_embedding)
                distances_to_refs[class_name] = distance
            
            closest_class = min(distances_to_refs, key=distances_to_refs.get)
            
            cluster_info[i] = {
                'texts_count': len(cluster_texts),
                'closest_reference_class': closest_class,
                'distance_to_closest': distances_to_refs[closest_class],
                'sample_texts': cluster_texts[:3]  # Примеры текстов
            }
        
        return cluster_info, cluster_labels

    def train_meta_classifier(self, df: pd.DataFrame) -> None:
        """Обучает мета-классификатор на основе всех признаков"""
        print("Обучаем мета-классификатор...")
        
        # Создаем предварительные псевдометки на основе эмбеддингов
        combined_texts = (df['title'].fillna('') + ' ' + 
                         df['description'].fillna('') + ' ' + 
                         df['other'].fillna('')).tolist()
        
        text_embeddings = self.get_text_embeddings(combined_texts)
        
        # Псевдометки на основе расстояний до эталонов
        pseudo_labels = []
        confidence_scores = []
        
        for embedding in text_embeddings:
            distances = self.calculate_semantic_distances(embedding)
            
            # Находим ближайший класс
            closest_class = min(distances, key=distances.get)
            min_distance = distances[closest_class]
            
            # Вычисляем уверенность
            sorted_distances = sorted(distances.values())
            if len(sorted_distances) > 1:
                confidence = (sorted_distances[1] - sorted_distances[0]) / sorted_distances[1]
            else:
                confidence = 0.5
            
            # Преобразуем в индексы классов
            class_to_idx = {'single_item': 0, 'multiple_items': 1, 'unclear': 2}
            pseudo_labels.append(class_to_idx[closest_class])
            confidence_scores.append(confidence)
        
        # Извлекаем численные признаки
        numerical_features = []
        for text in combined_texts:
            num_features = self.extract_numerical_features(text)
            numerical_features.append(num_features)
        
        numerical_features = np.array(numerical_features)
        
        # Объединяем эмбеддинги и численные признаки
        scaled_numerical = self.scaler.fit_transform(numerical_features)
        combined_features = np.hstack([text_embeddings, scaled_numerical])
        
        # Обучаем Random Forest только на примерах с высокой уверенностью
        high_confidence_mask = np.array(confidence_scores) > 0.3
        
        if np.sum(high_confidence_mask) > 10:  # Минимум примеров для обучения
            train_features = combined_features[high_confidence_mask]
            train_labels = np.array(pseudo_labels)[high_confidence_mask]
            
            self.meta_classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            )
            self.meta_classifier.fit(train_features, train_labels)
            
            print(f"Мета-классификатор обучен на {len(train_features)} примерах")
        else:
            print("Недостаточно данных для обучения мета-классификатора")

    def classify_text(self, title: str, description: str, other: str, images: str = '') -> tuple:
        """Классифицирует текст и возвращает класс с уверенностью"""
        # Объединяем все текстовые поля
        combined_text = f"{title} {description} {other}".strip()
        
        # Получаем эмбеддинг
        text_embedding = self.sentence_model.encode([combined_text])[0]
        
        # Вычисляем расстояния до эталонов
        distances = self.calculate_semantic_distances(text_embedding)
        
        # Извлекаем численные признаки
        numerical_features = self.extract_numerical_features(combined_text + ' ' + images)
        scaled_numerical = self.scaler.transform([numerical_features])
        
        # Если есть обученный мета-классификатор
        if self.meta_classifier is not None:
            combined_features = np.hstack([text_embedding.reshape(1, -1), scaled_numerical])
            
            # Предсказание мета-классификатора
            meta_pred = self.meta_classifier.predict(combined_features)[0]
            meta_proba = self.meta_classifier.predict_proba(combined_features)[0]
            meta_confidence = np.max(meta_proba)
            
            # Преобразуем обратно в названия классов
            idx_to_class = {0: 'Один предмет', 1: 'Разнообразие предметов', 2: 'Непонятно'}
            meta_class = idx_to_class[meta_pred]
            
            # Комбинируем с результатами эмбеддингов
            closest_class = min(distances, key=distances.get)
            embedding_confidence = 1 - distances[closest_class]
            
            # Преобразуем названия классов
            class_mapping = {
                'single_item': 'Один предмет',
                'multiple_items': 'Разнообразие предметов', 
                'unclear': 'Непонятно'
            }
            embedding_class = class_mapping[closest_class]
            
            # Финальное решение на основе согласованности
            if meta_class == embedding_class:
                final_confidence = min(0.95, (meta_confidence + embedding_confidence) / 2)
                return meta_class, final_confidence
            else:
                # Выбираем более уверенный результат
                if meta_confidence > embedding_confidence:
                    return meta_class, meta_confidence * 0.8
                else:
                    return embedding_class, embedding_confidence * 0.8
        
        else:
            # Используем только эмбеддинги
            closest_class = min(distances, key=distances.get)
            min_distance = distances[closest_class]
            
            # Вычисляем уверенность
            sorted_distances = sorted(distances.values())
            if len(sorted_distances) > 1:
                confidence = (sorted_distances[1] - sorted_distances[0]) / sorted_distances[1]
                confidence = min(0.9, max(0.3, confidence))
            else:
                confidence = 0.5
            
            # Преобразуем в итоговые названия классов
            class_mapping = {
                'single_item': 'Один предмет',
                'multiple_items': 'Разнообразие предметов',
                'unclear': 'Непонятно'
            }
            
            return class_mapping[closest_class], confidence

    def process_dataset(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """Обрабатывает весь датасет и создает разметку"""
        print("Загружаем датасет...")
        df = pd.read_csv(csv_path)
        
        print(f"Обрабатываем {len(df)} записей...")
        
        # Сначала обучаем мета-классификатор
        self.train_meta_classifier(df)
        
        # Применяем классификацию к каждой записи
        results = []
        for idx, row in df.iterrows():
            title = str(row.get('title', ''))
            description = str(row.get('description', ''))
            other = str(row.get('other', ''))
            images = str(row.get('images', ''))
            
            label, confidence = self.classify_text(title, description, other, images)
            results.append((label, confidence))
        
        # Разделяем результаты на классы и уверенность
        df['label'] = [result[0] for result in results]
        df['confidence'] = [result[1] for result in results]
        
        # Создаем финальный датасет
        labeled_df = df[['main_img', 'label', 'confidence']].copy()
        labeled_df = labeled_df.rename(columns={'main_img': 'image_id'})
        
        # Статистика
        print("\nСтатистика разметки:")
        print(labeled_df['label'].value_counts())
        print(f"\nСредняя уверенность: {labeled_df['confidence'].mean():.3f}")
        
        # Дополнительный анализ
        print("\nАнализ по классам:")
        for label in labeled_df['label'].unique():
            subset = labeled_df[labeled_df['label'] == label]
            print(f"{label}: {len(subset)} примеров, "
                  f"средняя уверенность: {subset['confidence'].mean():.3f}")
        
        # Сохраняем результат
        if output_path is None:
            output_path = csv_path.replace('.csv', '_labeled.csv')
        
        labeled_df.to_csv(output_path, index=False)
        print(f"\nРазмеченный датасет сохранен в: {output_path}")
        
        return labeled_df

    def analyze_dataset_semantics(self, df: pd.DataFrame) -> dict:
        """Выполняет семантический анализ всего датасета"""
        print("Выполняем семантический анализ датасета...")
        
        # Объединяем все тексты
        combined_texts = (df['title'].fillna('') + ' ' + 
                         df['description'].fillna('') + ' ' + 
                         df['other'].fillna('')).tolist()
        
        # Получаем эмбеддинги
        embeddings = self.get_text_embeddings(combined_texts)
        
        # Кластерный анализ
        cluster_info, cluster_labels = self.cluster_analysis(combined_texts, n_clusters=5)
        
        # Анализ распределения расстояний до эталонов
        distances_analysis = {
            'single_item': [],
            'multiple_items': [],
            'unclear': []
        }
        
        for embedding in embeddings:
            distances = self.calculate_semantic_distances(embedding)
            for class_name, distance in distances.items():
                distances_analysis[class_name].append(distance)
        
        # Статистика расстояний
        for class_name, distances in distances_analysis.items():
            print(f"\n{class_name}:")
            print(f"  Среднее расстояние: {np.mean(distances):.3f}")
            print(f"  Стандартное отклонение: {np.std(distances):.3f}")
            print(f"  Минимальное расстояние: {np.min(distances):.3f}")
            print(f"  Максимальное расстояние: {np.max(distances):.3f}")
        
        return {
            'cluster_info': cluster_info,
            'cluster_labels': cluster_labels,
            'distances_analysis': distances_analysis,
            'embeddings': embeddings
        }

def main():
    # Пути к файлам
    input_csv = "/workspace/AAA_project/experimental/final_classifier/meta_info_with_main_img.csv"
    output_csv = "/workspace/AAA_project/experimental/final_classifier/labeled_dataset.csv"
    
    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    # Создаем нейросетевой разметчик
    labeler = NeuralTextLabeler(device=device)
    
    # Обрабатываем датасет
    labeled_df = labeler.process_dataset(input_csv, output_csv)
    
    # Выполняем семантический анализ
    df = pd.read_csv(input_csv)
    semantic_analysis = labeler.analyze_dataset_semantics(df)
    
    print("\n=== СЕМАНТИЧЕСКИЙ АНАЛИЗ ===")
    print(f"Кластерный анализ выявил {len(semantic_analysis['cluster_info'])} групп:")
    
    for cluster_id, info in semantic_analysis['cluster_info'].items():
        print(f"\nКластер {cluster_id}:")
        print(f"  Количество текстов: {info['texts_count']}")
        print(f"  Ближайший эталонный класс: {info['closest_reference_class']}")
        print(f"  Расстояние до эталона: {info['distance_to_closest']:.3f}")
        print(f"  Примеры текстов: {info['sample_texts']}")
    
    print("\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"Общее количество размеченных примеров: {len(labeled_df)}")
    print("Распределение по классам:")
    for label in labeled_df['label'].unique():
        subset = labeled_df[labeled_df['label'] == label]
        percentage = len(subset) / len(labeled_df) * 100
        print(f"  {label}: {len(subset)} ({percentage:.1f}%) - "
              f"средняя уверенность: {subset['confidence'].mean():.3f}")

if __name__ == "__main__":
    main()