#!/bin/bash
# Удаляет кэш предобученных моделей HuggingFace (transformers, datasets, diffusers и т.д.)
# Подробно выводит размер кэша до и после удаления

# Определяем домашнюю директорию
USER_HOME="${HOME:-/root}"

# Пути к кэшу HuggingFace
HF_CACHE_DIRS=(
    "$USER_HOME/.cache/huggingface"
    "$USER_HOME/.cache/torch/transformers"
    "$USER_HOME/.cache/torch/hub/checkpoints"
    "$USER_HOME/.cache/torch/hub"
    "$USER_HOME/.huggingface"
    "$USER_HOME/.cache/diffusers"
)

printf "\U1F9EA Очистка кэша HuggingFace\n"
for cache_dir in "${HF_CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        size_before=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
        echo -e "\n\U1F4C1 Найден кэш: $cache_dir"
        echo "  📏 Размер до очистки: $size_before"
        rm -rf "$cache_dir"
        if [ ! -d "$cache_dir" ]; then
            echo "  🗑️  Кэш удалён."
        else
            echo "  ❌ Не удалось удалить кэш: $cache_dir"
            size_after=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
            echo "  📏 Размер после попытки удаления: $size_after"
        fi
    else
        echo -e "\n🟦 Кэш не найден: $cache_dir"
    fi
    sleep 0.2
done

echo -e "\n\U1F389 Очистка кэша HuggingFace завершена."