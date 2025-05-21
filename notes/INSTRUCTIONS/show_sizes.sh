#!/bin/bash
# Красивый вывод структуры директорий с размерами (data: только дерево папок, без файлов, с пометкой 👁️; venv: только факт наличия)
# Использование: bash show_sizes.sh

# ANSI жирный
BOLD='\033[1m'
NC='\033[0m'

# Функция для красивого вывода размера и пометки-цвета
human_size() {
    local size="$1"
    local mark=""
    if (( size > 1073741824 )); then
        mark=" 🔸"
    elif (( size > 10485760 )); then
        mark=" 🔹"
    fi
    echo -e "${BOLD}$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size B")${NC}$mark"
}

# Кэш размеров (ассоциативный массив bash)
declare -A SIZE_CACHE
get_size_cached() {
    local path="$1"
    if [[ -n "${SIZE_CACHE[$path]}" ]]; then
        echo "${SIZE_CACHE[$path]}"
    else
        local size=$(du -sb "$path" 2>/dev/null | cut -f1)
        SIZE_CACHE[$path]="$size"
        echo "$size"
    fi
}

# Функция для обхода дерева (кроме data и venv)
print_tree() {
    local dir="$1"
    local prefix="$2"
    local is_last="$3"
    local entries=()
    while IFS= read -r entry; do entries+=("$entry"); done < <(ls -A "$dir" 2>/dev/null | grep -v -E '^(data|venv|vsix_extensions)$' | sort)
    local count="${#entries[@]}"
    local i=1
    for entry in "${entries[@]}"; do
        local path="$dir/$entry"
        local size=$(get_size_cached "$path")
        local size_str=$(human_size "$size")
        local connector="├──"
        local next_prefix="$prefix│   "
        if [ $i -eq $count ]; then
            connector="└──"
            next_prefix="$prefix    "
        fi
        if [ -d "$path" ]; then
            echo -e "${prefix}${connector} ${BOLD}📁 $entry/${NC} $size_str"
            print_tree "$path" "$next_prefix" 0
        else
            echo -e "${prefix}${connector} 📄 $entry $size_str"
        fi
        i=$((i+1))
    done
    # Если есть папка data, выводим только дерево папок внутри неё и общий размер, а также пометку 👁️
    if [ -d "$dir/data" ]; then
        local data_size=$(get_size_cached "$dir/data")
        local data_size_str=$(human_size "$data_size")
        echo -e "${prefix}├── ${BOLD}📁 data/${NC} $data_size_str 👁️"
        print_data_tree "$dir/data" "$prefix│   "
    fi
    # Если есть папка venv, просто выводим факт наличия и общий размер
    if [ -d "$dir/venv" ]; then
        local venv_size=$(get_size_cached "$dir/venv")
        local venv_size_str=$(human_size "$venv_size")
        echo -e "${prefix}└── ${BOLD}📁 venv/${NC} $venv_size_str (скрыто содержимое)"
    fi
    # Если есть папка vsix_extensions, просто выводим факт наличия и общий размер
    if [ -d "$dir/vsix_extensions" ]; then
        local vsix_size=$(get_size_cached "$dir/vsix_extensions")
        local vsix_size_str=$(human_size "$vsix_size")
        echo -e "${prefix}└── ${BOLD}📁 vsix_extensions/${NC} $vsix_size_str (скрыто содержимое)"
    fi
}

# Функция для вывода только дерева папок внутри data (без файлов, но с размерами папок и пометкой 👁️)
print_data_tree() {
    local dir="$1"
    local prefix="$2"
    local entries=()
    while IFS= read -r entry; do entries+=("$entry"); done < <(ls -A "$dir" 2>/dev/null | sort)
    local count=0
    for entry in "${entries[@]}"; do
        if [ -d "$dir/$entry" ]; then
            count=$((count+1))
        fi
    done
    local i=1
    for entry in "${entries[@]}"; do
        local path="$dir/$entry"
        if [ -d "$path" ]; then
            local connector="├──"
            local next_prefix="$prefix│   "
            if [ $i -eq $count ]; then
                connector="└──"
                next_prefix="$prefix    "
            fi
            local size=$(get_size_cached "$path")
            local size_str=$(human_size "$size")
            echo -e "${prefix}${connector} ${BOLD}📁 $entry/${NC} $size_str 👁️"
            print_data_tree "$path" "$next_prefix"
            i=$((i+1))
        fi
    done
}

# Заголовок
clear
echo -e "${BOLD}Структура проекта /workspace/AAAproj${NC}"
echo

# Верхний уровень
print_tree "." "" 1