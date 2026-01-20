import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- НАСТРОЙКИ ---
SOURCE_DIR = Path('.')           
DEST_DIR = Path('RESULT_FOLDER') 
# -----------------

# <<< ВОТ ГЛАВНОЕ ИЗМЕНЕНИЕ: УТОЧНЕННЫЕ CSS-СЕЛЕКТОРЫ
# Они нацелены на внутреннюю структуру редактора CodeMirror, который использует Jupyter
CUSTOM_CSS = """
<style>
    /* 1. ГЛАВНЫЙ ФИКС: Перенос строк в ячейках с КОДОМ */
    /* Мы целимся в классы, которые использует редактор CodeMirror */
    .jp-InputArea .jp-Editor pre,
    .jp-CodeMirrorEditor pre,
    .CodeMirror-line {
        white-space: pre-wrap !important;      /* Основное правило для переноса */
        word-break: break-all !important;      /* Ломает даже самые длинные неразрывные строки */
        overflow-wrap: break-word !important;  /* Еще один синоним для надежности */
    }

    /* 2. Фикс для ВЫВОДОВ ячеек (output), на всякий случай */
    .jp-OutputArea-output pre, 
    .jp-RenderedText pre {
        white-space: pre-wrap !important;
        word-break: break-all !important;
    }
    
    /* 3. Опционально: немного уменьшим шрифт кода для компактности */
    .CodeMirror, .CodeMirror pre, .jp-Editor {
        font-size: 11px !important;
    }
</style>
"""

def main():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(exist_ok=True)
    print(f"📂 Целевая папка: {DEST_DIR.absolute()}")
    seen_files = set()

    for file in os.listdir(SOURCE_DIR):
        file_path = SOURCE_DIR / file
        if not file_path.is_file():
            continue
        
        if file.endswith('.py') and file != Path(__file__).name:
            dest_file = get_unique_name(file, seen_files)
            print(f"🐍 Копирую py: {file} -> {dest_file}")
            shutil.copy2(file_path, DEST_DIR / dest_file)

        elif file.endswith('.ipynb'):
            print(f"📓 Обработка ноутбука: {file}")
            
            # --- Шаг 1: Конвертация в HTML (наша надежная основа) ---
            html_filename = get_unique_name(file.replace('.ipynb', '.html'), seen_files)
            html_dest = DEST_DIR / html_filename
            
            cmd_html = [
                sys.executable, '-m', 'nbconvert',
                '--to', 'html',
                str(file_path),
                '--output', str(html_dest)
            ]
            
            try:
                subprocess.run(cmd_html, check=True, capture_output=True, text=True)
                
                # --- Шаг 2: Внедряем наш мощный CSS в готовый HTML ---
                patch_html_with_css(html_dest)
                print(f"   ✅ Создан HTML (с переносом строк): {html_filename}")

                # --- Шаг 3: Пробуем сделать PDF (он может быть кривым, но HTML нас спасет) ---
                pdf_filename = html_filename.replace('.html', '.pdf')
                print(f"   ⏳ Пробую конвертировать в PDF...")
                
                cmd_pdf = [
                    sys.executable, '-m', 'nbconvert',
                    '--to', 'webpdf',
                    '--allow-chromium-download',
                    '--WebPDFExporter.disable_sandbox=True',
                    str(file_path),
                    '--output-dir', str(DEST_DIR),
                    '--output', pdf_filename.replace('.pdf', '')
                ]
                
                res = subprocess.run(cmd_pdf, capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"   🎉 PDF готов: {pdf_filename}")
                else:
                    print(f"   ⚠️ PDF не получился. Ошибка:\n{res.stderr[:300]}...")
                
                # Удаляем HTML после обработки
                if html_dest.exists():
                    html_dest.unlink()

            except Exception as e:
                print(f"❌ Полный провал с {file}: {e}")

    print("\n🏁 Сборка завершена.")

def patch_html_with_css(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '</head>' in content:
            new_content = content.replace('</head>', f'{CUSTOM_CSS}\n</head>')
        else:
            new_content = CUSTOM_CSS + content
            
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        print(f"   ⚠️ Не удалось пропатчить HTML стилями: {e}")

def get_unique_name(filename, seen_set):
    if filename not in seen_set:
        seen_set.add(filename)
        return filename
    
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 2
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        if new_name not in seen_set:
            seen_set.add(new_name)
            return new_name
        counter += 1

if __name__ == "__main__":
    main()