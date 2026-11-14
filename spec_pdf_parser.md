# Спецификация: PDF-парсер годовых отчётов Сбербанка
## v1.1 — обёртка над Unstructured для Structure-Aware Graph RAG

> Изменение v1.0 → v1.1 (по результатам ревизии спеки и code review):
> 1. **Критично:** `partition_pdf(pages=[a,b])` в unstructured 0.27.36 **игнорируется**
>    (параметр молча уходит в `**kwargs`) — батчи работают только на вырезке страниц
>    через pypdf + `starting_page_number` (курсовой протокол Занятия 2).
> 2. **Критично:** `metadata.image_base64` заполняется только при
>    `extract_image_block_to_payload=True`; без него все `image_b64` в артефакте = None.
> 3. Реальные объёмы документов установлены: 2015 = 74 стр. (а не 222);
>    весь корпус = 2581 стр. Smoke «1143 элемента» относился к **всему** 74-страничному
>    документу 2015, а не к 3 страницам.
> 4. Спека приведена в соответствие с реализацией (`pdf_parser.py`, `cli.py`, `config.py`):
>    `page_label`, `report_period`, `skipped_page_ranges`, `data/figures/`, CLI-флаги.

Документ детализирует **шаги 1–2** из `spec_project.md` (v3.1, §11 «План реализации») и фиксирует интерфейс
`src/ingestion/pdf_parser.py` + артефакты, которые дальше потребляют
`structure_builder.py`, `chunk_enricher.py`, `table_structurer.py`, `visual_describer.py`.

---

## 1. Назначение и границы

| Вопрос | Ответ |
|---|---|
| Что делает | Извлекает типизированные элементы (Title, NarrativeText, Table, Image, ...) из 9 PDF-отчётов с полными метаданными |
| Что НЕ делает | Не строит дерево разделов (→ `structure_builder.py`), не режет чанки (→ `chunk_enricher.py`), не зовёт LLM/VLM |
| Вход | `data/raw/Копия Сбер {2015..2020,2022..2024}.pdf` |
| Выход | `data/elements/{doc_id}_elements.json` + `data/elements/{doc_id}_stats.json` + вырезы `data/figures/{doc_id}/figure-*` |
| Стратегия | `hi_res` (layout-aware, работает на нашей машине — проверено `scripts/test_hi_res.py`) |
| Рельеф данных (факт, pypdf) | 2015=74, 2016=310, 2017=198, 2018=213, 2019=300, 2020=323, 2022=275, 2023=402, 2024=486 стр.; всего 2581; 2021 отсутствует — валидная ситуация |

> **Проверка данных:** smoke 2015 показывает «ОАО «БПС-СБЕРБАНК» ГОДОВОЙ ОТЧЕТ
> ЗА 2015» — это отчёт **БПС-Сбербанка** (белорусская дочка), а не ПАО Сбербанк.
> Файл можно парсить и хранить в корпусе, но downstream metadata обязаны указывать
> `issuer` и `country`. Его нельзя по умолчанию смешивать с рядом российских отчётов
> во временных сравнениях. См. `spec_target.md` §3.

---

## 2. Контракт данных

### 2.1. Нормализованный элемент (`RawElement`)

Unstructured-элемент сериализуется в JSON-объект (реализовано в `src/core/models.py`):

```python
@dataclass
class RawElement:
    element_id: str        # f"{doc_id}:{index:05d}" → мапится на chunk.element_ids
    doc_id: str            # "sber_2015" ... "sber_2024" (см. 2.2)
    type: str              # "Title" | "NarrativeText" | "Table" | "Image" |
                           # "ListItem" | "Header" | "FigureCaption" | "Text" | ...
    text: str              # элемент в текстовом виде ("" если чисто графический)
    page: int              # metadata.page_number, 1-based
    page_label: Optional[str]  # метка печати (PdfReader.page_labels, контракт чанка К2);
                           # None, если метки недоступны/не совпадают по числу
    category_depth: int    # для Title: уровень заголовка (0..3); для остальных -1
    html: Optional[str]    # Table → metadata.text_as_html; для прочих None
    image_b64: Optional[str]  # Image → data-url; для прочих None
    area_px: Optional[int] # Image → w*h (для фильтра мелких картинок перед VLM)
    # --- схема v2: геометрия и служебные сигналы layout-модели ---
    bbox: Optional[List[float]]      # [x0, y0, x1, y1] в координатах Unstructured
    page_width: Optional[int]        # layout_width страницы (например, 4686)
    page_height: Optional[int]       # layout_height страницы (например, 2619)
    detection_class_prob: Optional[float]  # уверенность layout-модели (0..1)
    image_path: Optional[str]        # вырез figure-*.jpg относительно корня проекта
```

**Правила нормализации (реализованные):**
- `Header`/`Footer` со службным текстом (номера страниц, реквизиты) сохраняются,
  но потребитель (`chunk_enricher`) отбрасывает их по `type`.
- `Text` и `NarrativeText` НЕ смешиваем: сохраняем исходный тип Unstructured —
  разделение делается на этапе чанкования.
- `element_id` стабилен: индекс = порядковый номер в файле парсинга этого PDF
  (все элементы одного файла разбираются за один проход; сквозной индекс через батчи).
- `image_b64` храним только если `area_px >= 200*200` (фильтр мелких значков).
  `area_px` вычисляется по `coordinates.points`/bbox, fallback — через PIL.
- **Payload картинок (схема v2):** `extract_image_block_to_payload` управляется
  настройкой `store_image_payload` (по умолчанию `True`). Для полного корпуса
  рекомендуется `False`: base64 раздувает JSON (до ~1 МБ на дизайнерскую
  страницу), а jpg-вырезы всё равно пишутся в `data/figures/{doc_id}/`, и в
  элементе остаётся `image_path`. VLM читает файл по `image_path`.
- **Геометрия (схема v2):** `bbox`/`page_width`/`page_height` берутся из
  `metadata.coordinates` (`coordinates.points` + `coordinates.to_dict()` /
  `system.layout_*`). Это позволяет downstream-этапам использовать высоту
  блока как сигнал уровня заголовка и фильтровать caps-джанк инфографики без
  повторного парсинга.
- **Версия схемы:** `PARSE_SCHEMA_VERSION = 2`; подпись парсинга
  (`parse_signature`: schema_version, strategy, ocr_languages, min_image_area,
  store_image_payload, infer_table_structure) пишется в `_stats.json`.
  Resume пропускает файл только при совпадении `source_sha256` **и** подписи;
  `--force` перепарсивает принудительно, старые артефакты уходят в
  `data/elements/backup/`.

### 2.2. doc_id и год

```python
def make_doc_id(filename: str) -> str:
    """Копия Сбер 2015.pdf -> sber_2015; год = последняя группа \d{4} в имени файла."""
```

Год только из **имени файла** (не из текста внутри — метаданные unreliable).
2021 в данных нет — это валидная ситуация, список doc_id собирается из файлов.

### 2.3. Статистика (`{doc_id}_stats.json`)

```json
{
  "doc_id": "sber_2015",
  "source_url": "data/raw/Копия Сбер 2015.pdf",
  "source_sha256": "…",
  "year": 2015,
  "report_period": "2015",
  "n_pages": 74,
  "n_elements": 1143,
  "by_type": {"Title": 100, "NarrativeText": 421, "Text": 279, "ListItem": 133,
              "Image": 103, "Header": 70, "Table": 25, "FigureCaption": 12},
  "skipped_page_ranges": [],
  "parse_seconds": 0.0,
  "strategy": "hi_res",
  "ocr_languages": ["rus", "eng"],
  "parser_version": "unstructured-0.27.36",
  "created_at": "2026-11-06T10:00:00+03:00"
}
```

- `source_sha256` — по файлу PDF (курсовой контракт чанка требует его в будущих чанках).
- `n_elements`/`by_type` для 2015 в примере — реальные числа smoke-test (весь 74-страничный документ).
- `skipped_page_ranges` — интервалы неспарсенных батчей (после retry), `[]` при полном успехе.

---

## 3. Интерфейс

```python
# src/ingestion/pdf_parser.py (реализовано)

def parse_pdf(
    pdf_path: Path,
    *,
    settings: Optional[ParserSettings] = None,   # pydantic-settings, см. §9
    resume: bool = True,                         # докончить/пропустить уже спарсенный файл
) -> Optional[ParseResult]:
    """None — файл уже спарсен (resume сработал)."""
```

**Параметры парсинга (из `ParserSettings`):** `parse_strategy` ("hi_res"|"fast"),
`parse_page_batch` (страниц за один вызов partition), `parse_min_image_area` (200×200),
`ocr_languages` ("rus","eng"), `data_elements_dir`, `data_figures_dir`,
`tessdata_prefix`.

### 3.1. Батчевый протокол страниц (fix v1.1 — заменяет `pages=`)

На unstructured 0.27.36 проверено: `partition_pdf(pages=[a,b])` **игнорируется**
(параметр не входит в сигнатуру и молча уходит в `**kwargs`; парсятся ВСЕ страницы
документа). Поэтому батчи строятся курсовым протоколом Занятия 2:

```python
# для батча [start..end]:
buffer = io.BytesIO()
writer = pypdf.PdfWriter()
for i in range(start - 1, end):
    writer.add_page(reader.pages[i])
writer.write(buffer)
buffer.seek(0)
elements = partition_pdf(
    file=buffer,
    strategy="hi_res",
    languages=["rus", "eng"],
    infer_table_structure=True,          # таблицы → text_as_html
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True, # ← metadata.image_base64 (без него None)
    starting_page_number=start,          # ← абсолютная нумерация страниц
    extract_image_block_output_dir=figures_dir,  # побочный jpg-вырез
)
```

**Инварианты:**
- Один PDF = один JSON-файл артефакта. Частичная добыча не публикуется.
- Номер страницы валидируется интервалом батча: элемент вне `[start..end]` →
  `ValueError` («элемент потерял номер исходной страницы») — протокол Занятия 2.
- Свойства картинок: `area_px` из `coordinates` или из PIL-декодирования.

### 3.2. Config и резервные проверки

```python
def parse_all(raw_dir: Path, *, settings=None, resume=True) -> list[ParseResult]:
    """Все PDF из data/raw (sorted), изоляция сбоя одного файла, сводный лог."""
```

---

## 4. Отказоустойчивость и язык

| Ситуация | Поведение |
|---|---|
| PDF уже спарсен (`resume=True`, stats есть и source_sha256 совпал) | Пропуск без пересоздания |
| source_sha256 изменился (файл заменили) | Полный пере-парс, старый артефакт в `data/elements/backup/` |
| OCR-доступность tesseract | Проверка `shutil.which("tesseract")` (+ явный каталог) на старте; при отсутствии → `RuntimeError` с инструкцией установки |
| В TESSDATA нет языкового файла rus | Проверка `{lang}.traineddata` при старте (`preflight_check_ocr`) |
| Сбой одного батча | 1 повторная попытка; второй сбой → батч помечается в `skipped_page_ranges`, парсинг идёт дальше (без паузы) |
| Сбой одного файла в `parse_all` | Лог-ошибка, остальные файлы обрабатываются; `ParserEnvironmentError` всплывает сразу |

**Среда исполнения (зафиксировать значения):**
`tesseract` в PATH (у нас: `C:\Program Files\Tesseract-OCR`), env `TESSDATA_PREFIX` →
`<project>/tessdata` (rus, eng, osd traineddata в комплекте репозитория).

---

## 5. Логирование и прогресс

- Прогресс — logging-строки (формат фактически реализован):
  `[sber_2015] batch 7/12 | pages 121-140 | 47.3s | 214 elements`.
- Финал: `[done] {doc_id}: страниц N, элементов M, время T` + таблица `--stats`.
- Уровень INFO по умолчанию через `logging` (CLI настраивает basicConfig).
- `warnings` pdfminer («Cannot set non-stroke color») глушатся `filterwarnings` —
  это шум от векторной графики, на результат не влияет.
- tqdm-бар и файл-лог `logs/parser_{doc_id}.log` — НЕ реализовано (опция, не контракт).

---

## 6. Performance-оценка (v1.1, бенчмарк 7.4 снят)

Фактическая точка: smoke hi_res на **всем 74-страничном документе 2015** → 1143 элементов
(NarrativeText 421, Text 279, ListItem 133, Image 103, Title 100, Header 70,
Table 25, FigureCaption 12).

Бенчмарк steady-state (задача 7.4, отчёт 2015):

| Замер | Значение |
|---|---|
| Warmup 2 стр. (загрузка detectron2/OCR-моделей) | 41.4 с, 20 эл. |
| Steady hi_res 5 стр. (стр. 6–10) | 41.4 с, 68 эл. → **8.3 с/стр.** |
| Элементов/страницу (hi_res вырезка) | ~13.6 |
| Оценка корпуса 2581 стр. (hi_res тотально) | **≈ 5.9 ч** |

Замер через `_parse_batch` (протокол §3.1 v1.1); тессеракт-агент дефолтный.
Скорость по страницам ровная, но OCR-heavy страницы (сканы таблиц) могут
замедлить отдельные батчи — резерв ×1.5 → план 6–9 ч. Прогон fast-стратегии
существенно быстрее (только pdfminer/pypdfium2).

Кросс-файловый hi_res-смоук (проверка протокола на новом venv 3.11 /
unstructured 0.27.37; по 10 стр., задача 7.3):

| Файл | Элементов | Время | стр/с | Image с b64 | Table с html |
|---|---|---|---|---|---|
| 2016 | 150 | 115 с | 0.87 | 14/14 | 8/8 |
| 2019 | 289 | 129 с | 0.78 | 16/16 | 0 (таблиц нет) |
| 2023 | 103 | 63 с | 1.59 | 9/9 | 1/1 |
| 2024 | 137 | 48 с | 2.08 | 16/16 | 0 (таблиц нет) |

page_number всех элементов строго в [1..10] на всех файлах (протокол держится);
payload-изображения и html таблиц корректны. Замечание: у Title на всех 4 файлах
`category_depth = None` — глубину разделов придётся выстраивать на слое
`structure_builder.py` (шаг 2 плана), unstructured её здесь не даёт. Первый замер
(2016) включает прогрев моделей; атмосферные 1.6–2.1 стр/с на 2023/2024 показывают
реальную скорость после загрузки.

Правила при запуске полного прогона:
1. Полное время = 2581 × 8.3 с/стр. ≈ 5.9 ч hi_res; при худших страницах — запускать фоном на ночь.
2. Стратегия hi_res обязательна не для всех документов: fast достаточно для
   текст-доминантных годов; hi_res — для годов с диаграммами/таблицами (решать по smoke by_type).
3. Внутри фаила — **без параллельности по страницам** (OCR-агент и layout
   надёжнее вести последовательно; курсовой протокол того не требует).

Учтём: настройку OCR-агента unstructured (UNSTRUCTURED_OCR_AGENT, по умолчанию
unstructured.pytesseract) НЕ меняем — не проверено, чревато изменением качества OCR.

---

## 7. Артефакты и порядок файлов

```
data/
  elements/                     # артефакты парсера (промежуточный слой)
    sber_2015_elements.json     # list[RawElement], UTF-8, json.dump(ensure_ascii=False)
    sber_2015_stats.json
    backup/                     # старые артефакты при изменении sha256
  figures/                      # побочные вырезы Unstructured (figure-*.jpg)
    sber_2015/                  # по doc_id; вне контракта, для диагностики
  raw/                          # 9 PDF, только для чтения
```

- Дублирующий каталог `metadata/` из v1.0 отменён — не реализован и не нужен.
- Файловый контракт: `structure_builder.py` читает ТОЛЬКО `_elements.json`
  (не модули Unstructured) — нижний слой заменяем на «один API».
- Потребители картинок (`visual_describer.py`) предпочитают `image_path` на
  сохранённый вырез; `image_b64` поддерживается для артефактов, созданных с
  `store_image_payload=True`. При отключённом payload картинка не должна молча
  считаться обработанной — записывается причина пропуска/ошибка чтения файла.

---

## 8. Тестирование

Реализовано (`tests/`), `pytest -q` зелёный (42 теста):

| Тест | Проверяет |
|---|---|
| `test_models.py` | `RawElement`/`ParseResult` сериализация → JSON → обратно |
| `test_doc_id.py` | `make_doc_id()` на 9 реальных именах файлов → 9 уникальных doc_id, год из имени |
| `test_preflight.py` | нет tesseract / нет traineddata → `ParserEnvironmentError` с инструкцией |
| `test_image_filter.py` | мелкие картинки → `image_b64=None`, `area_px` сохраняется |
| `test_parse_pdf.py` | конвертация батчей, resume, backup при смене sha256, skipped_page_ranges |
| `test_cli.py` | режимы `--pdf/--all/--stats/--smoke/--fast-preview`, приоритет CLI над env |

Добавлено после фикса протокола §3.1 (v1.1, задачи 7.1–7.3):
- `test_parse_pdf.py::test_build_page_slice_cuts_requested_range` — вырезка pypdf отдаёт ровно запрошенный диапазон, ошибки «вне документа»;
- `test_parse_pdf.py::test_page_number_read_from_metadata_page_number` — abs. номер страницы читается из `metadata.page_number` (основной канал real-unstructured) и из `coordinates` (fallback);
- `test_parse_pdf.py::test_parse_pdf_batched` — контракт фиксирует вырезки батчей `[(1,2),(3,1)]` (file-буфер + starting_page_number);
- `test_parse_pdf.py::test_page_label_filled_from_pdf` — курс-поле `page_label` заполняется;
- `test_cli.py` — fake-заглушки под новый протокол (slices_seen).

Интегр. дым-проверка на реальной unstructured 0.27.36 (не pytest): fast срезы 21–22 и 40–42
2015 — page_number строго внутри интервала; hi_res стр. 2–3 — 31 эл., 2 Image с base64,
2 Table с html. Полный smoke 2015 (1143 ±10%) не перезапускался.

---

## 9. CLI (реализовано)

```bash
# Один файл
python -m src.ingestion --pdf "data/raw/Копия Сбер 2023.pdf"

# Все файлы с resume
python -m src.ingestion --all

# Быстрое превью (strategy=fast, без OCR), без артефактов
python -m src.ingestion --pdf "data/raw/Копия Сбер 2023.pdf" --fast-preview

# Smoke: первые N страниц первого/указанного PDF, без артефактов
python -m src.ingestion --smoke [--doc 2015] [--pages 3]

# Статистика по артефактам
python -m src.ingestion --stats

# Переопределение размера батча (CLI > env)
python -m src.ingestion --all --batch 10
```

Все параметры — `pydantic-settings` (`src/core/config.py`, env + `.env`,
поля `PARSE_STRATEGY`, `PARSE_PAGE_BATCH`, `PARSE_MIN_IMAGE_AREA`,
`OCR_LANGUAGES`, пути). CLI-флаги имеют приоритет над env
(`--batch`, `--fast-preview` → `settings.model_copy(update=…)`).

---

## 10. Связь с интерфейсами соседей (контракты)

| Потребитель | Что берёт из артефактов парсера |
|---|---|
| `structure_builder.py` | `RawElement(type="Title")` → дерево из `category_depth`, page ranges разделов |
| `chunk_enricher.py` | Все `RawElement`: `element_id` → `chunk.element_ids`, `source_sha256`, `page_label`, страницу из `page` |
| `table_structurer.py` | `RawElement(type="Table")` → поле `html` (text_as_html) + `page` |
| `visual_describer.py` | `RawElement(type="Image")` → `image_path` или `image_b64`, `page`, `area_px`; payload не обязателен (§2.1) |
| `graph/` (DocumentNode) | Page ranges разделов из `structure_builder` |
| eval | `stats.n_elements`, `stats.by_type` — для отчётов о составе корпуса |

---

## 11. Критерии приёмки парсера

- [ ] 9 артефактов `{doc_id}_elements.json` созданы, JSON-валидные, UTF-8
- [ ] Для каждого: `n_pages` совпадает с `len(reader.pages)` (pypdf) — фактические 74/310/198/213/300/323/275/402/486
- [ ] `by_type` согласован: сумма == n_elements; 2015 ≈ фактическому smoke (Title 100, Table 25, ...)
- [ ] Все элементы имеют `page` в [1..n_pages], `doc_id` консистентен, `page_label` либо строка либо null
- [ ] `[Title]` есть с `category_depth >= 0` (структура строится хотя бы из части заголовков)
- [ ] `source_sha256` воспроизводим при пересчёте
- [ ] Повторный запуск не пересоздаёт артефакты (resume)
- [ ] Прогон smoke соответствует базовому результату тест-скрипта (±10%)
- [ ] Батчебые проверки: элементы батча внутри [start..end] (после fix §3.1)
- [ ] Код покрыт docstrings (Google-style), типизация
- [ ] Курсовой код не затронут

---

## 12. Порядок работы (обновлён)

1. ~~**Fix батчей (§3.1):** вырезка страниц через pypdf + `starting_page_number`
   и `extract_image_block_to_payload=True`~~ — **ВЫПОЛНЕНО (задачи 7.1–7.3):**
   `build_page_slice` + новый `_parse_batch`; юнит-тесты + интегр. дым на 2015
   (fast: нумерация внутри интервала; hi_res: image_base64 ≠ null, table html)
2. Прогон 2023 целиком (402 стр.) → ревью артефакта глазами (Title/Table, страницы, html)
3. ~~Benchmark~~ — **снят (7.4):** hi_res steady 8.3 с/стр. → ≈ 5.9 ч на 2581 стр. (§6)
4. Прогон оставшихся 8 в фоне → `--stats` (9 doc_id, by_type согласованы)
5. Полная статистика → `openspec validate`, тег `lesson-01-parser-done`

**Оценка трудоёмкости:** fix батчей + smoke ≈ 1–2 ч; полный прогон — часы/ночи CPU (см. §6).

OpenSpec change `add-pdf-parser`: задачи 1–5 фактически выполнены
(задача 5.2 — замена пакета из `requirements.txt`); критический fix §3.1 внедрён
(7.1–7.4, см. `tasks.md` §7); оставшиеся — полный прогон корпуса (6.1) и 2023-ревью.
