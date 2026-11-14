# Tasks

## 1. Ядро моделей и конфигурации

- [x] 1.1 Создать `src/core/models.py`: dataclass `RawElement` (element_id, doc_id, type, text, page, category_depth, html, image_b64, area_px) и `ParseResult(elements, stats)`; проверить цикл JSON-serialisation туда-обратно командой округления `pytest -q tests/test_models.py`
- [x] 1.2 Создать `src/core/config.py` — наследник курсовой `Settings` с полями PARSE_STRATEGY, PARSE_PAGE_BATCH=20, PARSER_MIN_IMAGE_AREA=40000, OCR_LANGUAGES=["rus","eng"]; проверить переопределение через env (`PARSE_PAGE_BATCH=5`)

## 2. Утилиты парсера

- [x] 2.1 Реализовать `make_doc_id(filename) -> str` (regex последней 4-цифровой группы) с тестом на 9 реальных имён файлов: `pytest -q tests/test_doc_id.py` даёт 9 уникальных doc_id
- [x] 2.2 Реализовать расчёт `source_sha256` и подсчёт n_pages через pypdf; тест: sha256 вручную исполненная на маленьком файле совпадает, n_pages 2015 == 222 ⚠️ пересмотр v1.1: фактически 2015 = **74 стр.** (см. spec_pdf_parser.md §1), сводный факт — 2581 стр. на 9 файлов
- [x] 2.3 Реализовать preflight-проверку среды: `shutil.which("tesseract")` + наличие `{lang}.traineddata` в `TESSDATA_PREFIX`; юнит-тест с подставным PATH/TESSDATA_PREFIX проверяет `RuntimeError` с подсказкой по установке
- [x] 2.4 Реализовать фильтр мелких изображений (площадь < min_image_area → image_b64=None, area_px сохраняется); тест-фикстура: 40×40 → null, 800×600 → строка

## 3. Парсинг

- [x] 3.1 Реализовать `parse_pdf(pdf_path, *, strategy, languages, ..., batch_size, resume)`: батчи по 20 стр через partition_pdf(pages=[a,b]), глушение pdfminer-предупреждений `filterwarnings`, снижение повторное батча 1 раз; smoke: 3 страницы 2015 ≥1000 элементов (как в scripts/test_hi_res.py)
- [x] 3.2 Реализовать resume: пропуск при совпадающем `source_sha256` в `data/elements/`; тест двукратного запуска на фикстуре (малый PDF-файл) — второй прогон <5с, file mtime unchanged
- [x] 3.3 Реализовать реакцию на изменённый source: пере-парс + перенос старых файлов в `data/elements/backup/`; тест: подмена sha256-фикстуры вызывает пересоздание и backup
- [x] 3.4 Реализовать `parse_all(raw_dir)`: список PDF из data/raw, парсинг недостающих, сводный лог; тест на подменных каталогах (mock) пропускает спарсенные и обрабатывает нοвые

## 4. Сериализация и статистика

- [x] 4.1 Реализовать запись артефактов `data/elements/{doc_id}_elements.json` + `_stats.json` (UTF-8, ensure_ascii=False, atomic через tmp+rename); тест: JSON читаем, сумма by_type == n_elements, page в диапазоне
- [x] 4.2 Реализовать сводку прогресса: tqdm по батчам, формат `[sber_2015] batch 7/12 | pages 121-140 | 47.3s | 214 elements`; caplog-тест проверяет формат строки
- [x] 4.3 Реализовать отчёт о непройденных батчах в stats: поле `skipped_page_ranges`; тест: mock сбоя batch перезаписывает «121-140»

## 5. CLI

- [x] 5.1 Реализовать `__main__.py` с режимами `--pdf`, `--all`, `--stats`, `--fast-preview`, `--smoke` и приоритетом CLI-флагов над env; тесты каждого режима на tmp-каталоге
- [x] 5.2 Добавить `unstructured-inference>=1.6` в `requirements.txt`; проверить чистую установку из requirements в свежий venv или `pip check`

## 6. Интеграционный прогон корпуса

- [ ] 6.1 Полный прогон 9 отчётов: `python -m src.ingestion.pdf_parser --all` (фоновый, время — по бенчмарку 7.4: ≈ 5.9 ч hi_res на 2581 стр.); контроль консистенции: каждый артефакт имеет by_type с Title ≥ 50, Table ≥ 20 на документ, `--stats` показывает 9 doc_id
- [ ] 6.2 Ревью артефакта 2023 глазами: правильные страницы, заголовки русские, таблицы содержат html; замечания записать в `data/elements/REVIEW_notes.md`
- [ ] 6.3 Прогнать `python -m pytest -q` целиком (все тесты групп 1–5 зелёные) и `openspec validate --all --strict`

## 7. Критические фиксы до полного прогона (ревизия spec_pdf_parser.md v1.1)

- [x] 7.1 Блокирует 6.1: `partition_pdf(pages=[a,b])` молча игнорируется unstructured 0.27.36 (проверено experiment'ом: полный документ, page_number 1..N) — `_parse_batch` переводится на курсовой протокол Занятия 2: вырезка интервала страниц в `io.BytesIO` через `pypdf.PdfWriter` (`build_page_slice`) + `partition_pdf(file=buffer, starting_page_number=start)`; ничего не делаем с pages=. Реализовано; интегр. проверка на 2015 (fast, срезы 21–22 и 40–42): page_number элементов строго внутри интервала; юнит-тест `test_build_page_slice_cuts_requested_range`, `test_parse_pdf_batched` фиксирует вызовы `[(1,2),(3,1)]`.
- [x] 7.2 Блокирует визуальный слой: `_parse_batch` добавляет `extract_image_block_to_payload=True` (иначе `metadata.image_base64` всегда None; карточно-дисковые вырезы в data/figures остаются побочными). Также `infer_table_structure` только для hi_res. Реализовано; hi_res-смоук 2015 стр. 2–3: 31 эл., 2/2 Image с base64, 2/2 Table с html.
- [x] 7.3 Smoke после 7.1–7.2: кросс-файловый hi_res-смоук (10-05-2026 venv, 10 стр. × 4 файла): 2016 (150 эл/115s), 2019 (289/129s), 2023 (103/63s), 2024 (137/48s) — page_number строго в [1..10] везде, Image с base64 100% (55/55), Table с html 9/9, по 0 tables на 2019/2024 (п first 10 стр. их просто нет). ⚠ Важно для structure_builder: `category_depth` у Title = None на всех 4 файлах → глубина деревьев разделов не из unstructured (решать на шаге 2 плана). Smoke 2015 целиком (1143 ± 10%) не перезапущен — считается достаточным.
- [x] 7.4 Бенчмарк производительности (hi_res, 2015): warmup 2 стр. 41.4 с (загрузка моделей), steady 5 стр. 41.4 с → 8.3 с/стр.; оценка полного корпуса 2581 стр. ≈ 5.9 ч hi_res (fast — быстрее). Задача 6.1 рекомендует full-стратегию по VLM-приоритету.

## 8. Шаг 2 плана: structure_builder v1 (дерево разделов)

- [x] 8.1 Модуль `src/ingestion/structure_builder.py`: `SectionNode`/`DocumentStructure` (§5.1 спеки: breadcrumbs, parent/children, page_start/page_end), `build_document_structure` (алгоритм стека). Тесты: breadcrumbs/parent/branch-closing/page_end/сериализация.
- [x] 8.2 Эвристика уровней вместо отсутствующего `category_depth` (смоук 7.3: depth=None на hi_res у всех файлов): нумерация «1./2.1/2.1.3» → 0/1/2 (точка обязательна после первой группы, иначе «93 АС»)— словесные метки (ГЛАВА/РАЗДЕЛ/ПРИЛОЖЕНИЕ) → 0 — caps → 0 — фолбэк «наследует последний уровень», дз ≤2 за шаг.
- [x] 8.3 TOC-калибровка (второй источник уровней, сильнее эвристики): `parse_toc_entries` (построчные TOC-строки и жадный скан слитных Table/Image «Содержания», числа ≥2000 не страницы), `toc_level_map` (TOC без нумерации = level 0), совпадение по `normalize_section_name` (дефис-переносы). Проверено на срезе 2023 стр. 9–13: 36 записей из дизайн-TOC.
- [x] 8.4 Junk-фильтры (глаз-проверки 2015 hi_res + 2023 стр. 23–32/76–85): ФИО-строки списков советов, инфографика-милстоуны («>5,7%», «93 АС», «116,5 ПБ», «GRI 3-3», «Y»), дедуп подряд («Миссия» дважды). На срезе 76–85 отфильтровано 12 из 31 Title.
- [x] 8.5 Fuzzy-матч TOC↔Title (`_best_toc_level`): точный матч через `toc_level_map`, затем подстрочный (guard ≥8 символов) — закрывает межколоночные склейки TOC-имён («Кредитные рейтинги Обзор рынка» ↔ Title «Обзор рынка»); интегрирован в `build_document_structure` (точный → fuzzy). Тесты: fuzzy-подстрока, обратное вложение, guard коротких имён. Дополнительно: «Содержание»/«Оглавление» добавлены в junk (сам TOC не должен становиться разделом). Остались caps-джанки инфографики («ЧЕЛОВЕКО- ЦЕНТРИЧНОСТЬ», «НАДЕЖНОСТЬ») — требуют bbox-сигналов, которых нет в RawElement (перенесено в 8.7), и OCR-хвосты («Прогноз на 2024 год1»).
- [x] 8.6 Артефакт `{doc_id}_sections.json` + CLI: `build_from_elements_file` (читает `*_elements.json`, сам находит TOC через `parse_toc_entries`), `save_sections` (атомарная запись `os.replace`), `ParserSettings.data_structure_dir` (`data/structure`), режим `python -m src.ingestion --structure [--doc NAME]`. Тесты: roundtrip tmp_path, FileNotFoundError, CLI-режим (успех + «нет артефактов»). Проверено реальным срезом 2023 стр. 9–13: артефакт → CLI → 7 разделов, дерево/breadcrumbs корректны.
- [x] 8.7 Схема v2: `RawElement.bbox/page_width/page_height/detection_class_prob/image_path` + извлечение из `metadata.coordinates` (points + `to_dict()`/`system.layout_*`); `image_path` хранится относительно корня проекта. Проверено на 2023 стр. 23: bbox заголовков («Обзор рынка» 148 px против обычного текста 51 px) — сигнал уровня доступен без перепарсинга. Осталось: использовать bbox в эвристике уровней и junk-фильтре, OCR-хвосты имён.

## 9. Шаг 2.5 плана: чанкинг К2 + breadcrumbs

- [x] 9.1 `src/ingestion/pdf_to_chunks.py`: перенос курсового контракта Занятия 2 на наш `RawElement` — `element_records` (пропуск Image/PageBreak/пустых, таблица обязана иметь текст и несёт html), `chunk_record` (chunk_id = `{element_id}:c{part}:{sha256[:12]}`, evidence_id = `{doc_id}:p{page:03d}`), `split_text_records`/`make_chunks` (сплиттер 800/100, группа текста закрывается на смене страницы, таблица = 1 чанк), `source_meta_from_pdf` (doc_id/title/year/sha256), `save_chunks` (атомарная запись `data/chunks/{doc_id}_chunks.json`). Добавлен `langchain-text-splitters>=1.1` в requirements.
- [x] 9.2 `src/ingestion/chunk_enricher.py`: `find_section_for_page` (последний заголовок с `page_start <= page` через bisect), `enrich_chunks` — опциональные поля `section_path`, `breadcrumbs`, `section_level`, `parent_section`; страницы до первого заголовка остаются без структурных полей (базовый контракт К2 не меняется).
- [x] 9.3 CLI `--chunks [--doc NAME] [--chunk-size N] [--chunk-overlap N]`: элементы → (sections из артефакта или строятся на ходу) → чанки → обогащение → `data/chunks/`. `ParserSettings.data_chunks_dir`; `structure_builder.load_sections` (+ `from_dict` у `SectionNode`/`DocumentStructure`). Тесты: `tests/test_chunks.py` (9 кейсов: skip-Image/ошибки корпуса, таблица-чанк, границы страниц, стабильный digest, chunk_size, breadcrumbs/level/parent, bisect, roundtrip). Проверено срезом 2023 стр. 10–12: 93 элемента → 4 чанка, у всех корректные breadcrumbs.
- [ ] 9.4 Следующее: assert-валидации корпуса (`tests/test_corpus.py`, К2) на полном наборе, выравнивание `page_label`, фильтр `structure_eligible` (К5) — при переходе к retrieval.

## 10. Схема артефакта v2 и управление resume

- [x] 10.1 `store_image_payload` в `ParserSettings` + CLI `--no-image-payload`: base64 не пишется в JSON, jpg-вырезы остаются в `data/figures/{doc_id}`, `image_path` сохраняется. Замер на 2023: срез 3 стр. весил 1.17 МБ, из них 1.14 МБ — base64; для полного корпуса payload отключён.
- [x] 10.2 `PARSE_SCHEMA_VERSION = 2` + `parse_signature` в `_stats.json` (schema_version, strategy, ocr_languages, min_image_area, store_image_payload, infer_table_structure). Resume проверяет sha256 **и** подпись; при расхождении — перепарс с backup.
- [x] 10.3 CLI `--force`: принудительный перепарс с переносом старых артефактов в `data/elements/backup/`. Тесты: `test_signature_change_reparses_with_backup`, `test_force_reparses_actual_artifact`, `test_no_image_payload_setting`, `test_image_element_keeps_bbox_path_and_prob`.
- [x] 10.4 Быстрая межгодовая выборка `--sample`: `select_sample_pages` (TOC + квартили тела), `pages_to_ranges`, `parse_sample` пишет только в `data/samples` (elements/stats/sections/figures), не трогая канонические артефакты и resume. Тесты: выбор страниц, группировка диапазонов, изоляция samples, CLI-режим. Полный прогон 2023 отложен до валидации выборки.
- [x] 10.5 Багфикс backup: `_elements.json` и `_stats.json` имеют одинаковый суффикс, из-за чего второй файл затирал первый. Имя теперь `{stem}_{stamp}.json`; тест `test_backup_keeps_elements_and_stats_files`.
- [x] 10.6 Прогон `--sample` по 9 отчётам (`data/samples`, 10 стр./документ) + непрерывные срезы 1–60 для 2022/2023 (`data/samples_contig`). Итог: парсер/таблицы/картинки 100% (51/51 Table с HTML, 252/252 Image с path+bbox); TOC после v2: 2015 7, 2016 26, 2017 29, 2018 102, 2019 72, 2020 55, 2022 17, 2023 45 (на 1–60), 2024 — закладки 106. Выявлено: дерево из сырых Title негодно (2022 — 180 «разделов» на 60 стр.).

## 11. Structure v2: скелет из закладок/TOC вместо сырых Title

- [x] 11.1 `_skeleton_entries`: основной источник — закладки PDF; при их отсутствии — TOC; сырые Title только фолбэк при <5 записях скелета. Для 2018 (закладки начинаются со стр. 168) ранние TOC-записи добавляются перед outline.
- [x] 11.2 TOC v2: заголовки «Содержание/Оглавление/CONTENTS», окно `header + 1` страница (тело больше не попадает), HTML-строки таблиц, ведущие/хвостовые/мульти-колоночные страницы, склейка имени по двум ячейкам («Обращение…» | «Правления … 23»), снятие хвостового номера, фильтр OCR-лидеров (`или`, `лилии`, `ииии`) и повторов букв.
- [x] 11.3 TOC без страниц (2017): запись сохраняется с `page=None` и разрешается по одноимённому Title в теле (`title_index`: точный → fuzzy); имя заменяется полным вариантом из тела. Проверка на непрерывном срезе 2016/2017 стр. 1–45.
- [x] 11.4 Дедуп: гомоглифы (`B2B`/`В2В`), похожие имена на соседних страницах, сильно похожие имена глобально; Title со страниц оглавления исключены из дерева; фолбэк уровня — sibling, а не ложный ребёнок.
- [x] 11.5 Тесты: `test_toc_combines_split_name_across_cells`, `test_toc_does_not_capture_body_pages`, `test_toc_pages_resolved_from_body_titles`, `test_toc_header_variants_and_html_rows`, `test_toc_pages_are_not_sections`, `test_bbox_height_assigns_heading_levels`, junk-фильтры. Всего 191 зелёный.
- [ ] 11.6 Осталось: для 2016/2017 решить, использовать ли нумерованные части (`01.`–`10.`) как level 0 и Title тела как level 1; для 2015 TOC даёт 3–4 записи — нужен ручной/LLM-override либо fallback по Title. Это не блокирует retrieval на 2020–2024.
