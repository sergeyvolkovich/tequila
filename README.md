# tequila

Cleanroom-реализация кодировщика **TwinVQ** (Transform-domain Weighted Interleave
Vector Quantization, NTT Lab, 1994–2000) — аудиокодека семейства VQF/MPEG-4
TwinVQ. Проект пишется с нуля по открытым источникам знаний (FFmpeg как
поведенческий эталон и источник таблиц данных, научные публикации NTT,
поведенческое тестирование против оригинальных инструментов NTT).

## Цель

Энкодер, производящий `.vqf`-битстримы, которые корректно декодируются
FFmpeg (`libavformat/vqf.c` + `libavcodec/twinvqdec.c`) с качеством,
сопоставимым с оригинальным NTT TestEnc.

Поддерживаемые режимы (те же, что в корпусе верификации):

| Режим | Частота, кГц | Каналы | Битрейт, кбит/с |
|-------|--------------|--------|-----------------|
| 8m8   | 8            | 1      | 8               |
| 11m8  | 11.025       | 1      | 8               |
| 11m10 | 11.025       | 1      | 10              |
| 22m20 | 22.05        | 1      | 20              |
| 22m32 | 22.05        | 1      | 32              |
| 44m48 | 44.1         | 1      | 48              |
| 44s96 | 44.1         | 2      | 96              |

## Архитектура

Декодер восстанавливает спектр кадра как трёхслойную модель
(эмпирически верифицировано по 218 эталонам NTT, см.
[docs/ENCODE_PIPELINE.md](docs/ENCODE_PIPELINE.md)):

```
X̂ = ( VQ · Bark-огибающая · gain  +  PPC ) · LPC-огибающая

WAV → кадрирование (2 кадра pre-roll) → MDCT с синусным окном (по wtype)
    → LPC-анализ → LSP → split-VQ (+MA-предсказание)        [слой тембра]
    → обратный LPC-фильтр
    → (LONG) PPC: гармоническая гребёнка f0 ∈ [100, 600] Гц   [пич-структура]
    → Bark-огибающая подкадров + gain/sub_gain (mulaw)        [слой формы]
    → взвешенное перемежение → конъюгатный VQ (93% битов)     [остаток]
    → мультиплексирование полей → битовая упаковка → .vqf
```

Подробности: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — спецификация
(контейнер, режимы, wtype, MDCT, бюджет, перестановки);
[docs/ENCODE_PIPELINE.md](docs/ENCODE_PIPELINE.md) — как кодирует эталонный
энкодер NTT (мишень MDCT, слои, грамматика окон, PPC, усушка VQ).
Журнал выполнения шагов: [docs/PROGRESS.md](docs/PROGRESS.md).

## Сборка и тесты

```bash
cargo test               # все юнит-тесты (бюджет — против bit_budget.json,
                         #   golden PCM/MDCT — против python-оракула)
cargo build --release    # библиотека + CLI (target/release/tequila)
python3 tools/gen_vendor_rs.py   # регенерация vendored-кодов (только при обновлении FFmpeg-данных)
```

Текущее состояние: полный конвейер E2E (WAV → tequila → .vqf → ffmpeg) —
битовый I/O, 9 режимов, контейнер/кадры, MDCT/окна, декодер-порт,
LPC/LSP/bark/PPC/VQ-квантователи, мультиплексор, CLI, метаданные-теги.

## Использование (CLI)

```bash
tequila encode in.wav out.vqf [--mode 44_48] [--no-ppc] [--kappa 1.0]
                [--psy [--psy-beta F] [--psy-bark-beta F] [--psy-pre-mask F] …]
                [--title S] [--author S] [--comment S] [--copyright S]
                [--genre S] … (18 тегов) [--tag CHUNK=S]
tequila decode in.vqf out.wav
tequila info in.vqf
tequila bench [in.wav] [--mode 44_48]   # 3 прогона, RT-фактор
```

`--psy` — психоакустическая модель (docs/PSY.md): маскировка (spreading +
тональность + ATH + pre/post-masking) переставляет ошибку квантования в
неслышимые области при неизменном CBR-бюджете, адаптивный κ выравнивает
сладкую зону кодбуков по слышимой части остатка. Корпусная валидация
(python/analysis/psy_eval.py, PEAQ ODG через EAQUAL): seg-SNR +4.2 дБ,
доля слышимых over-mask бинов ×0.45, PEAQ ODG −3.34→−3.24. Битстрим-формат
и совместимость с FFmpeg не меняются.

Производительность (R14): MDCT/IMDCT — O(B·log B) через rustfft (FFT 4B,
SIMD внутри крейта), VQ-поиск — кэш кросс-матриц кодбуков + отсев строк,
гриды/геометрия PPC — кэши на поток, корни LSP — парная схема с
рекуррентными вращениями. Корпус 240 с аудио (9 режимов) кодируется
за ~21 с (**~12× realtime** на 2 ядрах Xeon; 44s96 стерео — ~7-8×).
`TEQUILA_PROFILE=1 tequila encode …` печатает разбивку по стадиям.
Битстримы байт-в-байт совпадают с до-оптимизационной версией
(проверено на корпусе; качество — e2e_bench.py: 12.08 дБ против 10.08 у
NTT TestEnc).

Теги пишутся чанками после COMM (порядок NAME, COMT, AUTH, «(c) », … — как у
NTT TestEnc; все 18 из `vqf_metadata_conv` FFmpeg). Нечётные длины
допустимы, выравнивания нет; `--tag CHUNK=S` — произвольный 4-байтовый
чанк (кроме служебных COMM/DSIZ/DATA). Повтор флага перекрывается
последним значением. `header_size` пересчитывается автоматически как
Σ(8+len). Пример:

```bash
tequila encode song.wav song.vqf --mode 44_48 \
    --title "You Shot A Hole In My Soul" --author "C.C.Catch" \
    --comment "Best music from 90s" --copyright "SonyMusics Production"
ffprobe -show_format song.vqf   # title/comment/author/copyright видны FFmpeg
```

Зависимости: rustc/cargo (edition 2021); из внешних крейтов — только
`rustfft` (чистая математика FFT, MIT OR Apache-2.0, кодек-логики не
содержит — чистая комната не затрагивается).
Для регенерации vendored-данных — python3; для верификации — ffmpeg/ffprobe
+ numpy (`python/reference/` — бит-точный оракул, corr = 1.0 против ffmpeg;
`python/analysis/` — зондирование/статистика).

## Лицензия и правовой статус

- Собственный код проекта — MIT (см. LICENSE).
- Таблицы данных (кодбуки VQ, параметры режимов) портированы из
  **FFmpeg** (`libavcodec/twinvqdata.h`, `twinvq.c`) — LGPL-2.1-or-later;
  см. `third_party/ffmpeg/` и уведомления в файлах. Комбинация с этими
  данными наследует LGPL-условия для соответствующих частей.
- Патентный анализ: [docs/PATENTS.md](docs/PATENTS.md). Ключевые патенты
  NTT (например, US 5,684,920, приоритет 1994-03-17) истекли в 2015.

## Структура

```
docs/            документация: ARCHITECTURE (спецификация), ENCODE_PIPELINE
                 (модель энкодера NTT), PSY (психоакустика --psy),
                 PATENTS, PROGRESS (журнал шагов)
src/             Rust-реализация (cleanroom)
  bitstream.rs    побитовый I/O MSB-first (кадры вплотную, битовый уровень)
  modes.rs        FrameType, find_mode, wtype-маппинги
  layout.rs       битовый бюджет кадра (init_bitstream_params)
  codec.rs        StreamInfo (режим ← COMM: каналы/битрейт/rate_flag)
  frame.rs        FrameData: parse/serialize — порядок полей кадра
  container.rs    контейнер VQF: чанки, теги (TAG_TABLE), writer/парсер
  dsp.rs          перестановки, окна, forward/compact MDCT
  decode.rs       точный порт тракта декодера (VQ/bark/PPC/LSP/TDAC)
  analysis.rs     LPC-анализ: автокорреляция, Левинсон, LSP-корни
  psycho.rs       психоакустическая модель (--psy): маскировка, ATH,
                  post-masking; перцептивные веса VQ/bark (docs/PSY.md)
  quantize.rs     LSP split-VQ+MA, bark, gain mulaw, VQ, PPC-гриды
  encoder.rs      конвейер кадра: мишень MDCT → слои → мультиплексор
  wav.rs          минималистичный WAV-ридер/райтер
  main.rs         CLI: encode / decode / info
  vendor/         данные FFmpeg (LGPL): tables.rs + generated/*.rs
  lib.rs
 tools/           gen_vendor_rs.py — генератор vendored-данных из FFmpeg
python/reference/ бит-точный python-декодер-оракул (corr=1.0 vs ffmpeg)
python/analysis/ инструменты эмпирического анализа формата (E0–E7,
                 массовая статистика 218 эталонов, битовый бюджет,
                 psy_eval.py — корпусная оценка режима --psy)
third_party/     эталонные исходники FFmpeg (сверка) + eaqual/ (PEAQ-Basic,
                 внешняя валидация --psy, см. eaqual/BUILD.md)
signals/         тестовый корпус WAV (генерируется скриптом)
```
