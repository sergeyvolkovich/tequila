# Инструменты эмпирического анализа формата (cleanroom)

Скрипты, которыми поведение TwinVQ исследовалось на живых объектах —
декодере FFmpeg и 218 эталонных битстримах NTT TestEnc — до написания
C++-энкодера. Результаты легли в основу docs/ARCHITECTURE.md (спецификация)
и docs/ENCODE_PIPELINE.md (модель энкодера NTT).

## Скрипты

* `exp_probe.py` — эмпирическое зондирование декодера FFmpeg
  **синтетическими** битстримами (полный контроль над каждым полем кадра).
  Эксперименты E0–E7: priming/латентность, базовый сигнал кодвектора #0,
  PPC-развёртки p_coef×g_coef, bark-полосы, LSP-огибающая, карта
  распространения битовых ошибок, wtype-последовательности, контейнер
  (metadata, обрезка DATA, отказ wtype=8).
* `real_stats.py` — массовый разбор **эталонных** NTT-битстримов:
  wtype-гистограммы, gain/sub_gain, PPC-активность, bark_use_hist,
  lpc_hist, seg-SNR декодирования против исходных WAV.
* `bit_budget.py` — битовый бюджет кадров всех 9 режимов × 2 канала ×
  3 ftype; аудит: сумма компонент == frame_bit_len.

## Результаты (results/)

* `vqf_probe_results.json` — вывод E0–E7.
* `encoder_probe.json` — калибровка MDCT-мишени, послойные корреляции,
  PPC-гребёнка, грамматика wtype, параметры кодбуков (E1–E3 из
  ENCODE_PIPELINE.md).
* `bit_budget.json` — полный бюджет (54 комбинации).
* `ntt_real_stats.json` — НЕ в репо (2.6 МБ сырых пер-кадровых данных):
  регенерируется `real_stats.py` по локальному корпусу.

## Запуск

Зависимости: python3 + numpy, системный ffmpeg/ffprobe (в PATH);
`vqflib.py`/`twinlayers.py` импортируются из `python/reference/`.

```bash
python3 bit_budget.py                 # самодостаточен (таблицы режимов)

# Пути к корпусу патчатся переменными окружения:
TVQ_CORPUS=/path/to/encoded_all/tvq_cleanroom \
TVQ_SIGNALS=/path/to/signals \
python3 real_stats.py                 # нужны 218 .vqf + исходные WAV

python3 exp_probe.py                  # пишет results/vqf_probe_results.json
```

Корпус эталонных битстримов NTT TestEnc (218 .vqf) и исходные WAV —
из комплекта tvq_cleanroom; в репо не входят (см. signals/ и
python/stage4_report.json для состава дифтестового поднабора).
