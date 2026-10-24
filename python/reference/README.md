# Reference-декодер TwinVQ (Python, cleanroom)

Бит-точный порт декодирующего тракта FFmpeg (twinvq.c / twinvqdec.c / vqf.c),
использованный для обратного инжиниринга энкодера NTT (см. docs/ENCODE_PIPELINE.md).

* `vqflib.py`      — контейнер VQF, побитовый парсер/райтер кадров, битовый бюджет.
* `twinlayers.py`  — перестановки, деквантование, Bark/PPC/LSP-слои, IMDCT+TDAC (PCM).
* `synth_check.py` — валидация против системного ffmpeg (corr=1.0 на 12 файлах).
* `enc_probe.py`   — эксперименты E1/E2/E3 (мишень MDCT, слои, грамматика wtype).

## Запуск валидации оракула

```bash
# нужен системный ffmpeg + numpy; путь к корпусу эталонных .vqf:
python3 synth_check.py /path/to/encoded_all/tvq_cleanroom
```

Ожидаемый результат: corr = 1.00000000, rel_err ≤ 3·10⁻⁶ на всех тестовых
файлах (моно/стерео, 4 режима, все ftype/wtype-переходы).

Анализ формата по живому декодеру и эталонам (E0–E7, массовая статистика,
бюджет) — инструменты в `python/analysis/` (см. их README); они импортируют
`vqflib.py`/`twinlayers.py` отсюда.
