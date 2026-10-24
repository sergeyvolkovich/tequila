# Патентное исследование: TwinVQ / VQF

Дата: 2026-10-24. Статус: предварительное (Google Patents частично
rate-limit'ил запросы; помечено, что требует дополнительной проверки).

## Резюме

Ключевые патенты NTT на технологию TwinVQ относятся к 1994–1998 гг.
(приоритеты) и **истекли** (20 лет от приоритета: 2014–2018). Формат файлов
`.vqf` и битстрим как таковые не охраняются авторским правом (факты/функциональные
требования), но **таблицы данных из FFmpeg лицензируются LGPL-2.1+** — проект
использует их по решению владельца репозитория, что делает связанные части
проекта LGPL-совместимыми.

## Подтверждённые патенты

### US 5,684,920 — основной патент TwinVQ (NTT)

* **Название**: "Acoustic signal transform coding method and decoding method
  having a high efficiency envelope flattening method therein"
* **Авторы**: Naoki Iwakami, Takehiro Moriya, Satoshi Miki
* **Правообладатель**: Nippon Telegraph and Telephone Corp (ныне NTT Inc)
* **Приоритет**: 1994-03-17 (JP), подача US: 1995-03-13, выдача: 1997-11-04
* **Истечение**: 2015-03-13 — **Expired - Lifetime** (подтверждено Google Patents)
* **Содержание**: MDCT → разделение на огибающую (LPC/Bark) и остаток →
  предсказание огибающей остатка → нормализация → векторное квантование
  «fine structure». Это ядро схемы TwinVQ (спектральное выравнивание + VQ).

### US 5,970,443 — Yamaha (система VQ с кодбуками)

* Yamaha Corp (Shigeki Fujii), приоритет 1996-09-24, выдача 1999-10-19,
  **истёк 2017-09-22** (Expired - Fee Related).
* Упоминает применимость к "Twin VQ system" — перекрёстная область, не блокер.

## Семейство и связанные заявки (требуют верификации)

По классификации CPC **G10L19/038** ("Vector quantisation, e.g. TwinVQ audio")
можно найти патенты-потомки, но они относятся к более поздним кодекам
(Huawei/Qualcomm/Google neural-кодеки и т.п.), не к TwinVQ.

Ожидаемые члены семейства NTT (по публикациям ICASSP-95/96 и известной
патентной практике NTT, **номера не подтверждены**):
* JP-эквиваленты US5684920 (приоритет 1994-03-17, JP1994-048652?)
* EP/WO-эквиваленты того же приоритета
* Патенты на LSP-квантование и conjugate VQ середины 1990-х

**Действие**: проверить через Espacenet по номеру JP-приоритета и
цитированиям US5684920 (см. раздел «Открытые вопросы»).

## Правовой контекст по компонентам

| Компонент | Статус |
|-----------|--------|
| Алгоритмы TwinVQ (US5684920 и семья) | Патенты истекли (2015–2018), public domain |
| Формат .vqf / битстрим | Не охраняется (функциональные факты) |
| Таблицы-кодбуки (из FFmpeg twinvq_data.h) | LGPL-2.1+ (copyright FFmpeg/авторов) |
| FFmpeg-код (twinvq.c и др.) | LGPL-2.1+ — используем только как **эталон для чтения**, не копируем логику дословно (данные таблиц — да, по решению владельца) |
| Название "TwinVQ" | Вероятно торговая марка NTT — не использовать в названии продукта |

## Источники

1. Google Patents, US5684920A:
   https://patents.google.com/patent/US5684920A/en — статус Expired,
   приоритет 1994-03-17, NTT.
2. Google Patents, US5970443A (Yamaha):
   https://patents.google.com/patent/US5970443A/en
3. Wikipedia, "TwinVQ": разработан NTT Human Interface Labs в 1994,
   MPEG-4 Audio (ISO/IEC 14496-3:1999, subpart 4), проприетарный формат .vqf,
   Yamaha SoundVQ.
4. RareWares, "NTT TwinVQ encoder": SDK Tvq20001012 (источник TestEnc,
   используемого в корпусе верификации).
5. Iwakami, Moriya, Miki et al., "High-quality audio-coding at less than
   64 kbit/s by using transform-domain weighted interleave vector
   quantization (TwinVQ)", ICASSP-95.
6. Miki, Moriya, Iwakami, Kaneko, "Applied services using twinVQ
   audio-coding method", NTT R&D 45(12), 1996.

## Открытые вопросы

1. Полный список патентов NTT семейства TwinVQ (JP/EP/WO) и их дат истечения —
   искать в Espacenet по inventors "Iwakami; Moriya; Miki" + assignee NTT,
   период 1994–1999. Ожидаемо все истекли не позднее ~2019.
2. Не заявлены ли патенты на MPEG-4 TwinVQ профиль отдельно (ISO/IEC 14496-3
   subpart 4 может покрываться patent pool MPEG LA / Via-LA — проверить
   статус пула MPEG-4 Audio, который закрыт/истёк для старых профилей).
3. Трактовка кодбуков как «данных»: сами значения таблиц выведены из NTT SDK;
   в юрисдикциях без copyright на данные (США, functional data) риск низкий,
   но файл FFmpeg как компиляция охраняется — потому LGPL.

## Практический вывод для проекта

Cleanroom-реализация энкодера TwinVQ **юридически осуществима**: патенты
истекли. Использование таблиц из FFmpeg накладывает LGPL на связанные части
(решение задокументировано в README). Название продукта не должно содержать
"TwinVQ" без разрешения NTT.
