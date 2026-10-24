# Архитектура tequila-энкодера TwinVQ

Технический документ, фиксирующий выводы из анализа эталонных источников
(FFmpeg `libavcodec/twinvq*.c`, `libavformat/vqf.c`, `libavutil/tx_template.c`).
Все утверждения ниже проверяются юнит-тестами.

Статус верификации: утверждения, помеченные `[E]`, подтверждены эмпирически
— зондированием живого декодера FFmpeg синтетическими битстримами,
разбором 218 эталонных битстримов NTT TestEnc и бит-точным python-оракулом
(`python/reference/`, corr = 1.0 против ffmpeg). Поведенческая модель
энкодера NTT (что и как он кодирует) — в [ENCODE_PIPELINE.md](ENCODE_PIPELINE.md);
этот документ — спецификация того, что строим.

## 1. Контейнер VQF (из libavformat/vqf.c)

```
"TWIN" | "97012000" (8 байт) | BE32 header_size
| чанки: 4-байтовый tag + BE32 len + payload ... 
| "DATA" | битстрим кадров (без длины!)
```

* `header_size` = Σ(8 + chunk_len) по всем чанкам до DATA. Позиция DATA-payload
  = 16 + header_size + 4.
* **COMM**: канон NTT — 16 байт: BE32 `channels-1`, BE32 `bitrate_kbps` (общий),
  BE32 `rate_flag` (44/22/11 или 8..44 = кГц) + 4 нулевых байта паддинга
  (официальный плеер ждёт поля на фиксированных офсетах: байт 36 = 0x00).
  Парсер понимает и 12-байтовый COMM чужих энкодеров (Yamaha).
* **DSIZ**: BE32 (в NTT SDK — размер исходных данных; FFmpeg просто кладёт в
  metadata). Официальный энкодер NTT его не пишет — наш writer тоже
  (парсер при этом понимает его в чужих файлах).
* **Текстовые меты-теги** (NAME, COMT, AUTH, «(c) », …) — tag + len + текст
  UTF-8, пишутся после COMM; FFmpeg маппит их в metadata (`vqf_metadata_conv`,
  18 шт.: NAME→title, COMT→comment, AUTH→author, «(c) »→copyright, FILE→filename,
  GENR→genre, …). Порядок записи tequila: NAME, COMT, AUTH, «(c) », далее по
  алфавиту chunk-id; нестандартные чанки — в конце. Нечётные длины допустимы,
  выравнивания нет (валидировано test3.vqf: длины 26/19/9/21, header_size=131).
  Реализация: `TAG_TABLE`/`order_meta_chunks` в `src/container.rs`, CLI-флаги
  `--title/--author/--comment/--copyright/…` + `--tag CHUNK=S`.
* **DATA**: кадры упакованы **на битовом уровне**, кадр `i` начинается с бита
  `i * frame_bit_len`, порядок MSB-first. `frame_bit_len = bitrate * size /
  sample_rate` (целочисленно).
* «Skip-байт» в начале кадра, который читает декодер (`skip_bits(get_bits(8))`),
  в файле НЕ хранится — это артефакт битового выравнивания пакетов демиксера
  (`vqf_read_packet` префиксует пакет байтами `[8-remaining_bits][last_byte]`).
  Энкодер просто пишет payload кадра бит-в-бит, начиная с wtype.
* В конце DATA — паддинг нулями до границы байта (+2 запасных байта, чтобы
  демиксер не спотыкался на EOF).

## 2. Режимы (twinvqdec.c)

9 режимов VQF: 08_08, 11_08, 11_10, 16_16, 22_20, 22_24, 22_32, 44_40, 44_48
(обозначение: `<кГц>_<кбит/с на канал>`). Выбор по extradata (= первые 12 байт
COMM): channels, bitrate, rate_flag.

`TwinVQModeTab`: `fmode[3]` (SHORT/MEDIUM/LONG: sub, bark_tab, bark_env_size,
bark_cb, bark_n_coef, bark_n_bit, cb0, cb1, cb_len_read), `size` (сэмплов на
кадр), LSP-параметры, PPC-параметры.

Подкадры: block_size = size / fmode[ftype].sub; SHORT sub∈{4,8,16},
MEDIUM sub∈{2,4}, LONG sub=1.

Стерео: `norm = 1`, моно: `norm = 2` (в масштабе IMDCT). Стерео-выход
пропускается через бабочку `L=c0+c1, R=c0-c1` ⇒ энкодер кодирует каналы
`c0=(L+R)/2`, `c1=(L−R)/2`.

## 3. Типы окон (wtype)

`wtype` — 4 бита в начале кадра, значения 0..8:

| wtype | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|---|---|---|---|---|---|---|---|---|
| ftype   | LONG | LONG | SHORT | LONG | MEDIUM | LONG | LONG | MEDIUM | MEDIUM |
| wsize-idx | 0 | 0 | 2 | 2 | 2 | 1 | 0 | 1 | 1 |

`types_sizes = { size/sub_long, size/sub_med, size/(2*sub_short) }`,
индекс из `wtype_to_wsize = {0,0,2,2,2,1,0,1,1}`.

В MEDIUM-кадрах: все подкадры имеют sub_wtype=8, КРОМЕ первого при wtype=4
(→4) и последнего при wtype=7 (→7).

Эмпирическая грамматика NTT-энкодера `[E]` (по 218 эталонам, детали в
ENCODE_PIPELINE.md §4.1):

```
стационар:      L0 L0 L0 ...                     (тишина: L0 всегда, без скобки)
атака/старт:    L1 → S2×n → L3 → L0...            (n = 1..26, растёт с резкостью)
```

* стартовая последовательность потока с ненулевым началом — `[1, 2, 3, 0…]`
  (pre-roll трактуется как транзиент); у чистой тишины скобки нет;
* wtype 0/1/6 синтезируются декодером идентично — различие чисто
  сигнальное (L1 — «приготовиться к атаке»);
* wtype=8 отвергается декодером FFmpeg и не встречается в NTT-потоках.

## 4. MDCT-конвенция FFmpeg TX (tx_template.c)

Прямой MDCT (naive, len = B = block_size, окно длиной 2B на стороне анализа):
`X[i] = scale · Σ_{j=0..2B-1} x[j]·cos((2j+1+B)(2i+1)·π/(4B))` — классический MDCT.

Обратный («компактный», len-выход; twinvq вызывает БЕЗ флага AV_TX_FULL_IMDCT):
```
len = B/2
y[i]      =  s·Σ_k X[k]·cos((2k+1)(2B−2i−1)·π/(4B))    i = 0..B/2−1
y[i+B/2]  = −s·Σ_k X[k]·cos((2k+1)(3B+2i+1)·π/(4B))    i = 0..B/2−1
```
т.е. на выходе только B сэмплов (первая «компактная» половина полного IMDCT).

Масштаб twinvq-декодера: `s = −sqrt(norm/B)/2^15`, norm = 2 (моно) / 1 (стерео).

### 4a. Мишень энкодера (адъюнкт синтеза; ранее — калибровка `[E]`)

Мишень кадра k — **точный адъюнкт синтеза декодера** (PR-пара,
`encoder::analysis_targets`): анализ = транспонирование всего тракта
`imdct_and_window` + `imdct_output` (схема Эдлера). Для LONG-кадров
результат тождествен калибровке E1 (совпадение с `c·mdct_fwd` до 2·10⁻⁹,
масштаб ровно 1.0):

```
X_i = c · Σ_{j=0..2B−1} x[off+j] · sin((j+½)π/(2B)) · cos((2j+1+B)(2i+1)π/(4B))
```

* смещение `off = (k−2)·size` — ровно −2 кадра (pre-roll; компенсируется
  двумя discard-пакетами декодера, выход выровнен со входом);
* масштаб `c = 2^16/√(norm·B)`.

Для SHORT/MEDIUM и переходных окон (L1/L3) адъюнкт не сводится к
«окну 2B без сдвига» (наивная версия нарушала TDAC — отрицательный SNR
на SHORT-кадрах). Геометрия (выведена из синтеза):

* `p_k = (size + first_wsize_k)/2` — `last_block_pos` после кадра k;
  `curr_k[c]` лежит на потоке в `(k−1)·size − p_k + c` (линейно) для
  `c < p_k + size1_{k+1}`; хвост сворачивается бабочкой кадра k+1
  (фолд с весами w2/w1 окна следующего кадра);
* на подкадр j: (i) y[ws/2−1−t] — адъюнкт бабочки j; (ii) y[n] —
  плоская часть (выживает до перезаписи бабочкой j+1; для последнего
  подкадра — весь хвост через фолд); (iii) y[b−ws_{j+1}/2+t] —
  P-часть следующей бабочки; центры первых наклонов — на сетке
  `c_k = (k−2)·size + size/2`, блоки сдвинуты на `(size−b)/2`;
* `X_j = −c·mdct_analysis_compact(y_j)` (σ = −1 выведено из совпадения
  с E1 на LONG).

PR без квантования: SNR 252–256 дБ при масштабе ровно 1.000000000 на
всех допустимых последовательностях окон (тест `analysis_synthesis_pr`).

LPC-анализ строится по **|X|** (амплитудной огибающей): без abs
сглаживание гасит знакопеременный спектр, env схлопывается до ~25 дБ
динамики — глубокую форму спектра обязан нести LPC (bark-слой имеет
лишь ~10 дБ: st ≥ 1.1); тест `lpc_envelope_covers_deep_spectral_shape`
(провал ≥ 40 дБ, реально ~70 дБ).

## 5. Синтез (порт imdct_and_window + imdct_output)

На канал — кольцевой буфер 2·size. Подкадр j:
1. `y[j]` = компактный IMDCT подкадра → buf1[j·B .. j·B+B).
2. Бабочка TDAC с окном `w[m] = sin(π(m+0.5)/(2·wsize))` (полуокно-рампа):
   ```
   src0 = хвост предыдущего подкадра/кадра (W/2 сэмплов, заканчивается на границе блока)
   src1 = y[j][0..W/2)
   out[P+t]         = src0[t]·w[W−1−t] − src1[W/2−1−t]·w[t]
   out[P+W−1−t]     = src0[t]·w[t]     + src1[W/2−1−t]·w[W−1−t]
   ```
3. `out[P+W .. P+W+B−W/2) = y[j][W/2..B)` (середина без окна; «перехлёст»
   последних W/2 сэмплов перезаписывается бабочкой следующего подкадра).
4. `P += B` (для MEDIUM + (B−W)/2).
5. src0 следующего = `y[j][B−W'/2..B)`.

Первый подкадр кадра берёт src0 из `prev_frame_ch + last_block_pos + (size−B)/2`,
где `last_block_pos = (size + first_wsize)/2` предыдущего кадра.

Выход кадра: `out = prev[last_block_pos..size) ++ curr[0..last_block_pos)`,
затем (стерео) бабочка L/R. Латентность декодера — 2 кадра (priming).

## 6. Порядок полей кадра (twinvq_read_bitstream)

```
wtype(4)
main VQ: for i in 0..n_div[ftype]-1:
    idx0 = bits(bits_main_spec[0][ftype][i>=change])
    idx1 = bits(bits_main_spec[1][ftype][i>=change])      // MSB-first
bark1[ch][sub][k]  (bark_n_bit каждый)
bark_use_hist[ch][sub] (1 бит)
НЕ-LONG: gain[ch](8); для каждого ch: sub_gain[ch][sub](5)  — gain всех каналов подряд, потом sub_gain всех каналов подряд
LONG:    gain[ch](8)
LSP: для каждого ch: lpc_hist(lsp_bit0), lpc_idx1(lsp_bit1), lpc_idx2[lsp_split](lsp_bit2)
LONG: PPC cb (read_cb_data с ftype=PPC), затем для каждого ch: p_coef(ppc_period_bit), g_coef(pgain_bit)
```

⚠️ Точный порядок gain/sub_gain: внешний цикл по каналам, внутри — gain затем
все sub_gain этого канала.

## 7. Битовый бюджет (init_bitstream_params)

```
total_fr_bits = bitrate * size / sample_rate
lsp_bits_per_block = n_ch*(lsp_bit0 + lsp_bit1 + lsp_split*lsp_bit2)
ppc_bits = n_ch*(pgain_bit + ppc_shape_bit + ppc_period_bit)
bse_bits[i] = n_ch*(fmode[i].bark_n_coef*fmode[i].bark_n_bit + 1)
bsize_no_main_cb[LONG] = bse_bits[2] + lsp + ppc + 4(WTYPE) + n_ch*8
bsize_no_main_cb[SHORT/MEDIUM] = lsp + n_ch*8 + 4 + fmode[i].sub*(bse_bits[i] + n_ch*5)
для i in {SHORT, MEDIUM, LONG}:
    bit_size  = total_fr_bits − bsize_no_main_cb[i]
    vect_size = n_ch * size
для i = PPC:
    bit_size  = n_ch * ppc_shape_bit
    vect_size = n_ch * ppc_shape_len
для всех i: n_div = (bit_size+13)/14;
    rounded_up = ceil(bit_size/n_div), rounded_down = floor(bit_size/n_div)
    bits_main_spec[0][i][0] = ceil(rounded_up/2), [1][i][0] = floor(rounded_up/2)
    bits_main_spec[0][i][1] = ceil(rounded_down/2), [1][i][1] = floor(rounded_down/2)
    bits_main_spec_change[i] = n_div − (rounded_up*n_div − bit_size)
    length[i][0/1] = ceil/floor(vect_size/n_div); length_change[i] = ...
```
При 7 битах у кодбука младшие 6 бит — индекс, старший — знак (де-kvant:
`sign = (idx & 0x40) ? -1 : +1; idx &= 0x3F`).

### 7a. Сводный бюджет по режимам `[E]` (моно; полный — python/analysis/results/bit_budget.json)

| Режим | size | мс/кадр | бит/кадр | сжатие | LONG: main_vq / bark / PPC / LSP / gain | SHORT: main_vq / bark / sub_gain |
|-------|------|---------|----------|--------|----------------------------------------|--------------------------------|
| 08_08 | 512  | 64.0  | 512  | 16000 | 424 / 18 / 42 / 15 / 8   | 397 / 40 / 40 |
| 11_08 | 512  | 46.4  | 371  | 22050 | 269 / 18 / 52 / 19 / 8   | 252 / 40 / 40 |
| 11_10 | 512  | 46.4  | 464  | 17640 | 362 / 18 / 52 / 19 / 8   | 345 / 40 / 40 |
| 16_16 | 1024 | 64.0  | 1024 | 16000 | 902 / 18 / 72 / 19 / 8   | 905 / 40 / 40 |
| 22_20 | 1024 | 46.4  | 928  | 17640 | 800 / 24 / 72 / 19 / 8   | 801 / 48 / 40 |
| 22_24 | 1024 | 46.4  | 1114 | 14700 | 986 / 24 / 72 / 19 / 8   | 987 / 48 / 40 |
| 22_32 | 512  | 23.2  | 743  | 11025 | 611 / 24 / 72 / 23 / 8   | 660 / 24 / 20 |
| 44_40 | 2048 | 46.4  | 1857 | 17640 | 1697 / 24 / 100 / 23 / 8 | 1630 / 96 / 80 |
| 44_48 | 2048 | 46.4  | 2229 | 14700 | 2069 / 24 / 100 / 23 / 8 | 2002 / 96 / 80 |

Аудит: сумма компонент == frame_bit_len во всех 54 комбинациях
(9 режимов × 2 канала × 3 ftype). Стерео удваивает поля на канал
(gain/sub_gain/bark/LSP/PPC) и расширяет main VQ до 2·size на вектор.

## 8. Перестановки interleave (construct_perm_table)

VQ-последовательность S[t] (t = 0..vect_size-1) ↔ плоский спектр
(канал-мажор, подкадры подряд): `flat[permut[ftype][t]] = S[t]`.
Перестановка = permutate_in_line (циклический сдвиг в строках) → transpose_perm
(транспонирование) → linear_perm (блоки по size/sub).

Энкодер: `S[t] = flat[permut[ftype][t]]`, разрезается на n_div векторов
длины length[i][j>=length_change], каждый квантуется парой кодбуков
(cb0+cb1, conjugate VQ с знаками при 7 битах).

## 9. Канал обработки декодера → порядок энкодера

Декодер (на канал): `X_j[k] = (VQ_j[k]·barkenv_j[k]·gain_j + PPC_j[k])·lpcenv[k]`

Энкодер (обратный порядок):
1. MDCT всех подкадров → X_j (мишень §4a).
2. LPC-анализ в MDCT-области (по спектру кадра) → LSP → split-VQ
   (lpc_idx1 + lpc_idx2[] + lpc_hist) → точная реплика decoder-конвертации
   LSP→lpcenv (включая интерполяционные quirk'и eval_lpcenv_or_interp) → E[k].
3. R_j = X_j / E.
4. (LONG) оценка PPC: период → width → shape/gain; R_j −= PPC_synth.
5. Bark-огибающая по подкадрам: VQ-поиск (bark_cb, MA-предсказание hist
   с весами 0.4/0.35/0.28 по ftype, quirk `if st < −1: st = 1`),
   gain/sub_gain квантование (mulaw) → env_j.
6. V_j = R_j / env_j → interleave (permut) → conjugate VQ поиск.
7. Запись полей по §6.

Эмпирические калибровки слоёв `[E]` (подробности и доказательства —
ENCODE_PIPELINE.md §4):

* PPC: гребёнка f0 ∈ [100, 600] Гц одинаково для всех режимов; при
  f0 < 100 Гц кодируется ближайшая гармоника сверху; ширина пика ∝
  периоду; позиции пиков — через `very_broken_op` (сломанное
  float-округление бинарного NTT — реплицировать как есть);
* кодбуки main VQ имеют «сладкую зону» масштаба: коэффициент передачи
  пары книг α(σ) = 0.18…0.77 — точный подбор gain балансирует остаток
  в эту зону (это ограничивает достижимую корреляцию 0.80–0.91);
* ошибки LSP «живут» на 1 кадр дольше остальных из-за MA-предсказания
  (карта влияния битовых флипов, E5);
* числовая дисциплина: внутренний синтез-контур энкодера обязан
  воспроизводить арифметику декодера (промоуты double→float в C),
  иначе замкнутый поиск сойдётся не к тому решению (полюса 0.5/(p+q)
  LSP-огибающей амплифицируют численную разницу на порядки).

## 10. Данные

Кодбуки и таблицы — из FFmpeg (`twinvq_data.h`, `metasound_twinvq_data.h`),
LGPL-2.1+. Генератор `tools/gen_vendor_rs.py` портирует их в
`src/vendor/generated/*.rs` (78 int16-кодовбук + lsp22_2 + 17 bark-таблиц
+ tab7..12 + tabs[] very_broken_op + 5 LSP-книг; генеренные файлы
коммитятся — сборка не требует python); таблицы режимов —
`src/vendor/tables.rs` (перенос mode_* из twinvqdec.c). Остальной код —
cleanroom (MIT), комбинация наследует LGPL для связанных частей.

## 11. Верификация

1. Юнит-тесты: битовый бюджет, перестановки, round-trip bitstream.
2. PR-тест: analysis→synthesis тождество (различные wtype/mode).
3. Эмпирическая калибровка масштаба/сдвига MDCT по декодированию системным
   ffmpeg (delta-спектр).
4. Прогон по корпусу signals/: ffmpeg decode без ошибок, seg-SNR.

## 12. Состояние верификации и инструменты (на 2026-10-24)

Реализация — Rust (Этап R1; C++-скелет снят — история 112e357).

| Инструмент | Что делает | Статус |
|------------|-----------|--------|
| `cargo test` — src/bitstream.rs | round-trip битового I/O, склейка/паддинг кадров на битовых границах, чтение за концом | ✅ 11 тестов |
| `cargo test` — src/modes.rs | инварианты 9 режимов (Σ bark_tab == block_size, размеры кодбуков), spot-значения против twinvq_data.h, wtype-таблицы, broken_op tabs | ✅ 6 тестов |
| `cargo test` — src/layout.rs | бюджет всех 18 комбинаций (9 режимов × 1/2 канала) против bit_budget.json; ручной пересчёт 44m48 LONG (n_div=148, change=145); самосогласованность дивизий; диапазон f0 PPC | ✅ 3 теста |
| `python/reference/synth_check.py` | python-декодер vs системный ffmpeg | ✅ corr=1.0, rel_err≤3·10⁻⁶ (12 файлов NTT: моно/стерео, 4 режима, все ftype/wtype) |
| `python/analysis/exp_probe.py` | синтетика → ffmpeg: E0–E7 (priming, PPC, bark, LSP, ошибки, wtype, контейнер) | ✅ результаты в results/ |
| `python/analysis/real_stats.py` | 218 эталонов NTT: статистика полей, seg-SNR | ✅ 218/218 парсятся; seg-SNR 5.8–16 дБ (44s96 до 29 дБ) |
| Корпус дифтестов | python/stage4_report.json (33 файла, 8 режимов) | ⬜ подключение на этапе golden-тестов Rust-парсера |

Карта влияния битовых флипов (E5 + серия dbg): main VQ / wtype / PPC
локализованы в выходных кадрах k−2, k−1; LSP-ошибка дополнительно задевает
кадр k (MA-предсказание) — используется для локализации расхождений при
дифтестировании энкодера против оракула.
