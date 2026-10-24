#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_vendor_rs.py — генерация Rust-модулей vendored-данных TwinVQ
из эталонных исходников FFmpeg (third_party/ffmpeg/, LGPL-2.1-or-later).

Источники (только чтение):
  twinvq_data.h           — struct twinvq_data: 78 массивов int16 (fcb/cb/shape)
                            + float lsp22_2; bark_tab_* (uint16);
                            tab7..tab12 (uint8, 2D) + таблица tabs[] для very_broken_op
  metasound_twinvq_data.h — LSP-книги (const float ff_metasound_lsp*)

Выход:
  src/vendor/generated/twinvq_cb_data.rs
  src/vendor/generated/metasound_lsp_data.rs

Файлы коммитятся в репозиторий (сборка не требует python). Регенерация:
  python3 tools/gen_vendor_rs.py        (из корня репо)

Точность порта проверяется cargo test: spot-значения против исходников,
инварианты режимов, битовый бюджет против python/analysis/results/bit_budget.json.
"""
import os
import re
import struct

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FF = os.path.join(ROOT, "third_party", "ffmpeg")
OUT = os.path.join(ROOT, "src", "vendor", "generated")


def read_ff(name):
    return open(os.path.join(FF, name)).read()


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def parse_braced(text, start):
    """text[start] == '{' -> (внутренности, индекс после '}')."""
    assert text[start] == "{", f"expected '{{' at {start}"
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def c_ints(body):
    return [int(x, 0) for x in re.findall(r"-?\d+", body)]


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def c_floats(body):
    return [float(x) for x in FLOAT_RE.findall(body)]


def f32_exact(v):
    """Принудительное округление к float32 (как C float) + repr,\
    который гарантированно раунд-трипится обратно в ту же f32."""
    return struct.unpack("f", struct.pack("f", v))[0]


def wrap(items, per_line, indent="    "):
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(indent + " ".join(f"{x}," for x in items[i:i + per_line]))
    return "\n".join(lines)


def emit_1d(name, rust_type, vals, per_line=12):
    n = len(vals)
    return (f"#[rustfmt::skip]\npub static {name}: [{rust_type}; {n}] = [\n"
            f"{wrap([str(v) for v in vals], per_line)}\n];\n")


def emit_f32(name, vals, per_line=6):
    strs = [repr(f32_exact(v)) for v in vals]
    n = len(vals)
    return (f"#[rustfmt::skip]\npub static {name}: [f32; {n}] = [\n"
            f"{wrap(strs, per_line)}\n];\n")


HEADER = """\
// Vendored-данные TwinVQ (NTT) из FFmpeg — LGPL-2.1-or-later.
// {src}
// Сгенерировано tools/gen_vendor_rs.py — НЕ РЕДАКТИРОВАТЬ ВРУЧНУЮ.
// Имена: оригинальные C-имена, приведённые к UPPER_SNAKE.
// Комбинация с этим файлом наследует LGPL-2.1-or-later для данных частей.
"""


def gen_cb_data():
    src = strip_comments(read_ff("twinvq_data.h"))
    out = [HEADER.format(src="Источник: third_party/ffmpeg/twinvq_data.h")]

    # 1) Поля struct twinvq_data: тип + имя + размер.
    decls = re.findall(r"(int16_t|float)\s+(\w+)\[(\d+)\];", src)
    m = re.search(r"\}\s*tab\s*=\s*\{", src)
    assert m, "не найден инициализатор tab"
    body, _ = parse_braced(src, m.end() - 1)

    n_i16 = n_f32 = 0
    for ctype, name, size in decls:
        im = re.search(rf"\.{name}\s*=\s*\{{", body)
        assert im, f"нет инициализатора .{name}"
        inner, _ = parse_braced(body, im.end() - 1)
        size = int(size)
        rust_name = name.upper()
        if ctype == "int16_t":
            vals = c_ints(inner)
            assert len(vals) == size, (name, len(vals), size)
            out.append(emit_1d(rust_name, "i16", vals))
            n_i16 += 1
        else:
            vals = c_floats(inner)
            assert len(vals) == size, (name, len(vals), size)
            out.append(emit_f32(rust_name, vals))
            n_f32 += 1

    # 2) bark_tab_*: static const uint16_t name[] = {...};
    n_bark = 0
    for m in re.finditer(r"static\s+const\s+uint16_t\s+(bark_tab_\w+)\s*\[\s*\]\s*=\s*\{", src):
        inner, _ = parse_braced(src, m.end() - 1)
        out.append(emit_1d(m.group(1).upper(), "u16", c_ints(inner)))
        n_bark += 1

    # 3) tab7..tab12: static const uint8_t tabN[][W] = {{...},...};
    n_tab = 0
    for m in re.finditer(r"static\s+const\s+uint8_t\s+(tab\d+)\s*\[\s*\]\s*\[\s*(\d+)\s*\]\s*=\s*\{", src):
        outer, _ = parse_braced(src, m.end() - 1)
        width = int(m.group(2))
        rows = []
        pos = 0
        while True:
            rm = re.compile(r"\{").search(outer, pos)
            if not rm:
                break
            inner, after = parse_braced(outer, rm.start())
            vals = c_ints(inner)
            assert len(vals) == width, (m.group(1), len(vals), width)
            rows.append(vals)
            pos = after
        name = m.group(1).upper()
        body_rows = ",\n".join("    [" + " ".join(f"{v}," for v in r) + " ]" for r in rows)
        out.append(f"#[rustfmt::skip]\npub static {name}: [[u8; {width}]; {len(rows)}] = [\n{body_rows}\n];\n")
        n_tab += 1

    # 4) Таблица tabs[] (very_broken_op): {size, tab} — 13 входов.
    #    Указатель заменён номером таблицы (7..12), 0 = NULL.
    tm = re.search(r"\}\s*tabs\s*\[\]\s*=\s*\{", src)
    assert tm, "не найден инициализатор tabs[]"
    tbody, _ = parse_braced(src, tm.end() - 1)
    entries = re.findall(r"\{\s*(\d+)\s*,\s*(?:NULL|&tab(\d+))", tbody)
    assert len(entries) == 13, f"ожидалось 13 входов tabs[], найдено {len(entries)}"
    rows = ",\n".join(f"    ({size}, {tab if tab else 0})" for size, tab in entries)
    out.append(
        "/// very_broken_op: выбор «сломанной» таблицы по b5 (twinvq_data.h tabs[]).\n"
        "/// (size, номер таблицы 7..12); 0 = NULL (b5=0 не используется).\n"
        f"#[rustfmt::skip]\npub static BROKEN_OP_TABS: [(usize, u8); 13] = [\n{rows}\n];\n")

    path = os.path.join(OUT, "twinvq_cb_data.rs")
    open(path, "w").write("\n".join(out))
    print(f"twinvq_cb_data.rs: {n_i16} int16, {n_f32} float, "
          f"{n_bark} bark_tab, {n_tab} tab7..12, tabs[13] -> {os.path.getsize(path)} байт")


def gen_lsp_data():
    src = strip_comments(read_ff("metasound_twinvq_data.h"))
    out = [HEADER.format(src="Источник: third_party/ffmpeg/metasound_twinvq_data.h")]
    n = 0
    for m in re.finditer(r"(?:static\s+)?const\s+float\s+(ff_metasound_\w+)\s*\[\s*\]\s*=\s*\{", src):
        inner, _ = parse_braced(src, m.end() - 1)
        vals = c_floats(inner)
        out.append(emit_f32(m.group(1).upper(), vals))
        n += 1
    path = os.path.join(OUT, "metasound_lsp_data.rs")
    open(path, "w").write("\n".join(out))
    print(f"metasound_lsp_data.rs: {n} LSP-книг -> {os.path.getsize(path)} байт")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gen_cb_data()
    gen_lsp_data()
