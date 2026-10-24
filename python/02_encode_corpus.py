#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_encode_corpus.py — автоматизированный прогон NTT TestEnc по корпусу.

Для каждого WAV из каталога signals/ запускает TestEnc (32-бит i386 бинарник
из SDK) и складывает результаты:
  encoded/<режим>/<имя>.vqf   — битстримы (эталонный "оракул")
  encoded/logs/<режим>.log    — stderr енкодера
  encoded/manifest.json       — параметры, sha256, время кодирования

Режим (частота/каналы) TestEnc берёт из заголовка WAV (см. encoder.cxx),
поэтому передаётся только битрейт: TestEnc -b <kbps> in.wav out.vqf

Запуск (в окружении с i386 поддержкой):
  python3 02_encode_corpus.py --testenc /path/to/TestEnc \
      [--signals signals] [--out encoded] [--jobs 1] [--dry-run]

Проверка окружения перед массовым запуском:
  # Debian/Ubuntu: sudo apt install libc6-i386
  ./TestEnc --version    # должен напечатать версию модуля
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from vqf_format import NTT_ENCODER_MODES


def mode_dir_name(rate_flag, channels, kbps):
    return "%d%s%d" % (rate_flag, "s" if channels == 2 else "m", kbps)


def find_jobs(signals_dir):
    """Собрать (wav, rate_flag, channels, kbps) по структуре каталогов."""
    jobs = []
    for entry in sorted(os.listdir(signals_dir)):
        mdir_path = os.path.join(signals_dir, entry)
        if not os.path.isdir(mdir_path):
            continue
        # режим из имени каталога: <rate><s|m><kbps>
        digits = "".join(c for c in entry if c.isdigit())
        ch = 2 if "s" in entry else 1
        if not digits or "s" not in entry and "m" not in entry:
            print("!! пропускаю непонятный каталог %s" % entry)
            continue
        rate_flag = None
        for r in (44, 22, 16, 11, 8):
            if entry.startswith(str(r)):
                rate_flag = r
                break
        if rate_flag is None:
            print("!! не распознана частота в %s" % entry)
            continue
        kbps = int(digits[len(str(rate_flag)):])
        for wav in sorted(os.listdir(mdir_path)):
            if wav.lower().endswith(".wav"):
                jobs.append((os.path.join(mdir_path, wav), rate_flag, ch, kbps))
    return jobs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(args):
    (testenc, wav, rate_flag, ch, kbps, outdir, logdir, dry) = args
    mdir = mode_dir_name(rate_flag, ch, kbps)
    name = os.path.splitext(os.path.basename(wav))[0]
    vqf = os.path.join(outdir, mdir, name + ".vqf")
    cmd = [testenc, "-b", str(kbps), "-T", name, wav, vqf]
    if dry:
        print("[dry-run] %s" % " ".join(cmd))
        return {"wav": wav, "cmd": cmd, "dry_run": True}

    os.makedirs(os.path.dirname(vqf), exist_ok=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=3600, errors="replace")
        elapsed = round(time.time() - t0, 2)
        rec = {
            "mode": mdir, "wav": wav, "vqf": vqf, "cmd": cmd,
            "returncode": proc.returncode, "elapsed_s": elapsed,
            "wav_sha256": sha256(wav),
        }
        if proc.returncode == 0 and os.path.exists(vqf) and os.path.getsize(vqf) > 0:
            rec["vqf_sha256"] = sha256(vqf)
            rec["vqf_bytes"] = os.path.getsize(vqf)
            status = "OK  "
        else:
            status = "FAIL"
        log = os.path.join(logdir, mdir + ".log")
        with open(log, "a") as f:
            f.write("\n===== %s =====\n$ %s\n[rc=%d, %.2fs]\n%s\n%s\n" % (
                name, " ".join(cmd), proc.returncode, elapsed,
                proc.stdout, proc.stderr))
        print("%s %-8s %-14s %6.2fs" % (status, mdir, name, elapsed))
        return rec
    except FileNotFoundError:
        print("FAIL TestEnc не найден: %s" % testenc)
        raise
    except subprocess.TimeoutExpired:
        print("FAIL timeout: %s" % wav)
        return {"wav": wav, "cmd": cmd, "error": "timeout"}


def main():
    ap = argparse.ArgumentParser(description="Прогон NTT TestEnc по корпусу")
    ap.add_argument("--testenc", required=True, help="путь к бинарнику TestEnc")
    ap.add_argument("--signals", default="signals", help="каталог с WAV (от 01)")
    ap.add_argument("--out", default="encoded", help="каталог вывода")
    ap.add_argument("--jobs", type=int, default=1,
                    help="параллельных процессов (енкодер однопоточный, 1-2 достаточно)")
    ap.add_argument("--dry-run", action="store_true", help="только напечатать команды")
    args = ap.parse_args()

    if not os.path.isfile(args.testenc) and not args.dry_run:
        sys.exit("TestEnc не найден: %s" % args.testenc)
    if not os.path.isdir(args.signals):
        sys.exit("Каталог сигналов не найден: %s (сначала 01_gen_signals.py)" % args.signals)

    jobs = find_jobs(args.signals)
    if not jobs:
        sys.exit("В %s не найдено WAV-файлов" % args.signals)

    outdir = os.path.abspath(args.out)
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    print("Найдено %d WAV для кодирования\n" % len(jobs))
    tasks = [(os.path.abspath(args.testenc), wav, rf, ch, kbps,
              outdir, logdir, args.dry_run) for (wav, rf, ch, kbps) in jobs]

    if args.dry_run:
        for t in tasks:
            run_one(t)
        print("\n[dry-run] команды показаны, ничего не выполнялось")
        return 0

    results = []
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(run_one, tasks))
    else:
        results = [run_one(t) for t in tasks]

    manifest = {
        "testenc": os.path.abspath(args.testenc),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "total": len(results),
        "ok": sum(1 for r in results if r.get("vqf_sha256")),
    }
    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    ok = manifest["ok"]
    print("\nИтог: %d/%d успешно" % (ok, len(results)))
    print("Битстримы: %s/\nМанифест:  %s" % (outdir, mpath))
    if ok:
        print("Проверка:  ffprobe -hide_banner \"%s\" | head -20" % results[0]["vqf"])
        print("Разбор:    python3 03_parse_vqf.py \"%s\" --summary" % results[0]["vqf"])
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
