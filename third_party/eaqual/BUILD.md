# EAQUAL (PEAQ-Basic) — внешняя валидация режима --psy

Сборка (v0.1.3alpha, zplane.development; патчи под современный gcc/libsndfile
уже применены к src/: CRLF→LF, sf_open_read→sf_open, вырезан pcmbitwidth):

    cd build/unix && make -f Makefile.sys -j4
    # → bin/EAQUAL -fref ref.wav -ftest test.wav  (печатает ODG/DIX + MOVs)

Makefile.sys линкует СИСТЕМНЫЙ libsndfile (pkg-config sndfile).
Родной Makefile требует старый bundled-libsndfile и не используется.
Лицензия — см. license/ (GPL).
