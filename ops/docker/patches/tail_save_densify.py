# ---------------------------------------------------------------------------
# ORIGIN: rivals/god-image-tourn-1dccbd24755b58a3-20260907-position-1/ops/docker/patches/tail_save_densify.py
#         (2026-09-07 image tournament winner, hotkey 5EhyCWPu). Copied VERBATIM on
#         2026-09-08 for the I6 winner-recipe port (PLAN-IMAGE-0908 v2); everything
#         below this header is the unmodified original.
# LICENSE: the source repo ships Apache-2.0 (LICENSE.md) plus the Gradients-on-Demand
#         NOTICE (Copyright 2025 Grads LLC; attribution to Gradients.io required) -
#         see this repo's LICENSE.md / NOTICE.
# GATE:   config key train.tailsave_deadline_ts (TrainConfig default 0.0 -> `if _tsv_deadline > 0.0` false); MUST be applied after deadline_anneal (shared CFG anchor)
#         -> applying at build time changes nothing until IMG_RECIPE=winner sets it
#            (ops/docker/scripts/image_toolkit_entrypoint.py).
# ---------------------------------------------------------------------------
"""Build-time patch: TAIL-SAVE DENSIFIER — rapatkan cadence checkpoint di ekor run.

MASALAH. Validator mematikan container dengan SIGKILL tanpa grace tepat saat
`hours_to_complete` habis, dan artefak yang terkirim untuk run yang terpotong
adalah save komplit TERAKHIR di grid. Dengan `save_every` statis (mis. 200
langkah @ ~1,8 dtk/langkah), titik potong jatuh rata-rata ~180 dtk SETELAH save
terakhir — pekerjaan optimisasi ratusan langkah terbuang setiap kali run
menyentuh tembok jam. Dan titik potongnya tidak deterministik: throughput mesin
bergeser antar-run (kami mengukur selisih belasan persen pada konfigurasi
byte-identik), jadi tidak ada nilai `save_every` statis yang menutup lubang ini.

MEKANISME (lingkar-tertutup, bukan nilai knob):
  1. UKUR throughput nyata (dtk/langkah) dari waktu dinding yang berlalu.
  2. RAMAL langkah terakhir yang masih muat sebelum `tailsave_deadline_ts`.
  3. Saat langkah sekarang masuk `tailsave_frac` terakhir dari ramalan itu,
     TAMBAHKAN save ekstra tiap `save_every // tailsave_div` langkah.
Ramalan diulang setiap langkah, jadi perubahan kecepatan mesin di tengah run
ikut terserap. Kondisi save asli TIDAK disentuh — mekanisme ini hanya bisa
MENAMBAH titik save, tak pernah mengurangi; gagal ramal = perilaku upstream
persis. Berbeda dari sekadar mengecilkan `save_every` (itu nilai knob):
di sini cadence adalah FUNGSI dari throughput terukur dan jam mati, dua input
runtime yang tidak dikenal konfigurasi statis mana pun.

INERT UNLESS SELECTED: `tailsave_deadline_ts` default 0.0 -> seluruh blok
tidak berjalan; resep lain byte-identik. BUILD gagal keras bila anchor
upstream bergeser.
"""

CFG = "/app/ai-toolkit/toolkit/config_modules.py"
PROC = "/app/ai-toolkit/jobs/process/BaseSDTrainProcess.py"

CFG_OLD = """        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.lr_scheduler_params = kwargs.get('lr_scheduler_params', {})"""

CFG_NEW = """        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.lr_scheduler_params = kwargs.get('lr_scheduler_params', {})
        # [tailsave] jam mati absolut utk pemadat save ekor. 0 = mekanisme mati.
        self.tailsave_deadline_ts = kwargs.get('tailsave_deadline_ts', 0.0)
        self.tailsave_frac = kwargs.get('tailsave_frac', 0.10)
        self.tailsave_div = kwargs.get('tailsave_div', 4)"""

PROC_OLD = """                # sets progress bar to match out step
                if self.progress_bar is not None:
                    self.progress_bar.update(step - self.progress_bar.n)"""

PROC_NEW = """                # sets progress bar to match out step
                if self.progress_bar is not None:
                    self.progress_bar.update(step - self.progress_bar.n)

                # [tailsave] Pemadat save ekor: ukur throughput nyata, ramalkan
                # langkah saat jam mati tiba, dan di jendela terakhir ramalan itu
                # sisipkan save ekstra - checkpoint terakhir yang selamat dari
                # SIGKILL jadi jauh lebih segar. Hanya MENAMBAH save; kondisi save
                # upstream tidak disentuh.
                _tsv_deadline = float(getattr(self.train_config, 'tailsave_deadline_ts', 0.0) or 0.0)
                if _tsv_deadline > 0.0 and step > self.start_step:
                    import time as _tsv_time
                    if not hasattr(self, '_tsv_t0'):
                        self._tsv_t0 = _tsv_time.time()
                        self._tsv_step0 = step
                        self._tsv_last = -1
                    _tsv_done = step - self._tsv_step0
                    if _tsv_done >= 20:
                        _tsv_sps = (_tsv_time.time() - self._tsv_t0) / float(_tsv_done)
                        _tsv_left = _tsv_deadline - _tsv_time.time()
                        if _tsv_sps > 0.0 and _tsv_left > 0.0:
                            _tsv_end = step + int(_tsv_left / _tsv_sps)
                            _tsv_frac = min(0.5, max(0.02, float(
                                getattr(self.train_config, 'tailsave_frac', 0.10) or 0.10)))
                            _tsv_span = max(1.0, _tsv_frac * float(_tsv_end - self._tsv_step0))
                            if float(step) >= float(_tsv_end) - _tsv_span:
                                _tsv_div = max(2, int(getattr(self.train_config, 'tailsave_div', 4) or 4))
                                _tsv_every = max(1, int(self.save_config.save_every) // _tsv_div)
                                if step % _tsv_every == 0 and step != self._tsv_last \\
                                        and step % self.save_config.save_every != 0:
                                    self._tsv_last = step
                                    print(f'[sn56-tailsave] save ekstra @ step {step} '
                                          f'(ramalan akhir {_tsv_end})', flush=True)
                                    self.save(step)"""


def _patch(path, old, new, tag):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if new in src:
        print("[tsv] %s sudah terpasang" % tag)
        return
    n = src.count(old)
    if n != 1:
        raise SystemExit("[tsv] FATAL: anchor muncul %d kali di %s (%s) - butuh tepat 1" % (n, path, tag))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new, 1))
    print("[tsv] terpasang: %s" % tag)


_patch(CFG, CFG_OLD, CFG_NEW, "field tailsave_* di TrainConfig")
_patch(PROC, PROC_OLD, PROC_NEW, "pemadat save ekor di loop training")
