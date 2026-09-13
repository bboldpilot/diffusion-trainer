# ---------------------------------------------------------------------------
# ORIGIN: rivals/god-image-tourn-1dccbd24755b58a3-20260907-position-1/ops/docker/patches/deadline_anneal.py
#         (2026-09-07 image tournament winner, hotkey 5EhyCWPu). Copied VERBATIM on
#         2026-09-08 for the I6 winner-recipe port (PLAN-IMAGE-0908 v2); everything
#         below this header is the unmodified original.
# LICENSE: the source repo ships Apache-2.0 (LICENSE.md) plus the Gradients-on-Demand
#         NOTICE (Copyright 2025 Grads LLC; attribution to Gradients.io required) -
#         see this repo's LICENSE.md / NOTICE.
# GATE:   config key train.deadline_ts (TrainConfig default 0.0 -> controller block skipped, `if _da_deadline > 0.0`)
#         -> applying at build time changes nothing until IMG_RECIPE=winner sets it
#            (ops/docker/scripts/image_toolkit_entrypoint.py).
# ---------------------------------------------------------------------------
"""Build-time patch: DEADLINE-AWARE CLOSED-LOOP LR RETARGET (`deadline_anneal`).

MASALAH YANG DIPERBAIKI (terukur, bukan dugaan).

Training kami dihentikan oleh JAM MATI, bukan oleh selesainya jadwal. Pada task
krea2 berjatah 0,75 jam, run berhenti di sekitar langkah 1700 dari 2000 yang
dikonfigurasi. Yang dikirim ke validator adalah `last.safetensors` - bobot apa
adanya pada detik jam habis.

Dan titik potong itu TIDAK TETAP. Dua run dengan konfigurasi BYTE-IDENTIK pada
mesin yang sama kami ukur berjalan pada 1,55 dan 1,85 detik/langkah - selisih
19% - sehingga langkah yang tercapai ikut bergeser dari run ke run.

Sementara itu jadwal LR ai-toolkit disetel ke jumlah langkah NOMINAL:

    toolkit/config_modules.py:367    self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
    jobs/process/BaseSDTrainProcess.py:2086
                                     lr_scheduler_params['total_iters'] = self.train_config.steps

dan template krea2 kami memakai default `constant` pada lr 1e-4. Kami verifikasi
di log run nyata: lr tetap 1.0e-04 pada langkah 200, 800, 1400, dan 1900.

Akibatnya: **artefak yang dikirim selalu diambil pada laju belajar penuh**, di
tengah jitter, tidak pernah pada keadaan yang sudah mengendap.

MEKANISMENYA.

Patch ini menambahkan pengendali lingkar-tertutup di dalam loop training:

  1. UKUR throughput sebenarnya - detik per langkah dari waktu dinding yang
     benar-benar berlalu, bukan dari asumsi.
  2. RAMAL langkah akhir: berapa langkah lagi yang muat sebelum jam mati.
  3. RETARGET anneal ke ramalan itu, dan ULANGI ramalannya SETIAP LANGKAH,
     sehingga jika mesin melambat atau mempercepat di tengah run, jendela
     anneal ikut bergeser.

KENAPA INI BUKAN SEKADAR NILAI KNOB. Menyetel `lr_scheduler: cosine` hanyalah
memilih nilai untuk knob yang sudah ada, dan itu TIDAK menyelesaikan masalah di
atas: `total_iters` tetap diikat ke `train_config.steps` (2000) yang tidak
pernah tercapai, sehingga cosine bawaan hanya akan turun sampai sekitar 83% saja
lalu run mati. Tidak ada knob mana pun di ai-toolkit yang mengenal konsep jam
mati - `deadline_ts` bukan bagian dari kontrak konfigurasinya. Yang ditambahkan
di sini adalah UMPAN BALIK RUNTIME: mengukur, meramal, lalu menyesuaikan
sasarannya sendiri sepanjang run. Itu logika baru, bukan pilihan angka.

BENTUK ANNEAL. Cosine dari 1,0 menuju `deadline_anneal_floor` sepanjang
`deadline_anneal_frac` terakhir dari run yang DIRAMALKAN. Lantai dipakai
(bukan nol) supaya model tetap belajar sedikit sampai akhir; ramalan yang
meleset ke arah pesimis tidak membekukan training.

AMAN SAAT RAMALAN BELUM PASTI. Selama `_DA_WARMUP` langkah pertama tidak ada
sampel throughput yang cukup, jadi LR dibiarkan apa adanya. Kalau `deadline_ts`
tidak diberikan (mis. jalur non-krea2), seluruh blok tidak melakukan apa pun.

BIAYA. Beberapa operasi aritmetika skalar per langkah. Tidak ada tensor, tidak
ada sinkronisasi GPU.
"""
import sys

CFG = "/app/ai-toolkit/toolkit/config_modules.py"
PROC = "/app/ai-toolkit/jobs/process/BaseSDTrainProcess.py"

CFG_OLD = """        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.lr_scheduler_params = kwargs.get('lr_scheduler_params', {})"""

CFG_NEW = """        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.lr_scheduler_params = kwargs.get('lr_scheduler_params', {})
        # [deadline_anneal] jam mati absolut (unix ts). 0 = mekanisme tidak aktif.
        self.deadline_ts = kwargs.get('deadline_ts', 0.0)
        self.deadline_anneal_frac = kwargs.get('deadline_anneal_frac', 0.30)
        self.deadline_anneal_floor = kwargs.get('deadline_anneal_floor', 0.05)"""

PROC_OLD = """            self.step_num = step
            # default to true so various things can turn it off
            self.is_grad_accumulation_step = True"""

PROC_NEW = """            self.step_num = step

            # [deadline_anneal] Pengendali lingkar-tertutup: ukur throughput nyata,
            # ramalkan langkah saat jam mati tiba, lalu arahkan anneal LR ke ramalan
            # itu - dan ramalkan ULANG tiap langkah supaya perubahan kecepatan mesin
            # (kami mengukur selisih 19% antar run konfigurasi-identik) ikut terserap.
            # Tanpa ini artefak yang dikirim selalu diambil pada LR penuh, karena
            # jadwal bawaan diikat ke train_config.steps yang tidak pernah tercapai.
            _da_deadline = float(getattr(self.train_config, 'deadline_ts', 0.0) or 0.0)
            if _da_deadline > 0.0:
                import math as _da_math
                import time as _da_time
                if not hasattr(self, '_da_t0'):
                    self._da_t0 = _da_time.time()
                    self._da_step0 = step
                    self._da_base = [g.get('lr', 0.0) for g in optimizer.param_groups]
                    self._da_frac = min(0.9, max(0.05, float(
                        getattr(self.train_config, 'deadline_anneal_frac', 0.30) or 0.30)))
                    self._da_floor = min(1.0, max(0.0, float(
                        getattr(self.train_config, 'deadline_anneal_floor', 0.05) or 0.05)))
                    self._da_said = False
                _da_done = step - self._da_step0
                _DA_WARMUP = 25
                if _da_done >= _DA_WARMUP:
                    _da_elapsed = _da_time.time() - self._da_t0
                    _da_sps = _da_elapsed / float(_da_done)
                    _da_left = _da_deadline - _da_time.time()
                    _da_mult = None
                    if _da_sps > 0.0:
                        if _da_left <= 0.0:
                            # Sudah melewati sasaran anneal (ramalan meleset ke arah
                            # optimis, atau harness memberi kelonggaran di ujung).
                            # TAHAN di lantai - jangan biarkan LR beku di nilai acak
                            # terakhir, dan jangan pula melompat balik ke laju penuh.
                            _da_mult = self._da_floor
                        else:
                            _da_fit = int(_da_left / _da_sps)
                            _da_end = min(int(self.train_config.steps), step + _da_fit)
                            _da_span = max(1.0, self._da_frac * float(_da_end - self._da_step0))
                            _da_into = float(step) - (float(_da_end) - _da_span)
                            if _da_into > 0.0:
                                _da_p = min(1.0, _da_into / _da_span)
                                _da_mult = self._da_floor + (1.0 - self._da_floor) * 0.5 * (
                                    1.0 + _da_math.cos(_da_math.pi * _da_p))
                                if not self._da_said:
                                    self._da_said = True
                                    print('[deadline_anneal] mulai anneal: langkah %d, ramalan akhir %d'
                                          % (step, _da_end), flush=True)
                    if _da_mult is not None:
                        for _da_g, _da_b in zip(optimizer.param_groups, self._da_base):
                            _da_g['lr'] = _da_b * _da_mult

            # default to true so various things can turn it off
            self.is_grad_accumulation_step = True"""


def _patch(path, old, new, tag):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if new in src:
        print("[da] %s sudah terpasang" % tag)
        return
    n = src.count(old)
    if n != 1:
        raise SystemExit("[da] FATAL: anchor muncul %d kali di %s (%s) - butuh tepat 1" % (n, path, tag))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new, 1))
    print("[da] terpasang: %s" % tag)


_patch(CFG, CFG_OLD, CFG_NEW, "field deadline_* di TrainConfig")
_patch(PROC, PROC_OLD, PROC_NEW, "pengendali lingkar-tertutup di loop training")
