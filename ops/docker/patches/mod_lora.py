# ---------------------------------------------------------------------------
# ORIGIN: rivals/god-image-tourn-1dccbd24755b58a3-20260907-position-1/ops/docker/patches/mod_lora.py
#         (2026-09-07 image tournament winner, hotkey 5EhyCWPu). Copied VERBATIM on
#         2026-09-08 for the I6 winner-recipe port (PLAN-IMAGE-0908 v2); everything
#         below this header is the unmodified original.
# LICENSE: the source repo ships Apache-2.0 (LICENSE.md) plus the Gradients-on-Demand
#         NOTICE (Copyright 2025 Grads LLC; attribution to Gradients.io required) -
#         see this repo's LICENSE.md / NOTICE.
# GATE:   env KREA2_MODLORA (default "0" -> get_transformer_block_names returns ["blocks"] exactly as upstream)
#         -> applying at build time changes nothing until IMG_RECIPE=winner sets it
#            (ops/docker/scripts/image_toolkit_entrypoint.py).
# ---------------------------------------------------------------------------
"""Build-time patch: MODULATION-PATH LoRA (`mod_lora`) untuk krea2.

KENAPA. Kami menemukan bahwa sebagian jalur krea2 TIDAK PERNAH diadaptasi sama
sekali - gradiennya persis NOL - dan itu bukan pilihan desain kami, melainkan
efek samping default yang tidak pernah diperiksa siapa pun.

  toolkit/lora_special.py:367-377
      if self.transformer_only and is_unet:
          transformer_block_names = base_model.get_transformer_block_names()
          if not any([bn in clean_name for bn in transformer_block_names]):
              skip = True          # <- tidak dipasangi LoRA sama sekali

  extensions_built_in/diffusion_models/krea2/krea2.py:461-462
      def get_transformer_block_names(self):
          return ["blocks"]

  toolkit/config_modules.py:201
      self.transformer_only = kwargs.get('transformer_only', True)   # default True
  dan base_diffusion_krea2.yaml TIDAK menyetelnya  =>  gerbang ini AKTIF.

Konsekuensinya: setiap modul yang namanya tidak memuat "blocks" di-SKIP. Itu
mencakup tiga modul di jalur maju:

  src/mmdit.py:374  self.tmlp  = Linear(tdim -> 6144) + GELU + Linear(6144 -> 6144)
  src/mmdit.py:395  self.tproj = GELU + Linear(6144 -> 36864)
  src/mmdit.py:393  self.last  = LastLayer(6144, patch, channels)

  src/mmdit.py:422-423
      t    = self.tmlp(temb(t, ...))
      tvec = self.tproj(t)

KENAPA JUSTRU INI YANG MENARIK. Input `tproj` HANYA timestep - bukan konten
gambar - dan keluarannya adalah modulasi 6-arah yang dipakai SERENTAK oleh
seluruh 28 blok. Jadi LoRA di `tmlp`+`tproj` = koreksi modulasi global yang
TERKONDISI-TIMESTEP, yang **dipelajari model sendiri**.

Bandingkan dengan arm penjadwalan timestep yang sudah kami bakar:
  - `KREA2_TSTYPE=sigmoid`  -> RUGI +1,74%, dan BALIK ARAH di leave-one-out
  - hardcode jumlah langkah -> semua aturan tetap RUGI (optimum pindah antar-run)
Keduanya mencoba mendapatkan derajat kebebasan per-timestep dengan MENEBAK
jadwalnya lalu menghardcode. Patch ini memberi derajat kebebasan yang sama
tetapi membiarkan gradien yang menemukannya - kebal terhadap tebakan jadwal
yang salah, dan itulah tepatnya yang membunuh `sigmoid`.

BATAS YANG JUJUR. Karena `tproj` buta konten, dia TIDAK BISA menambah detail
per-gambar; plafonnya adalah koreksi modulasi. `last.linear` (6144 -> patch^2*ch,
langsung ke laten) yang membawa satu-satunya jalur konten-dependen di sini.

SENGAJA TIDAK DIIKUTKAN:
  -  : DIBUANG SETELAH UJI. Pencocokan di lora_special.py:376 adalah
    SUBSTRING (), dan "tmlp" adalah substring dari
    "txtmlp" -- juga "tmlp." adalah substring dari "txtmlp.1". Memasukkannya
    diam-diam membuka jalur TEKS. Terbukti lewat uji kering sebelum run.
    Tidak ada string pembeda yang aman, jadi tmlp dilepas.
  - `txtmlp` / `txtfusion` : jalur TEKS. Menyentuhnya akan mengacaukan pembacaan
    diferensial kanal text (bobot 0,25) vs no-text (bobot 0,75) yang jadi dasar
    hampir semua vonis kami.
  - `first`                : proyeksi masuk; tidak terkondisi timestep.
  - modulasi berbasis nn.Parameter (bukan nn.Linear) : LoRA fisik tidak bisa
    menyentuhnya.

BIAYA. s/langkah praktis tidak berubah: `tproj`/`tmlp` jalan SEKALI per forward
(bukan 28 kali), pada tensor 1-token. Parameter LoRA tambahan pada rank 32:
  tproj  32*(6144+36864) = 1,38 M
  tmlp   32*(256+6144) + 32*(6144+6144) = 0,60 M
  last   32*(6144+64)   = 0,20 M
  total  ~2,2 M  di atas basis blocks - tidak material untuk waktu langkah.

DEFAULT MATI. `KREA2_MODLORA=0` (default) mereproduksi perilaku sekarang persis.
`KREA2_MODLORA=1` membuka jalur modulasi.
"""
import os
import sys

TARGET = "/app/ai-toolkit/extensions_built_in/diffusion_models/krea2/krea2.py"

ANCHOR = '''    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]'''

REPLACEMENT = '''    # [mod_lora] Modul di bawah ini menerima gradien PERSIS NOL dengan
    # transformer_only=True (default) karena namanya tidak memuat "blocks":
    #   tproj       GELU + Linear(6144 -> 36864)  -> modulasi 6-arah untuk 28 blok,
    #                                                input HANYA timestep
    #   tmlp        Linear(tdim->6144)+GELU+Linear -> embedder timestep pemberi tproj
    #   last.linear Linear(6144 -> patch^2*ch)     -> proyeksi keluaran terakhir
    # Membukanya memberi model derajat kebebasan modulasi per-timestep yang
    # DIPELAJARI, bukan dijadwalkan-dengan-tebakan seperti arm sigmoid yang gagal.
    MODULATION_LORA_TARGETS = ["tproj", "last.linear"]

    def get_transformer_block_names(self) -> Optional[List[str]]:
        names = ["blocks"]
        if os.environ.get("KREA2_MODLORA", "0") not in ("0", "", "false", "False"):
            names = names + list(self.MODULATION_LORA_TARGETS)
            # OPSEC: metode ini dipanggil ratusan kali per run, dan log kami masuk
            # ke instance Loki yang TERBUKA TANPA AUTH (lawan membacanya). Mencetak
            # nama mekanisme kami tiap panggilan = mengiklankan lever baru ke lapangan.
            # Cetak SEKALI, dan hanya kalau diminta eksplisit.
            if os.environ.get("KREA2_MODLORA_VERBOSE", "0") == "1":
                cls = type(self)
                if not getattr(cls, "_modlora_announced", False):
                    cls._modlora_announced = True
                    print("[mod_lora] jalur modulasi dibuka", flush=True)
        return names'''


def main():
    if not os.path.isfile(TARGET):
        print("[mod_lora] TARGET tidak ada: %s" % TARGET, file=sys.stderr)
        return 1
    src = open(TARGET, encoding="utf-8").read()

    if "[mod_lora]" in src:
        print("[mod_lora] sudah terpasang, lewati")
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print("[mod_lora] GAGAL: anchor muncul %d kali (butuh tepat 1)" % n, file=sys.stderr)
        return 1

    src = src.replace(ANCHOR, REPLACEMENT, 1)

    if "\nimport os" not in src and not src.startswith("import os"):
        src = "import os\n" + src
        print("[mod_lora] menambahkan 'import os'")

    open(TARGET, "w", encoding="utf-8").write(src)
    print("[mod_lora] terpasang di %s" % TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
