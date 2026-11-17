"""
Analisis Kombinasi Image Enhancement dan Image Filtering
Kondisi citra : Salt & Pepper Noise
Enhancement   : Histogram Equalization
Filtering     : Median Filter

Install: pip install opencv-python numpy matplotlib   (tkinter sudah bawaan Python)
Jalankan: python skenario_enhancement_filtering.py

Menu (panel kiri):
  1. Citra Asli -> Noise Salt & Pepper (+ histogram)   <- tampilan awal
  2. Skenario A : Enhancement saja
  3. Skenario B : Filtering saja
  4. Skenario C : Enhancement -> Filtering
  5. Skenario D : Filtering -> Enhancement
  6. Bandingkan C vs D (asli, noise, hasil C, hasil D berdampingan)
Dropdown "Persentase Noise" dan "Kernel Median" mengubah semua tampilan.

Zoom (berlaku serentak untuk semua citra, agar bagian yang sama bisa dibandingkan):
  - Scroll mouse di atas citra : zoom in / out di posisi kursor
  - Klik-tahan kiri + geser    : pan (geser area pandang)
  - Tombol Zoom + / Zoom - / Reset, atau klik dua kali pada citra untuk reset
"""
import os
import textwrap
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PATH_CITRA = "Tomat.jpg"   # <- citra bersih (atau pilih lewat tombol "Pilih Citra")
SEED = 42
ZOOM_STEP = 0.7            # <1 : setiap zoom in memperkecil area pandang ke 70%
ZOOM_MAKS = 32             # zoom maksimum (32x)


# ---------------- Fungsi pengolahan ----------------
def tambah_salt_pepper(img, persen, acak):
    p = persen / 100.0
    noisy = img.copy()
    noisy[acak < p / 2] = 0          # pepper
    noisy[acak > 1 - p / 2] = 255    # salt
    return noisy


def enhancement(img, level=256):
    """Histogram Equalization sesuai rumus materi kuliah:
         w = (c_w * t) / (n_x * n_y)
       c_w = histogram kumulatif, t = L-1 (threshold keabuan), n_x * n_y = jumlah piksel.
       Hasil dibulatkan ke bilangan bulat terdekat (0,5 dibulatkan ke atas).
       (Catatan: cv2.equalizeHist memakai rumus sedikit berbeda, yaitu mengurangi
        nilai kumulatif terkecil, sehingga angkanya tidak persis sama dengan materi.)
    """
    hist = np.bincount(img.ravel(), minlength=level)
    cdf = np.cumsum(hist)
    w = np.floor(cdf * (level - 1) / img.size + 0.5)
    lut = np.clip(w, 0, level - 1).astype(np.uint8)
    return lut[img]


def ukur(img):
    """Pengukuran intensitas seluruh citra (BUKAN filter): rata-rata (kecerahan),
       nilai tengah, dan standar deviasi (kontras)."""
    return (f"Rata-rata {img.mean():.1f}\nNilai tengah {np.median(img):.0f}\n"
            f"Std {img.std():.1f}")


def filtering(img, k):               # Median Filter
    return cv2.medianBlur(img, k)


# ---------------- Aplikasi ----------------
MENU = [
    ("1. Citra Asli -> Noise Salt & Pepper", "asli"),
    ("2. Skenario A: Enhancement saja", "A"),
    ("3. Skenario B: Filtering saja", "B"),
    ("4. Skenario C: Enhancement -> Filtering", "C"),
    ("5. Skenario D: Filtering -> Enhancement", "D"),
    ("6. Bandingkan C vs D", "CD"),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Image Enhancement & Image Filtering - Salt & Pepper Noise")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1450, sw - 40)}x{min(820, sh - 100)}+10+10")
        try:
            root.state("zoomed")         # maksimalkan (Windows)
        except tk.TclError:
            pass
        self.bersih = None
        self.acak = None
        self.img_axes = []       # axes khusus citra (bukan histogram)
        self.xlim = None         # area pandang bersama (koordinat piksel)
        self.ylim = None
        self.pan_last = None

        # ---- panel kiri ----
        kiri = ttk.Frame(root, padding=10)
        kiri.pack(side="left", fill="y")

        ttk.Button(kiri, text="Pilih Citra...", command=self.pilih_citra).pack(fill="x", pady=2)
        self.lbl_file = ttk.Label(kiri, text="(belum ada citra)", foreground="gray",
                                  wraplength=290)
        self.lbl_file.pack(anchor="w", pady=(0, 2))
        ttk.Button(kiri, text="Simpan Tampilan (PNG)", command=self.simpan).pack(fill="x", pady=2)
        ttk.Separator(kiri).pack(fill="x", pady=10)

        ttk.Label(kiri, text="Persentase Noise", font=("", 10, "bold")).pack(anchor="w")
        self.cb_noise = ttk.Combobox(
            kiri, state="readonly", width=12,
            values=[f"{v}%" for v in (1, 2, 5, 10, 15, 20, 30, 40, 50)])
        self.cb_noise.set("10%")
        self.cb_noise.pack(anchor="w", pady=(0, 10))
        self.cb_noise.bind("<<ComboboxSelected>>", lambda e: self.gambar())

        ttk.Label(kiri, text="Kernel Median Filter", font=("", 10, "bold")).pack(anchor="w")
        self.cb_kernel = ttk.Combobox(kiri, state="readonly", width=12,
                                      values=["3 x 3", "5 x 5", "7 x 7"])
        self.cb_kernel.set("3 x 3")
        self.cb_kernel.pack(anchor="w", pady=(0, 15))
        self.cb_kernel.bind("<<ComboboxSelected>>", lambda e: self.gambar())

        ttk.Label(kiri, text="Menu", font=("", 10, "bold")).pack(anchor="w")
        self.menu_var = tk.StringVar(value="asli")
        for teks, kode in MENU:
            ttk.Radiobutton(kiri, text=teks, value=kode, variable=self.menu_var,
                            command=self.gambar).pack(anchor="w", pady=2)

        ttk.Separator(kiri).pack(fill="x", pady=12)
        ttk.Label(kiri, text="Zoom", font=("", 10, "bold")).pack(anchor="w")
        self.lbl_zoom = ttk.Label(kiri, text="100%")
        self.lbl_zoom.pack(anchor="w")
        fz = ttk.Frame(kiri)
        fz.pack(fill="x", pady=2)
        ttk.Button(fz, text="Zoom +", width=8,
                   command=lambda: self.zoom(ZOOM_STEP)).pack(side="left", padx=(0, 2))
        ttk.Button(fz, text="Zoom -", width=8,
                   command=lambda: self.zoom(1 / ZOOM_STEP)).pack(side="left", padx=2)
        ttk.Button(fz, text="Reset", width=8, command=self.reset_zoom).pack(side="left", padx=2)
        ttk.Label(kiri, text="Scroll = zoom, drag = geser,\nklik 2x = reset",
                  foreground="gray").pack(anchor="w", pady=(2, 0))

        ttk.Separator(kiri).pack(fill="x", pady=12)
        ttk.Label(kiri, text="Penjelasan", font=("", 10, "bold")).pack(anchor="w")
        fi = ttk.Frame(kiri)
        fi.pack(fill="both", expand=True, pady=(2, 0))
        self.txt_info = tk.Text(fi, width=38, height=5, wrap="word", relief="flat",
                                borderwidth=0, bg=root.cget("bg"), font=("", 9),
                                state="disabled", cursor="arrow")
        sb = ttk.Scrollbar(fi, orient="vertical", command=self.txt_info.yview)
        self.txt_info.config(yscrollcommand=sb.set)
        self.txt_info.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")


        # ---- panel kanan (grafik) ----
        try:
            self.fig = Figure(figsize=(10, 6), dpi=100, layout="constrained")
        except TypeError:                # matplotlib lama
            self.fig = Figure(figsize=(10, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        if os.path.exists(PATH_CITRA):
            self.muat(PATH_CITRA)
        else:
            self.pilih_citra()

    # ---- citra ----
    def muat(self, path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
        if img is None:
            messagebox.showerror("Error", f"Citra tidak bisa dibaca:\n{path}")
            return
        self.lbl_file.config(text=os.path.basename(path), foreground="black")
        self.bersih = img
        # medan acak dibuat sekali -> pola noise konsisten antar persentase
        self.acak = np.random.RandomState(SEED).rand(*img.shape)
        self.xlim = self.ylim = None
        self.gambar()

    def pilih_citra(self):
        path = filedialog.askopenfilename(
            title="Pilih citra",
            filetypes=[("Citra", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
                                  "*.JPG", "*.JPEG", "*.PNG")),
                       ("Semua file", "*.*")])
        if path:
            self.muat(path)

    # ---- perhitungan langkah per menu ----
    def langkah(self):
        persen = int(self.cb_noise.get().rstrip("%"))
        k = int(self.cb_kernel.get()[0])
        asli = self.bersih
        noisy = tambah_salt_pepper(asli, persen, self.acak)
        t_noise = f"Citra + Noise Salt & Pepper ({persen}%)"
        m = self.menu_var.get()

        if m == "asli":
            return [("Citra Asli", asli), (t_noise, noisy)]
        if m == "A":
            return [("Citra Asli", asli), (t_noise, noisy),
                    ("A: Histogram Equalization", enhancement(noisy))]
        if m == "B":
            return [("Citra Asli", asli), (t_noise, noisy),
                    (f"B: Median Filter {k}x{k}", filtering(noisy, k))]
        if m == "C":
            he = enhancement(noisy)
            return [("Citra Asli", asli), (t_noise, noisy),
                    ("Langkah 1: Histogram Equalization", he),
                    (f"C: lalu Median Filter {k}x{k}", filtering(he, k))]
        if m == "CD":
            return [("Citra Asli", asli), (t_noise, noisy),
                    (f"C: HE lalu Median {k}x{k}", filtering(enhancement(noisy), k)),
                    (f"D: Median {k}x{k} lalu HE", enhancement(filtering(noisy, k)))]
        mf = filtering(noisy, k)            # skenario D
        return [("Citra Asli", asli), (t_noise, noisy),
                (f"Langkah 1: Median Filter {k}x{k}", mf),
                ("D: lalu Histogram Equalization", enhancement(mf))]

    # ---- gambar ----
    def gambar(self):
        if self.bersih is None:
            return
        h, w = self.bersih.shape
        if self.xlim is None:
            self.xlim, self.ylim = (-0.5, w - 0.5), (h - 0.5, -0.5)

        steps = self.langkah()
        n = len(steps)
        self.fig.clear()
        axs = self.fig.subplots(2, n, squeeze=False)
        self.img_axes = []
        for i, (judul, img) in enumerate(steps):
            judul = "\n".join(textwrap.wrap(judul, 26)) + "\n" + ukur(img)
            ax = axs[0, i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(judul, fontsize=8)
            ax.axis("off")
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            self.img_axes.append(ax)
            hist = np.bincount(img.ravel(), minlength=256)
            axh = axs[1, i]
            axh.vlines(np.arange(256), 0, hist, linewidth=1.2, color="#1f77b4")
            axh.set_xlim(-8, 263)                 # beri ruang agar nilai 0 dan 255 terlihat
            axh.set_ylim(0, hist.max() * 1.15)
            axh.text(0.5, 0.97, f"n(0) = {hist[0]:,} | n(255) = {hist[255]:,}",
                     transform=axh.transAxes, ha="center", va="top", fontsize=7)
            axh.set_title("Histogram", fontsize=8)
            axh.tick_params(labelsize=7)
        self.update_label_zoom()
        self.set_info(self.penjelasan(steps))
        self.canvas.draw_idle()

    def set_info(self, teks):
        self.txt_info.config(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", teks)
        self.txt_info.config(state="disabled")
        self.txt_info.yview_moveto(0)

    # ---- penjelasan inti per tampilan ----
    def penjelasan(self, steps):
        persen = int(self.cb_noise.get().rstrip("%"))
        k = int(self.cb_kernel.get()[0])
        m = self.menu_var.get()
        catatan = ("\n\nAngka pada judul citra adalah pengukuran intensitas seluruh citra, "
                   "bukan filter: Rata-rata = kecerahan, Nilai tengah = median intensitas "
                   "semua piksel, Std = standar deviasi (makin besar umumnya makin tinggi "
                   "kontras). Satu-satunya filter yang dipakai di program ini adalah "
                   "Median Filter.")
        noise_tinggi = ""
        if persen >= 30:
            noise_tinggi = (f"\n\nPerhatian: pada noise setinggi {persen}%, Median Filter "
                            f"{k}x{k} sering belum cukup (sisa bintik masih tampak, seperti "
                            f"contoh noise 50% di materi). Coba kernel lebih besar, tetapi "
                            f"detail akan makin melembut.")

        if m == "asli":
            return (f"Citra asli dibandingkan dengan citra yang diberi noise salt & pepper "
                    f"{persen}%. Sebagian piksel diganti acak menjadi hitam (0) atau putih (255), "
                    f"tampak sebagai bintik pada citra. Di histogram muncul dua lonjakan tinggi "
                    f"di intensitas 0 dan 255." + catatan)
        if m == "A":
            return (f"Skenario A: Histogram Equalization (HE) meratakan sebaran intensitas "
                    f"sehingga kontras meningkat dan histogram melebar (tidak benar-benar rata "
                    f"karena derajat keabuan terbatas). Namun HE tidak membedakan noise dan "
                    f"detail: bintik tetap ada. Bintik putih tetap di 255, sedangkan bintik "
                    f"hitam bergeser menjadi abu gelap karena nilainya ikut dihitung ulang. "
                    f"Rumus: w = (c_w x (L-1)) / jumlah piksel." + catatan)
        if m == "B":
            return (f"Skenario B: Median Filter {k}x{k} mengganti tiap piksel dengan nilai "
                    f"tengah dari piksel itu dan tetangganya. Piksel noise berada di nilai "
                    f"ekstrem sehingga tersingkir: bintik berkurang dan lonjakan 0/255 di "
                    f"histogram menyusut. Harganya, detail halus dan tepi sedikit melembut, "
                    f"serta kontras tidak diperbaiki." + noise_tinggi + catatan)
        if m == "C":
            return (f"Skenario C (HE lalu Median): langkah 1, HE menaikkan kontras, tetapi noise "
                    f"ikut terbawa dan memengaruhi histogram kumulatifnya, sehingga setelah "
                    f"noise dibuang kontras (Std) bisa lebih rendah daripada skenario D. "
                    f"Langkah 2, Median {k}x{k} membuang bintik noise. Bandingkan dengan skenario D (menu 5 atau "
                    f"menu 6): lihat perbedaan sisa bintik, detail, dan kontras pada citra dan "
                    f"histogram." + noise_tinggi + catatan)
        if m == "D":
            return (f"Skenario D (Median lalu HE): langkah 1, Median {k}x{k} membuang noise lebih "
                    f"dulu. Langkah 2, HE memperbaiki kontras dari distribusi intensitas yang "
                    f"sudah lebih bebas dari lonjakan 0/255. Bandingkan dengan skenario C (menu 4 "
                    f"atau menu 6): lihat perbedaan sisa bintik, detail, dan kontras pada citra "
                    f"dan histogram." + noise_tinggi + catatan)
        return (f"Perbandingan C vs D pada citra yang sama. C: HE dulu, lalu Median. D: Median "
                f"dulu, lalu HE. Perhatikan (1) sisa bintik noise, (2) detail dan tepi, "
                f"(3) bentuk histogram, terutama lonjakan di 0 dan 255, serta (4) Mean "
                f"(kecerahan) dan Std (kontras) pada judul tiap citra. Gunakan zoom untuk "
                f"membandingkan area yang sama. Jadikan pengamatan ini dasar jawaban soal "
                f"analisis nomor 3 dan 4." + noise_tinggi + catatan)

    # ---- zoom & pan ----
    def terapkan_view(self):
        for ax in self.img_axes:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
        self.update_label_zoom()
        self.canvas.draw_idle()

    def update_label_zoom(self):
        if self.bersih is None or self.xlim is None:
            return
        w = self.bersih.shape[1]
        self.lbl_zoom.config(text=f"{w / (self.xlim[1] - self.xlim[0]) * 100:.0f}%")

    def zoom(self, faktor, cx=None, cy=None):
        """faktor < 1 = zoom in, > 1 = zoom out; pusat zoom (cx, cy) dalam koordinat piksel."""
        if self.bersih is None:
            return
        h, w = self.bersih.shape
        x0, x1 = self.xlim
        y1, y0 = self.ylim          # ylim terbalik (atas = kecil)
        if cx is None:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lebar = min(max((x1 - x0) * faktor, w / ZOOM_MAKS), w)
        tinggi = min(max((y1 - y0) * faktor, h / ZOOM_MAKS), h)
        # pertahankan posisi relatif kursor terhadap area pandang
        rx = (cx - x0) / (x1 - x0)
        ry = (cy - y0) / (y1 - y0)
        nx0 = cx - rx * lebar
        ny0 = cy - ry * tinggi
        self.set_view(nx0, ny0, lebar, tinggi)

    def set_view(self, x0, y0, lebar, tinggi):
        h, w = self.bersih.shape
        x0 = min(max(x0, -0.5), w - 0.5 - lebar)
        y0 = min(max(y0, -0.5), h - 0.5 - tinggi)
        self.xlim = (x0, x0 + lebar)
        self.ylim = (y0 + tinggi, y0)
        self.terapkan_view()

    def reset_zoom(self):
        if self.bersih is None:
            return
        h, w = self.bersih.shape
        self.xlim, self.ylim = (-0.5, w - 0.5), (h - 0.5, -0.5)
        self.terapkan_view()

    def on_scroll(self, e):
        if e.inaxes in self.img_axes and e.xdata is not None:
            self.zoom(ZOOM_STEP if e.button == "up" else 1 / ZOOM_STEP, e.xdata, e.ydata)

    def on_press(self, e):
        if e.inaxes not in self.img_axes:
            return
        if e.dblclick:
            self.reset_zoom()
        elif e.button == 1:
            self.pan_last = (e.x, e.y, e.inaxes)

    def on_motion(self, e):
        if self.pan_last is None:
            return
        px, py, ax = self.pan_last
        x0, x1 = self.xlim
        y1, y0 = self.ylim
        bb = ax.bbox
        dx = (e.x - px) * (x1 - x0) / bb.width
        dy = (e.y - py) * (y1 - y0) / bb.height
        # geser berlawanan arah gerak mouse (efek "menyeret" citra); sumbu y piksel terbalik
        self.set_view(x0 - dx, y0 + dy, x1 - x0, y1 - y0)
        self.pan_last = (e.x, e.y, ax)

    def on_release(self, e):
        self.pan_last = None

    def simpan(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=f"hasil_{self.menu_var.get()}.png")
        if path:
            self.fig.savefig(path, dpi=150)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()