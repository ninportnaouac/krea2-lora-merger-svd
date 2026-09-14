
from __future__ import annotations
import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

from lora_to_lora_merge import scan_file, merge_standard_loras

MAX_LORAS = 8

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Krea2 LoRA → LoRA Merger + SVD")
        self.geometry("900x720")
        self.minsize(820, 650)

        self.files = []
        self.weights = []

        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Krea2 LoRA → LoRA Merger + SVD", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="Fusionne plusieurs LoRA Krea2. Mode Exact = plus gros ; Compressé SVD = rank réduit."
        ).pack(anchor="w", pady=(0, 10))

        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="Ajouter LoRA", command=self.add_loras).pack(side="left")
        ttk.Button(btns, text="Retirer sélection", command=self.remove_selected).pack(side="left", padx=6)
        ttk.Button(btns, text="Vérifier les LoRA", command=self.check_files).pack(side="left")

        self.tree = ttk.Treeview(top, columns=("file","weight","type"), show="headings", height=10)
        self.tree.heading("file", text="Fichier")
        self.tree.heading("weight", text="Poids")
        self.tree.heading("type", text="Type")
        self.tree.column("file", width=520)
        self.tree.column("weight", width=80, anchor="center")
        self.tree.column("type", width=170, anchor="center")
        self.tree.pack(fill="x", pady=10)

        weight_frame = ttk.Frame(top)
        weight_frame.pack(fill="x")
        ttk.Label(weight_frame, text="Poids du LoRA sélectionné :").pack(side="left")
        self.weight_var = tk.DoubleVar(value=1.0)
        self.weight_spin = ttk.Spinbox(weight_frame, from_=-2.0, to=2.0, increment=0.05, textvariable=self.weight_var, width=8)
        self.weight_spin.pack(side="left", padx=5)
        ttk.Button(weight_frame, text="Appliquer", command=self.apply_weight).pack(side="left")

        mode_frame = ttk.LabelFrame(top, text="Mode de merge", padding=8)
        mode_frame.pack(fill="x", pady=8)
        self.mode_var = tk.StringVar(value="compressed")
        ttk.Radiobutton(mode_frame, text="Exact (qualité totale, fichier plus gros)", value="exact", variable=self.mode_var, command=self.update_mode).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Compressé SVD (recommandé)", value="compressed", variable=self.mode_var, command=self.update_mode).grid(row=1, column=0, sticky="w")
        ttk.Label(mode_frame, text="Rank final :").grid(row=1, column=1, padx=(25,5))
        self.rank_var = tk.StringVar(value="64")
        self.rank_combo = ttk.Combobox(mode_frame, textvariable=self.rank_var, values=["16","32","64","128","256"], width=8, state="readonly")
        self.rank_combo.grid(row=1, column=2, sticky="w")
        ttk.Label(mode_frame, text="Conseil : rank 64 = bon compromis taille/qualité").grid(row=2, column=0, columnspan=3, sticky="w", pady=(5,0))

        out_frame = ttk.LabelFrame(top, text="Sortie", padding=8)
        out_frame.pack(fill="x", pady=8)

        ttk.Label(out_frame, text="Dossier :").grid(row=0, column=0, sticky="w")
        self.out_dir = tk.StringVar(value=os.path.abspath("./output_lora"))
        ttk.Entry(out_frame, textvariable=self.out_dir).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(out_frame, text="Parcourir", command=self.choose_out_dir).grid(row=0, column=2)

        ttk.Label(out_frame, text="Nom :").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.out_name = tk.StringVar(value="Krea2_Merged_LoRA.safetensors")
        ttk.Entry(out_frame, textvariable=self.out_name).grid(row=1, column=1, sticky="ew", padx=5, pady=(6,0))

        ttk.Label(out_frame, text="Trigger :").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.trigger_var = tk.StringVar(value="")
        ttk.Entry(out_frame, textvariable=self.trigger_var).grid(row=2, column=1, sticky="ew", padx=5, pady=(6,0))
        ttk.Label(out_frame, text="Ex: cyrano, funnycrushcomic").grid(row=2, column=2, sticky="w", pady=(6,0))

        self.trigger_txt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_frame, text="Créer aussi un fichier .txt avec le trigger",
                        variable=self.trigger_txt_var).grid(row=3, column=1, columnspan=2, sticky="w", padx=5, pady=(4,0))

        ttk.Label(out_frame, text="Précision :").grid(row=4, column=0, sticky="w", pady=(6,0))
        self.dtype_var = tk.StringVar(value="fp16")
        ttk.Combobox(out_frame, textvariable=self.dtype_var, values=["fp16","bf16","fp32"], state="readonly", width=10).grid(row=4, column=1, sticky="w", padx=5, pady=(6,0))
        out_frame.columnconfigure(1, weight=1)

        action = ttk.Frame(top)
        action.pack(fill="x", pady=8)
        self.merge_btn = ttk.Button(action, text="MERGER LES LoRA", command=self.start_merge)
        self.merge_btn.pack(side="left")
        ttk.Button(action, text="Fermer", command=self.destroy).pack(side="right")

        self.progress = ttk.Progressbar(top, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0,8))

        ttk.Label(top, text="Journal :").pack(anchor="w")
        self.log = tk.Text(top, height=18, wrap="word")
        self.log.pack(fill="both", expand=True)

        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.update_mode()

    def write(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.update_idletasks()

    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, path in enumerate(self.files):
            kind = "?"
            try:
                s = scan_file(path)
                if s.lokr_layers:
                    kind = f"LoKr ({s.lokr_layers})"
                else:
                    kind = f"LoRA ({s.lora_layers})"
            except Exception:
                kind = "Erreur"
            self.tree.insert("", "end", iid=str(i), values=(os.path.basename(path), f"{self.weights[i]:.2f}", kind))

    def add_loras(self):
        paths = filedialog.askopenfilenames(title="Choisir les LoRA", filetypes=[("Safetensors","*.safetensors")])
        for p in paths:
            if p not in self.files and len(self.files) < MAX_LORAS:
                self.files.append(p)
                self.weights.append(1.0)
        self.refresh_tree()

    def remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        idxs = sorted((int(x) for x in sel), reverse=True)
        for i in idxs:
            del self.files[i]
            del self.weights[i]
        self.refresh_tree()

    def on_select(self, _evt=None):
        sel = self.tree.selection()
        if sel:
            self.weight_var.set(self.weights[int(sel[0])])

    def apply_weight(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Sélectionne un LoRA dans la liste.")
            return
        i = int(sel[0])
        self.weights[i] = float(self.weight_var.get())
        self.refresh_tree()

    def update_mode(self):
        self.rank_combo.config(state="readonly" if self.mode_var.get() == "compressed" else "disabled")

    def choose_out_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.out_dir.set(d)

    def check_files(self):
        self.log.delete("1.0", "end")
        if not self.files:
            self.write("Ajoute au moins un LoRA.")
            return
        for p in self.files:
            try:
                r = scan_file(p)
                self.write(f"— {r.filename} —")
                self.write(f"  LoRA standard : {r.lora_layers}")
                self.write(f"  LoKr          : {r.lokr_layers}")
                self.write(f"  Incomplets    : {len(r.incomplete)}")
                if r.lokr_layers:
                    self.write("  ⚠️ LoKr détecté : pas de merge exact LoRA→LoRA pour ce fichier.")
                else:
                    self.write("  ✅ Compatible avec le merge exact LoRA→LoRA.")
                self.write("")
            except Exception as e:
                self.write(f"❌ {os.path.basename(p)} : {e}")

    def start_merge(self):
        if not self.files:
            messagebox.showerror("Erreur", "Ajoute au moins un LoRA.")
            return
        self.merge_btn.config(state="disabled")
        self.progress.start(10)
        threading.Thread(target=self.do_merge, daemon=True).start()

    def do_merge(self):
        try:
            outdir = self.out_dir.get().strip()
            os.makedirs(outdir, exist_ok=True)
            name = self.out_name.get().strip() or "Krea2_Merged_LoRA.safetensors"
            if not name.lower().endswith(".safetensors"):
                name += ".safetensors"
            out = os.path.join(outdir, name)

            dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }[self.dtype_var.get()]

            compress = self.mode_var.get() == "compressed"
            rank = int(self.rank_var.get())
            mode_text = f"Compressé SVD rank {rank}" if compress else "Exact"
            self.after(0, lambda: self.write(f"Début du merge — {mode_text}..."))
            result = merge_standard_loras(
                self.files,
                self.weights,
                out,
                out_dtype=dtype,
                compress=compress,
                target_rank=rank,
                trigger_words=self.trigger_var.get(),
                write_trigger_txt=self.trigger_txt_var.get(),
            )

            def done():
                self.write("")
                self.write("✅ Merge terminé.")
                self.write(f"Sortie : {result['output']}")
                self.write(f"Couches fusionnées : {result['merged_layers']}")
                self.write(f"Contributions LoRA : {result['contributors_total']}")
                self.write(f"Rank exact : {result['exact_rank_min']} → {result['exact_rank_max']}")
                self.write(f"Rank final : {result['final_rank_min']} → {result['final_rank_max']}")
                if result["compressed"]:
                    self.write(f"Énergie SVD conservée moyenne : {result['average_energy']*100:.2f}%")
                if result.get("trigger_words"):
                    self.write(f"Trigger enregistré : {result['trigger_words']}")
                if result.get("trigger_txt"):
                    self.write(f"Fichier trigger : {result['trigger_txt']}")
                self.write(f"Shape warnings : {len(result['shape_warnings'])}")
                messagebox.showinfo("Terminé", "Le LoRA fusionné a été créé.")
            self.after(0, done)

        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: self.write("❌ Erreur :\n" + err))
        finally:
            self.after(0, self.progress.stop)
            self.after(0, lambda: self.merge_btn.config(state="normal"))


if __name__ == "__main__":
    App().mainloop()
