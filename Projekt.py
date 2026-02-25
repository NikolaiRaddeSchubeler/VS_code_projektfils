import csv
import os
import statistics
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _open_csv(filepath):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return open(filepath, "r", encoding=encoding, newline="")
        except UnicodeDecodeError:
            continue
    return open(filepath, "r", encoding="utf-8", newline="")


def _detect_dialect(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,|\t,")
    except Exception:
        return csv.get_dialect("excel")


def _normalize_header(text):
    return str(text).strip().lower().replace(" ", "")


def _column_to_index(text):
    if text is None:
        return None
    raw = str(text).strip().upper()
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw) - 1
        return idx if idx >= 0 else None
    if not raw.isalpha():
        return None
    value = 0
    for ch in raw:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def extract_p_mech(filepath, column_hint="D", start_row=1):
    with _open_csv(filepath) as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = _detect_dialect(sample)
        reader = csv.reader(f, dialect=dialect)

        try:
            start_row = int(start_row)
        except (TypeError, ValueError):
            start_row = 1
        if start_row < 1:
            start_row = 1

        col_index = _column_to_index(column_hint)
        p_mech_idx = None
        values = []
        used_header = False

        for row_index, row in enumerate(reader, start=1):
            if row_index < start_row:
                continue
            if not row:
                continue

            if p_mech_idx is None:
                for i, cell in enumerate(row):
                    key = _normalize_header(cell)
                    if "p_mech" in key or key == "pmech":
                        p_mech_idx = i
                        used_header = True
                        values.clear()
                        break
                if p_mech_idx is not None:
                    continue

            if p_mech_idx is not None:
                if p_mech_idx < len(row):
                    val = _parse_float(row[p_mech_idx])
                    if val is not None:
                        values.append(val)
                continue

            if col_index is not None and col_index < len(row):
                val = _parse_float(row[col_index])
                if val is not None:
                    values.append(val)

        if not values:
            if col_index is None and not used_header:
                raise ValueError("P_mech header nicht gefunden. Bitte Spalte angeben (z.B. D oder 4).")
            raise ValueError("Keine gueltigen Zahlenwerte in P_mech gefunden.")

        return values


def summarize_values(values):
    if len(values) == 1:
        return {
            "count": 1,
            "min": values[0],
            "max": values[0],
            "mean": values[0],
            "median": values[0],
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Auswertung - P_mech")
        self.root.geometry("720x520")

        header = tk.Label(
            root,
            text="CSV per Drag & Drop oder Button auswaehlen",
            font=("Segoe UI", 12, "bold"),
        )
        header.pack(pady=(14, 6))

        self.drop_area = tk.Label(
            root,
            text="Dateien hier ablegen",
            relief="groove",
            bd=2,
            width=60,
            height=4,
        )
        self.drop_area.pack(padx=16, pady=6)

        self.select_btn = tk.Button(
            root,
            text="CSV auswaehlen",
            command=self.select_files,
            width=20,
        )
        self.select_btn.pack(pady=(4, 10))

        settings = tk.Frame(root)
        settings.pack(pady=(0, 8))

        tk.Label(settings, text="Spalte (z.B. D oder 4):").grid(row=0, column=0, padx=6, sticky="e")
        self.col_entry = tk.Entry(settings, width=8)
        self.col_entry.insert(0, "D")
        self.col_entry.grid(row=0, column=1, padx=6, sticky="w")

        tk.Label(settings, text="Startzeile (1-basiert):").grid(row=0, column=2, padx=6, sticky="e")
        self.start_entry = tk.Entry(settings, width=8)
        self.start_entry.insert(0, "1")
        self.start_entry.grid(row=0, column=3, padx=6, sticky="w")

        if not DND_AVAILABLE:
            note = tk.Label(
                root,
                text="Drag & Drop benoetigt tkinterdnd2 (optional). Button funktioniert immer.",
                fg="gray",
            )
            note.pack(pady=(0, 8))

        self.output = scrolledtext.ScrolledText(root, height=18, wrap="word")
        self.output.pack(fill="both", expand=True, padx=16, pady=8)

        self._setup_dnd()

    def _setup_dnd(self):
        if not DND_AVAILABLE:
            return
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        files = self._split_dnd_files(event.data)
        self._process_files(files)

    def _split_dnd_files(self, data):
        if hasattr(self.root, "splitlist"):
            return list(self.root.splitlist(data))
        return [data]

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="CSV auswaehlen",
            filetypes=[("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")],
        )
        if files:
            self._process_files(files)

    def _process_files(self, files):
        self.output.delete("1.0", tk.END)
        for path in files:
            self._process_single_file(path)

    def _process_single_file(self, path):
        path = os.path.abspath(path)
        self.output.insert(tk.END, f"Datei: {path}\n")
        try:
            column_hint, start_row = self._get_settings()
            self.output.insert(
                tk.END,
                f"Einstellungen: Spalte={column_hint or '-'} Startzeile={start_row}\n",
            )
            values = extract_p_mech(path, column_hint=column_hint, start_row=start_row)
            stats = summarize_values(values)
            self.output.insert(tk.END, "Ergebnis (P_mech):\n")
            self.output.insert(
                tk.END,
                f"- Anzahl: {stats['count']}\n"
                f"- Min: {stats['min']}\n"
                f"- Max: {stats['max']}\n"
                f"- Mittelwert: {stats['mean']}\n"
                f"- Median: {stats['median']}\n",
            )
            preview = ", ".join(f"{v}" for v in values[:10])
            self.output.insert(tk.END, f"- Erste 10 Werte: {preview}\n\n")
        except Exception as exc:
            self.output.insert(tk.END, f"Fehler: {exc}\n\n")
            messagebox.showerror("Fehler", f"{os.path.basename(path)}: {exc}")

    def _get_settings(self):
        col_hint = self.col_entry.get().strip() if hasattr(self, "col_entry") else ""
        start_raw = self.start_entry.get().strip() if hasattr(self, "start_entry") else "1"
        try:
            start_row = int(start_raw)
        except ValueError:
            start_row = 1
        if start_row < 1:
            start_row = 1
        return col_hint or "D", start_row


def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
