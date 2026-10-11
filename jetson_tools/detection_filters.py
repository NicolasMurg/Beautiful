"""Local category selection and native checkbox controls for both camera views."""
import json
from pathlib import Path
import threading


class DetectionFilters:
    def __init__(self, names, path):
        self.names = {int(k): v for k, v in names.items()}
        self.path = Path(path)
        self.lock = threading.Lock()
        self.revision = 0
        try:
            disabled = set(json.loads(self.path.read_text()).get('disabled', []))
        except (OSError, ValueError, TypeError):
            disabled = set()
        self.enabled = frozenset(k for k, v in self.names.items() if v not in disabled)

    def snapshot(self):
        with self.lock:
            return self.enabled, self.revision

    def set_enabled(self, enabled):
        enabled = frozenset(enabled) & self.names.keys()
        with self.lock:
            if enabled == self.enabled:
                return
            self.enabled = enabled
            self.revision += 1
            disabled = [v for k, v in self.names.items() if k not in enabled]
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({'disabled': disabled}, indent=2) + '\n')
        temp.replace(self.path)


def filter_result(result, enabled):
    """Keep boxes, masks and records aligned using Ultralytics Results slicing."""
    if result.boxes is None:
        return result
    classes = result.boxes.cls
    keep = classes == -1
    keep &= False
    for class_id in enabled:
        keep |= classes == class_id
    return result[keep]


BALL_CATEGORIES = frozenset(('ball', 'sports ball', 'billiard ball', 'cue ball', 'pool ball'))


def filter_scores(result, default_conf, ball_conf):
    """Use the ball threshold only for ball labels, preserving actual model scores."""
    if result.boxes is None:
        return result
    classes, scores = result.boxes.cls, result.boxes.conf
    ball = classes == -1
    ball &= False
    for class_id, name in result.names.items():
        if name in BALL_CATEGORIES:
            ball |= classes == class_id
    return result[((ball & (scores >= ball_conf)) | (~ball & (scores >= default_conf)))]


class FilterPanel:
    def __init__(self, filters, on_change, on_close):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.filters = tk, filters
        self.root = tk.Tk()
        self.root.title('Detection filters - both cameras')
        panel_x = max(0, self.root.winfo_screenwidth() - 380)
        self.root.geometry(f'350x660+{panel_x}+40')
        self.root.protocol('WM_DELETE_WINDOW', on_close)
        ttk.Label(self.root, text='Detection categories', font=('', 15, 'bold')).pack(anchor='w', padx=14, pady=(14, 4))
        ttk.Label(self.root, text='Checked = show boxes in both camera views').pack(anchor='w', padx=14)
        self.search = tk.StringVar()
        entry = ttk.Entry(self.root, textvariable=self.search)
        entry.pack(fill='x', padx=14, pady=10)
        entry.insert(0, '')
        buttons = ttk.Frame(self.root)
        buttons.pack(fill='x', padx=14)
        ttk.Button(buttons, text='All on', command=lambda: self.select_all(True)).pack(side='left')
        ttk.Button(buttons, text='All off', command=lambda: self.select_all(False)).pack(side='left', padx=6)
        self.count = ttk.Label(self.root)
        self.count.pack(anchor='w', padx=14, pady=8)
        area = ttk.Frame(self.root)
        area.pack(fill='both', expand=True, padx=14, pady=(0, 14))
        self.canvas = tk.Canvas(area, highlightthickness=0)
        scrollbar = ttk.Scrollbar(area, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.list_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.list_frame, anchor='nw')
        self.list_frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Button-4>', lambda e: self.canvas.yview_scroll(-3, 'units'))
        self.canvas.bind('<Button-5>', lambda e: self.canvas.yview_scroll(3, 'units'))
        enabled, _ = filters.snapshot()
        self.variables = {k: tk.BooleanVar(value=k in enabled) for k in filters.names}
        self.rows = []
        preferred = ('ball', 'billiard ball', 'cue ball', 'sports ball', 'diamond shape', 'beverage can', 'bottle', 'cup', 'person')
        order = sorted(filters.names, key=lambda k: (filters.names[k] not in preferred, filters.names[k]))
        for k in order:
            row = ttk.Checkbutton(self.list_frame, text=filters.names[k], variable=self.variables[k], command=self.changed)
            row.pack(anchor='w', fill='x', pady=3)
            self.rows.append((filters.names[k], row))
        self.on_change = on_change
        self.search.trace_add('write', self.search_changed)
        self.refresh_count()

    def refresh_count(self):
        active, _ = self.filters.snapshot()
        self.count.configure(text=f'{len(active)} of {len(self.variables)} categories enabled')

    def changed(self):
        self.filters.set_enabled(k for k, v in self.variables.items() if v.get())
        self.refresh_count()
        self.on_change()

    def select_all(self, value):
        for var in self.variables.values():
            var.set(value)
        self.changed()

    def search_changed(self, *_):
        term = self.search.get().strip().casefold()
        for name, row in self.rows:
            row.pack_forget()
        for name, row in self.rows:
            if term in name.casefold():
                row.pack(anchor='w', fill='x', pady=3)
        self.canvas.yview_moveto(0)

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(700, lambda: self.root.attributes('-topmost', False))

    def poll(self):
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()
