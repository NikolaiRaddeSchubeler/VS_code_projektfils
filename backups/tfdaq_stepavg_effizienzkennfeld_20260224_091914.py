# Find intervals of a given parameter being constant and average selected parameters in data files
#!/usr/bin/env python

import sys
sys.path.append('../lib')
import os  
from pathlib import Path
import csv
import bisect
import matplotlib.pyplot as plt
import numpy as np
import tfdaq_stepfind
import re
from datetime import datetime
#-------------------------------------------------------------------------------

## User settings

# Default data path, can be overriden by user input
default_data_path = '../data'

# Search data path recursively, can be overriden by user input
path_recursive = False

# Parameter used for detecting step_list
stepfind_param = 'N_mot' # Parameter checked for being constant within interval
stepfind_interval = 8    # Fixed interval width in seconds to find
stepfind_std_lim = 10    # Limit value for (standard) deviation of parameter within interval
stepfind_threshold = 7000 # Ignore steps where parameter is below this value

# Parameters to average (any of the parameters available in tfdaq_sensorconfig can be added)
#avg_params = ['N_mot', 'P_dc', 'T_winding_0', 'U_dc', 'I_dc', 'I_q', 'I_d' 'P_mech']
avg_params = ['N_mot', 'rho_air', 'p_amb', 'T_amb', 'P_mech', 'U_dc', 'I_dc', 'p_diff_1', 'p_diff_2', 'p_diff_3', 'p_diff_4', 'M_mot']

# Enable individual plots
plot_en = False
#-------------------------------------------------------------------------------

## Global variables
avg_params_sem = [f'{p}_SEM' for p in avg_params]
derived_params = [
  'A_impeller',
  'v_1_mean', 'v_2_mean',
  'm_dot_1_mean', 'm_dot_2_mean',
  'p_sta_1', 'p_sta_1_SEM',
  'P_dc_mean', 'eta_mech',
  'T_amb_K_mean', 'p_amb_mean_pa',
  'N_korr',
  'p1_abs_mean', 'p2_abs_mean',
  'k1', 'k2',
  'p2_p1_ratio',
  'm_dot_1_norm', 'm_dot_2_norm',
  'u_v_1_mean', 'u_v_2_mean',
  'u_m_dot_1_mean', 'u_m_dot_2_mean',
  'u_m_dot_1_norm', 'u_m_dot_2_norm',
  'u_p2_p1_ratio'
]
avg_data_columns = ['FILE', 'ROI_START', 'ROI_STOP'] + avg_params + avg_params_sem + derived_params
avg_params_units = ['' for n in range(len(avg_params))]
avg_params_sem_units = ['' for n in range(len(avg_params))]
derived_params_units = [
  'm^2',
  'm/s', 'm/s',
  'kg/s', 'kg/s',
  'Pa', 'Pa',
  'W', '',
  'K', 'Pa',
  'rpm',
  'Pa', 'Pa',
  '', '',
  '',
  'kg/s', 'kg/s',
  'm/s', 'm/s',
  'kg/s', 'kg/s',
  'kg/s', 'kg/s',
  ''
]
avg_data_units = ['', 's', 's']
avg_data = []
#-------------------------------------------------------------------------------

IMP_AREA = np.pi * 0.125**2
T_NORM = 288.15
P_NORM = 101325.0

def _to_pa(value, unit):
  if unit is None:
    return value
  u = unit.strip().lower()
  if u == 'pa':
    return value
  if u == 'hpa':
    return value * 100.0
  if u == 'kpa':
    return value * 1000.0
  if u == 'mbar':
    return value * 100.0
  if u == 'bar':
    return value * 100000.0
  return value

def calc_velocity_and_massflow(p_dyn_pa, rho, area):
  v = np.sqrt(2.0 * p_dyn_pa / rho)
  m_dot = rho * v * area
  return v, m_dot

def _to_kelvin(t_c):
  return float(np.asarray(t_c, dtype = float) + 273.15)

def _corr_factor(t_k, p_abs_pa):
  if p_abs_pa <= 0:
    raise ValueError(f'p_abs_pa muss > 0 sein, ist aber {p_abs_pa}. Einheiten/Offsets prüfen.')
  return float(np.sqrt((t_k / T_NORM) * (P_NORM / p_abs_pa)))

def _safe_float(val):
  if val == '' or val is None:
    return None
  try:
    return float(val)
  except:
    return None

def _u_v_from_pdyn_rho(v_mean, u_p_dyn, rho_mean, u_rho):
  if v_mean == 0 or rho_mean == 0:
    return None
  dv_dp = 1.0 / (rho_mean * v_mean)
  dv_dr = -v_mean / (2.0 * rho_mean)
  return float(np.sqrt((dv_dp * u_p_dyn)**2 + (dv_dr * u_rho)**2))

def _u_mdot_from_rho_v(rho_mean, u_rho, v_mean, u_v, area=IMP_AREA):
  dmdot_drho = area * v_mean
  dmdot_dv = area * rho_mean
  return float(np.sqrt((dmdot_drho * u_rho)**2 + (dmdot_dv * u_v)**2))

def _u_k_from_T_p(k_mean, t_k_mean, u_t_k, p_abs_mean, u_p_abs):
  rel = 0.5 * np.sqrt((u_t_k / t_k_mean)**2 + (u_p_abs / p_abs_mean)**2)
  return float(k_mean * rel)

def _u_mnorm_from_m_k(m_mean, u_m, k_mean, u_k):
  return float(np.sqrt((k_mean * u_m)**2 + (m_mean * u_k)**2))

def _sort_key(row):
  # Group by rpm, then Blende: Blende0 first, then descending. Unknowns go last.
  file_idx = avg_data_columns.index('FILE')
  file_val = row[file_idx] if file_idx < len(row) else ''
  base = os.path.basename(file_val) if file_val else ''
  blende_match = re.search(r'_Blende(\d+)_', base)
  rpm_match = re.search(r'(\d+)\s*rpm', base, re.IGNORECASE)
  if not rpm_match:
    return (1, float('inf'), float('inf'), base)
  rpm_num = int(rpm_match.group(1))
  blende_num = int(blende_match.group(1)) if blende_match else float('inf')
  # Blende0 first, then higher numbers descending within same rpm
  if blende_num == 0:
    blende_key = (0, 0)
  else:
    blende_key = (1, -blende_num)
  return (0, rpm_num, blende_key, base)

def _extract_rpm(file_path):
  base = os.path.basename(file_path) if file_path else ''
  rpm_match = re.search(r'(\d+)\s*rpm', base, re.IGNORECASE)
  return int(rpm_match.group(1)) if rpm_match else None


def _collect_csv_files(input_path, recursive=False):
  if not input_path:
    return []
  if os.path.isdir(input_path):
    if recursive:
      return [str(f) for f in Path(input_path).rglob('*.csv') if (f.is_file() and ('_avg.csv' not in str(f)))]
    return [str(f) for f in Path(input_path).glob('*.csv') if (f.is_file() and ('_avg.csv' not in str(f)))]
  if os.path.isfile(input_path) and str(input_path).lower().endswith('.csv'):
    return [str(input_path)]
  return []


def _scan_speedlines_from_files(files):
  rpms = []
  seen = set()
  for fp in files:
    rpm = _extract_rpm(fp)
    if rpm is None:
      continue
    rpm_i = int(abs(rpm))
    if rpm_i not in seen:
      seen.add(rpm_i)
      rpms.append(rpm_i)
  rpms.sort()
  return rpms


def _startup_gui(default_path='.', recursive_default=False):
  try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
  except Exception as e:
    print(f'  note: GUI not available, fallback to console input ({e}).')
    return None

  result = {'cancelled': True}

  root = tk.Tk()
  root.title('tfdaq Auswertung Start')
  root.geometry('820x640')

  path_var = tk.StringVar(value=str(default_path))
  recursive_var = tk.BooleanVar(value=bool(recursive_default))
  export_csv_var = tk.BooleanVar(value=True)
  create_eff_var = tk.BooleanVar(value=True)
  create_motor_var = tk.BooleanVar(value=True)
  create_torque_var = tk.BooleanVar(value=True)
  export_png_var = tk.BooleanVar(value=True)
  export_pdf_var = tk.BooleanVar(value=True)
  info_var = tk.StringVar(value='Bitte Pfad wählen und Speedlines scannen.')
  rpm_values = []

  top_frame = tk.LabelFrame(root, text='Datenpfad', padx=8, pady=8)
  top_frame.pack(fill='x', padx=10, pady=8)

  tk.Label(top_frame, text='Pfad / Datei:').grid(row=0, column=0, sticky='w')
  entry = tk.Entry(top_frame, textvariable=path_var, width=82)
  entry.grid(row=0, column=1, columnspan=4, sticky='we', padx=(6, 6))

  def _browse_folder():
    p = filedialog.askdirectory(initialdir=path_var.get() or '.')
    if p:
      path_var.set(p)
      _scan_speedlines()

  def _browse_file():
    p = filedialog.askopenfilename(
      initialdir=(os.path.dirname(path_var.get()) if path_var.get() else '.'),
      filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )
    if p:
      path_var.set(p)
      _scan_speedlines()

  tk.Button(top_frame, text='Ordner...', command=_browse_folder, width=10).grid(row=1, column=1, sticky='w', pady=(6, 0))
  tk.Button(top_frame, text='Datei...', command=_browse_file, width=10).grid(row=1, column=2, sticky='w', pady=(6, 0))
  tk.Checkbutton(top_frame, text='Rekursiv suchen', variable=recursive_var).grid(row=1, column=3, sticky='w', pady=(6, 0))

  speed_frame = tk.LabelFrame(root, text='Speedline-Legende (anklicken = berücksichtigen)', padx=8, pady=8)
  speed_frame.pack(fill='both', expand=True, padx=10, pady=8)

  list_frame = tk.Frame(speed_frame)
  list_frame.pack(fill='both', expand=True)
  speed_list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, exportselection=False, height=14)
  speed_list.pack(side='left', fill='both', expand=True)
  speed_scroll = tk.Scrollbar(list_frame, orient='vertical', command=speed_list.yview)
  speed_scroll.pack(side='right', fill='y')
  speed_list.config(yscrollcommand=speed_scroll.set)

  def _scan_speedlines():
    nonlocal rpm_values
    files = _collect_csv_files(path_var.get().strip(), recursive_var.get())
    rpm_values = _scan_speedlines_from_files(files)
    speed_list.delete(0, tk.END)
    for rpm in rpm_values:
      speed_list.insert(tk.END, f'{rpm} rpm')
    if rpm_values:
      speed_list.select_set(0, tk.END)
      info_var.set(f'{len(files)} CSV-Dateien gefunden, {len(rpm_values)} Speedlines erkannt.')
    else:
      info_var.set(f'{len(files)} CSV-Dateien gefunden, keine Speedlines im Dateinamen erkannt.')

  def _select_all():
    speed_list.select_set(0, tk.END)

  def _select_none():
    speed_list.selection_clear(0, tk.END)

  btn_row = tk.Frame(speed_frame)
  btn_row.pack(fill='x', pady=(8, 0))
  tk.Button(btn_row, text='Speedlines scannen', command=_scan_speedlines, width=18).pack(side='left')
  tk.Button(btn_row, text='Alle', command=_select_all, width=8).pack(side='left', padx=(8, 0))
  tk.Button(btn_row, text='Keine', command=_select_none, width=8).pack(side='left', padx=(6, 0))
  tk.Label(btn_row, textvariable=info_var, anchor='w').pack(side='left', padx=(12, 0))

  out_frame = tk.LabelFrame(root, text='Zu erzeugende Bilder / Dokumente', padx=8, pady=8)
  out_frame.pack(fill='x', padx=10, pady=8)
  tk.Checkbutton(out_frame, text='Messwert-CSV (data_avg_*.csv)', variable=export_csv_var).grid(row=0, column=0, sticky='w')
  tk.Checkbutton(out_frame, text='Efficiency Map erzeugen', variable=create_eff_var).grid(row=1, column=0, sticky='w')
  tk.Checkbutton(out_frame, text='Motor Map erzeugen', variable=create_motor_var).grid(row=2, column=0, sticky='w')
  tk.Checkbutton(out_frame, text='Torque-Trennlinien Map erzeugen', variable=create_torque_var).grid(row=3, column=0, sticky='w')
  tk.Checkbutton(out_frame, text='PNG speichern', variable=export_png_var).grid(row=1, column=1, sticky='w', padx=(20, 0))
  tk.Checkbutton(out_frame, text='PDF speichern', variable=export_pdf_var).grid(row=2, column=1, sticky='w', padx=(20, 0))

  action_frame = tk.Frame(root)
  action_frame.pack(fill='x', padx=10, pady=(4, 10))

  def _cancel():
    result.clear()
    result['cancelled'] = True
    root.destroy()

  def _start():
    p = path_var.get().strip()
    if not p:
      messagebox.showerror('Fehler', 'Bitte einen Pfad oder eine Datei eingeben.')
      return
    files = _collect_csv_files(p, recursive_var.get())
    if not files:
      messagebox.showerror('Fehler', 'Keine passenden CSV-Dateien gefunden.')
      return

    selected_idx = list(speed_list.curselection())
    selected_rpms = []
    for idx in selected_idx:
      if (0 <= idx < len(rpm_values)):
        selected_rpms.append(int(rpm_values[idx]))

    result.clear()
    result.update({
      'cancelled': False,
      'inputfiles': p,
      'path_recursive': bool(recursive_var.get()),
      'selected_speedlines': selected_rpms,
      'export_csv': bool(export_csv_var.get()),
      'create_eff_map': bool(create_eff_var.get()),
      'create_motor_map': bool(create_motor_var.get()),
      'create_torque_map': bool(create_torque_var.get()),
      'export_png': bool(export_png_var.get()),
      'export_pdf': bool(export_pdf_var.get())
    })
    root.destroy()

  tk.Button(action_frame, text='Abbrechen', command=_cancel, width=12).pack(side='right')
  tk.Button(action_frame, text='Start', command=_start, width=12).pack(side='right', padx=(0, 8))

  _scan_speedlines()
  root.mainloop()
  return result
 

def smooth(y, window = 3):
  y = np.asarray(y, dtype = float)
  if (window is None) or (window <= 1) or (len(y) < int(window)):
    return np.array(y, dtype = float)
  w = int(window)
  kernel = np.ones(w, dtype = float) / float(w)
  left = w // 2
  right = w - 1 - left
  y_pad = np.pad(y, (left, right), mode = 'edge')
  return np.convolve(y_pad, kernel, mode = 'valid')


def plot_efficiency_map(
  data,
  savepath='efficiency_map.png',
  save_svg=False,
  highlight_global_max=False,
  save_png=True,
  save_pdf=True
):
  try:
    if data is None:
      print('  warning: no dataframe for efficiency map.')
      return

    if hasattr(data, 'columns'):
      columns = list(data.columns)
    elif isinstance(data, dict):
      columns = list(data.keys())
    else:
      print('  warning: efficiency map skipped, unsupported data container.')
      return

    required = ['m_dot_1_norm', 'p2_p1_ratio']
    missing = [c for c in required if c not in columns]
    if missing:
      print(f'  warning: efficiency map skipped, missing columns: {missing}')
      return

    speed_col = 'N_korr' if 'N_korr' in columns else ('N_mot' if 'N_mot' in columns else None)
    if speed_col is None:
      print("  warning: efficiency map skipped, missing columns: ['N_korr' or 'N_mot']")
      return

    p_mech_col = 'P_mech_mean' if 'P_mech_mean' in columns else ('P_mech' if 'P_mech' in columns else None)
    p_dc_col = 'P_DC_mean' if 'P_DC_mean' in columns else ('P_dc_mean' if 'P_dc_mean' in columns else None)
    if (p_mech_col is None) or (p_dc_col is None):
      print('  warning: efficiency map skipped, missing columns: P_mech_mean/P_mech or P_DC_mean/P_dc_mean')
      return

    if p_mech_col != 'P_mech_mean':
      print('  note: using P_mech as P_mech_mean (ROI-mean column fallback).')
    if p_dc_col != 'P_DC_mean':
      print('  note: using P_dc_mean as P_DC_mean (name fallback).')

    def _to_float_array(values):
      arr = []
      for v in values:
        sv = _safe_float(v)
        arr.append(np.nan if sv is None else sv)
      return np.asarray(arr, dtype=float)

    p_mech = _to_float_array(data[p_mech_col])
    p_dc = _to_float_array(data[p_dc_col])
    n_spd = _to_float_array(data[speed_col])
    m_dot = _to_float_array(data['m_dot_1_norm'])
    ratio = _to_float_array(data['p2_p1_ratio'])

    eta = p_mech / p_dc

    valid = np.isfinite(p_mech) & np.isfinite(p_dc) & np.isfinite(eta) & np.isfinite(m_dot) & np.isfinite(ratio) & np.isfinite(n_spd)
    valid &= (p_dc > 0) & (p_mech > 0)
    valid &= (eta > 0) & (eta <= 1.2)
    valid &= (m_dot > 0) & (ratio > 0)

    n_valid = int(np.count_nonzero(valid))
    if n_valid < 6:
      print(f'  warning: efficiency map skipped, fewer than 6 valid points ({n_valid}).')
      return

    x = m_dot[valid]
    y = ratio[valid]
    z = eta[valid]
    n = n_spd[valid]

    # Group speed lines with dynamic bin width (as in original script behavior).
    n_abs = np.abs(n)
    span_n = float(np.nanmax(n_abs) - np.nanmin(n_abs)) if len(n_abs) > 0 else 0.0
    if span_n >= 3000:
      n_step = 500.0
    elif span_n >= 1200:
      n_step = 200.0
    else:
      n_step = 100.0
    # Surface/eta smoothing grouping stays coarse as defined above.
    n_group_surface = np.round(n_abs / n_step) * n_step
    finite_n_surface = np.isfinite(n_group_surface)
    unique_groups_surface = np.sort(np.unique(n_group_surface[finite_n_surface])) if np.any(finite_n_surface) else np.array([])
    n_groups = len(unique_groups_surface)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = (0.06 * x_span) if x_span > 0 else max(0.03 * abs(x_min), 0.03)
    y_pad = (0.12 * y_span) if y_span > 0 else max(0.05 * abs(y_min), 0.05)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    scipy_ok = False
    griddata = None
    cKDTree = None
    UnivariateSpline = None
    try:
      from scipy.interpolate import griddata as _griddata
      from scipy.spatial import cKDTree as _cKDTree
      griddata = _griddata
      cKDTree = _cKDTree
      scipy_ok = True
    except Exception:
      scipy_ok = False

    # Visual smoothing of efficiency per speed group via polynomial of degree 3.
    z_plot = np.array(z, dtype=float)
    for rpm in unique_groups_surface:
      idx = np.where(n_group_surface == rpm)[0]
      if len(idx) < 4:
        continue
      xg = x[idx]
      zg = z[idx]
      xu = np.unique(xg)
      if len(xu) < 4:
        continue
      zu = np.array([np.mean(zg[xg == xv]) for xv in xu], dtype = float)
      try:
        coeff_z = np.polyfit(xu, zu, 3)
        z_fit = np.polyval(coeff_z, xg)
        z_plot[idx] = np.clip(z_fit, 0.0, 1.2)
      except Exception:
        pass

    levels_fill = np.arange(0.88, 0.97, 0.01)
    levels_line = levels_fill

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#f5f6f7')
    cmap_name = 'cividis'

    method_txt = 'matplotlib.triangulation (fallback)'
    cf = None
    cs = None

    if scipy_ok:
      try:
        nx, ny = 260, 260
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        Xi, Yi = np.meshgrid(xi, yi)

        Zi = griddata((x, y), z_plot, (Xi, Yi), method='linear')

        tree = cKDTree(np.column_stack((x, y)))
        qpts = np.column_stack((Xi.ravel(), Yi.ravel()))
        dist, _ = tree.query(qpts, k=1)
        dist = dist.reshape(Xi.shape)

        if len(x) > 1:
          dnn, _ = tree.query(np.column_stack((x, y)), k=2)
          nn_scale = float(np.nanmedian(dnn[:, 1]))
        else:
          nn_scale = 0.0
        diag = float(np.hypot(np.ptp(x), np.ptp(y)))
        d_thresh = max(2.2 * nn_scale, 0.025 * diag, 1e-9)

        base_mask = np.isnan(Zi) | (dist > d_thresh)
        Zi_masked = np.ma.array(Zi, mask=base_mask)

        cf = ax.contourf(Xi, Yi, Zi_masked, levels=levels_fill, cmap=cmap_name, extend='both')
        cs = ax.contour(Xi, Yi, Zi_masked, levels=levels_line, colors='black', linewidths=0.95, alpha=0.95)
        method_txt = 'scipy.griddata + KDTree mask'
      except Exception:
        scipy_ok = False

    if not scipy_ok:
      import matplotlib.tri as mtri
      tri = mtri.Triangulation(x, y)
      try:
        analyzer = mtri.TriAnalyzer(tri)
        tri.set_mask(analyzer.get_flat_tri_mask(min_circle_ratio=0.01))
      except Exception:
        pass
      cf = ax.tricontourf(tri, z_plot, levels=levels_fill, cmap=cmap_name, extend='both')
      cs = ax.tricontour(tri, z_plot, levels=levels_line, colors='black', linewidths=0.95, alpha=0.95)

    if cs is not None:
      # One label per contour level: pick midpoint of longest segment for each level.
      placed_labels = []
      try:
        all_segs = getattr(cs, 'allsegs', [])
        for li, lvl in enumerate(cs.levels):
          if (li >= len(all_segs)) or (len(all_segs[li]) == 0):
            continue
          segs = all_segs[li]
          seg_lens = []
          for seg in segs:
            if len(seg) < 2:
              seg_lens.append(0.0)
              continue
            dxy = np.diff(np.asarray(seg, dtype = float), axis = 0)
            seg_lens.append(float(np.sum(np.hypot(dxy[:, 0], dxy[:, 1]))))
          if len(seg_lens) == 0:
            continue
          best_idx = int(np.argmax(seg_lens))
          best_seg = np.asarray(segs[best_idx], dtype = float)
          if len(best_seg) == 0:
            continue
          mid_pt = best_seg[len(best_seg) // 2]
          txt = ax.clabel(
            cs,
            levels = [lvl],
            manual = [(float(mid_pt[0]), float(mid_pt[1]))],
            inline = True,
            fmt = '%.2f',
            fontsize = 9
          )
          placed_labels.extend(txt)
      except Exception:
        placed_labels = ax.clabel(cs, inline = True, fmt = '%.2f', fontsize = 9)

      for t in placed_labels:
        t.set_bbox(dict(fc='white', ec='none', alpha=0.6, pad=0.1))

    ax.scatter(
      x, y, c=z_plot, cmap=cmap_name,
      vmin=float(np.min(levels_fill)), vmax=float(np.max(levels_fill)),
      s=16, edgecolors='#2f2f2f', linewidths=0.35, alpha=0.9, zorder=5
    )

    # Line grouping is separate/finer to avoid merging close speed levels (e.g. 11500 vs 11700).
    n_group_lines = None
    if 'FILE' in columns:
      try:
        file_vals = np.asarray(data['FILE'], dtype = object)
        if len(file_vals) == len(valid):
          file_vals = file_vals[valid]
          file_rpm = []
          for fv in file_vals:
            m_rpm = re.search(r'(\d+)\s*rpm', str(fv), flags = re.IGNORECASE)
            file_rpm.append(float(m_rpm.group(1)) if m_rpm else np.nan)
          file_rpm = np.asarray(file_rpm, dtype = float)
          if np.count_nonzero(np.isfinite(file_rpm)) >= 2:
            n_group_lines = file_rpm
      except Exception:
        n_group_lines = None

    if n_group_lines is None:
      line_step = 10.0
      n_group_lines = np.round(n_abs / line_step) * line_step

    finite_n_lines = np.isfinite(n_group_lines)
    unique_groups_lines = np.sort(np.unique(n_group_lines[finite_n_lines])) if np.any(finite_n_lines) else np.array([])
    n_groups_lines = len(unique_groups_lines)
    if n_groups_lines <= 20:
      line_colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(1, n_groups_lines)))
    else:
      # For many speed lines: generate many unique hues and darken slightly.
      line_colors = plt.cm.hsv(np.linspace(0.02, 0.98, max(1, n_groups_lines), endpoint=False))
      line_colors[:, :3] = 0.85 * line_colors[:, :3]

    handles, labels = [], []
    for gi, rpm in enumerate(unique_groups_lines):
      idx = np.where(n_group_lines == rpm)[0]
      if len(idx) < 2:
        continue
      order = np.argsort(x[idx])
      x_line = x[idx][order]
      y_line = y[idx][order]

      color = line_colors[gi % len(line_colors)]

      x_plot = x_line
      y_plot = y_line
      if len(x_line) >= 4:
        xu = np.unique(x_line)
        if len(xu) >= 4:
          yu = np.array([np.mean(y_line[x_line == xv]) for xv in xu], dtype = float)
          try:
            coeff = np.polyfit(xu, yu, 3)
            x_plot = np.linspace(float(np.min(xu)), float(np.max(xu)), 120)
            y_plot = np.polyval(coeff, x_plot)
          except Exception:
            x_plot = x_line
            y_plot = y_line

      (h,) = ax.plot(
        x_plot, y_plot, '-', color = color, linewidth = 1.9,
        alpha = 0.98, zorder = 8
      )
      handles.append(h)
      labels.append(rf"$N_\mathrm{{korr}} = {int(round(rpm))}\ \mathrm{{rpm}}$")

    if handles:
      ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

    if highlight_global_max and np.any(np.isfinite(z_plot)):
      gidx = int(np.nanargmax(z_plot))
      gx, gy, gz = float(x[gidx]), float(y[gidx]), float(z_plot[gidx])
      ax.scatter([gx], [gy], marker='*', s=95, color='crimson', edgecolors='black', linewidths=0.5, zorder=8)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    cbar = fig.colorbar(cf, ax=ax, ticks=levels_fill[::2])
    cbar.set_label('η = P_mech / P_DC [-]')

    ax.set_xlabel("Normierter Massenstrom $\\dot{m}_1$ [–]")
    ax.set_ylabel("Druckverhältnis $p_2/p_1$ [–]")
    ax.set_title('DC-to-Shaft Efficiency Map (Messdaten)')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.15)

    savepath = str(savepath)
    out_dir = os.path.dirname(savepath) or '.'
    os.makedirs(out_dir, exist_ok=True)

    png_file = savepath
    pdf_file = os.path.splitext(png_file)[0] + '.pdf'

    fig.tight_layout()
    saved = []
    if save_png:
      fig.savefig(png_file, dpi=300)
      saved.append(png_file)
    if save_pdf:
      fig.savefig(pdf_file)
      saved.append(pdf_file)
    if save_svg:
      svg_file = os.path.splitext(png_file)[0] + '.svg'
      fig.savefig(svg_file)
      saved.append(svg_file)

    plt.close(fig)
    if saved:
      print(f"  efficiency map saved: {' / '.join(saved)} ({method_txt})")
    else:
      print('  note: efficiency map generated, but all export formats were disabled.')

  except Exception as e:
    print(f'  warning: efficiency map failed: {e}')
    try:
      plt.close('all')
    except Exception:
      pass


def plot_torque_contour_map(
  data,
  savepath='torque_contour_map.png',
  save_png=True,
  save_pdf=True
):
  try:
    if data is None:
      print('  warning: torque contour map skipped, no dataframe.')
      return

    if hasattr(data, 'columns'):
      columns = list(data.columns)
    elif isinstance(data, dict):
      columns = list(data.keys())
    else:
      print('  warning: torque contour map skipped, unsupported data container.')
      return

    required = ['m_dot_1_norm', 'p2_p1_ratio', 'M_mot']
    missing = [c for c in required if c not in columns]
    if missing:
      print(f'  warning: torque contour map skipped, missing columns: {missing}')
      return

    def _to_float_array(values):
      arr = []
      for v in values:
        sv = _safe_float(v)
        arr.append(np.nan if sv is None else sv)
      return np.asarray(arr, dtype = float)

    x_all = _to_float_array(data['m_dot_1_norm'])
    y_all = _to_float_array(data['p2_p1_ratio'])
    m_all = np.abs(_to_float_array(data['M_mot']))

    valid = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(m_all)
    valid &= (x_all > 0.0) & (y_all > 0.0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 6:
      print(f'  warning: torque contour map skipped, fewer than 6 valid points ({n_valid}).')
      return

    x = x_all[valid]
    y = y_all[valid]
    m = m_all[valid]

    speed_col = 'N_korr' if 'N_korr' in columns else ('N_mot' if 'N_mot' in columns else None)
    n_group_lines = None
    if 'FILE' in columns:
      try:
        file_vals = np.asarray(data['FILE'], dtype = object)
        if len(file_vals) == len(valid):
          file_vals = file_vals[valid]
          file_rpm = []
          for fv in file_vals:
            m_rpm = re.search(r'(\d+)\s*rpm', str(fv), flags = re.IGNORECASE)
            file_rpm.append(float(m_rpm.group(1)) if m_rpm else np.nan)
          file_rpm = np.asarray(file_rpm, dtype = float)
          if np.count_nonzero(np.isfinite(file_rpm)) >= 2:
            n_group_lines = np.abs(file_rpm)
      except Exception:
        n_group_lines = None

    if (n_group_lines is None) and (speed_col is not None):
      try:
        n_all = np.abs(_to_float_array(data[speed_col]))
        n_use = n_all[valid]
        if len(n_use) > 0:
          line_step = 20.0
          n_group_lines = np.round(n_use / line_step) * line_step
      except Exception:
        n_group_lines = None

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = (0.06 * x_span) if x_span > 0 else max(0.03 * abs(x_min), 0.03)
    y_pad = (0.12 * y_span) if y_span > 0 else max(0.05 * abs(y_min), 0.05)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    m_min = float(np.nanmin(m))
    m_max = float(np.nanmax(m))
    if not np.isfinite(m_min) or not np.isfinite(m_max):
      print('  warning: torque contour map skipped, invalid M_mot values.')
      return
    if m_max <= m_min:
      m_max = m_min + 0.1
    rng = m_max - m_min
    target_step = max(rng / 8.0, 1e-6)
    cand_steps = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype = float)
    step = float(cand_steps[np.argmin(np.abs(cand_steps - target_step))])
    l0 = np.floor(m_min / step) * step
    l1 = np.ceil(m_max / step) * step
    levels = np.arange(l0, l1 + 0.5 * step, step)
    if len(levels) < 2:
      levels = np.array([m_min, m_max], dtype = float)

    fig, ax = plt.subplots(figsize = (9, 7))
    ax.set_facecolor('#f5f6f7')
    from matplotlib.colors import LinearSegmentedColormap
    cmap_bg = LinearSegmentedColormap.from_list(
      'darkblue_red',
      ['#001247', '#0b3b8c', '#2f7fb9', '#d65f4a', '#b00020'],
      N = 256
    )

    method_txt = 'matplotlib.triangulation (fallback)'
    cf = None
    cs = None

    scipy_ok = False
    try:
      from scipy.interpolate import griddata as _griddata
      from scipy.spatial import cKDTree as _cKDTree
      griddata = _griddata
      cKDTree = _cKDTree
      scipy_ok = True
    except Exception:
      scipy_ok = False

    if scipy_ok:
      try:
        nx, ny = 240, 240
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        Xi, Yi = np.meshgrid(xi, yi)
        Mi = griddata((x, y), m, (Xi, Yi), method = 'linear')

        tree = cKDTree(np.column_stack((x, y)))
        qpts = np.column_stack((Xi.ravel(), Yi.ravel()))
        dist, _ = tree.query(qpts, k = 1)
        dist = dist.reshape(Xi.shape)

        if len(x) > 1:
          dnn, _ = tree.query(np.column_stack((x, y)), k = 2)
          nn_scale = float(np.nanmedian(dnn[:, 1]))
        else:
          nn_scale = 0.0
        diag = float(np.hypot(np.ptp(x), np.ptp(y)))
        d_thresh = max(2.2 * nn_scale, 0.025 * diag, 1e-9)

        mask = np.isnan(Mi) | (dist > d_thresh)
        Mi_masked = np.ma.array(Mi, mask = mask)
        cf = ax.contourf(
          Xi, Yi, Mi_masked,
          levels = levels,
          cmap = cmap_bg,
          extend = 'both',
          alpha = 0.92,
          zorder = 1
        )
        cs = ax.contour(
          Xi, Yi, Mi_masked,
          levels = levels,
          colors = 'black',
          linewidths = 1.15,
          alpha = 0.95,
          zorder = 4
        )
        method_txt = 'scipy.griddata + KDTree mask'
      except Exception:
        scipy_ok = False

    if not scipy_ok:
      import matplotlib.tri as mtri
      tri = mtri.Triangulation(x, y)
      try:
        analyzer = mtri.TriAnalyzer(tri)
        tri.set_mask(analyzer.get_flat_tri_mask(min_circle_ratio = 0.01))
      except Exception:
        pass
      cf = ax.tricontourf(
        tri, m,
        levels = levels,
        cmap = cmap_bg,
        extend = 'both',
        alpha = 0.92,
        zorder = 1
      )
      cs = ax.tricontour(
        tri, m,
        levels = levels,
        colors = 'black',
        linewidths = 1.15,
        alpha = 0.95,
        zorder = 4
      )

    if cs is not None:
      placed_labels = []
      try:
        all_segs = getattr(cs, 'allsegs', [])
        for li, lvl in enumerate(cs.levels):
          if (li >= len(all_segs)) or (len(all_segs[li]) == 0):
            continue
          segs = all_segs[li]
          seg_lens = []
          for seg in segs:
            if len(seg) < 2:
              seg_lens.append(0.0)
              continue
            dxy = np.diff(np.asarray(seg, dtype = float), axis = 0)
            seg_lens.append(float(np.sum(np.hypot(dxy[:, 0], dxy[:, 1]))))
          if len(seg_lens) == 0:
            continue
          best_seg = np.asarray(segs[int(np.argmax(seg_lens))], dtype = float)
          if len(best_seg) == 0:
            continue
          mid_pt = best_seg[len(best_seg) // 2]
          txt = ax.clabel(
            cs,
            levels = [lvl],
            manual = [(float(mid_pt[0]), float(mid_pt[1]))],
            inline = True,
            fmt = '%.2f',
            fontsize = 9
          )
          placed_labels.extend(txt)
      except Exception:
        placed_labels = ax.clabel(cs, inline = True, fmt = '%.2f', fontsize = 9)

      for t in placed_labels:
        t.set_bbox(dict(fc = 'white', ec = 'none', alpha = 0.65, pad = 0.1))

    # Overlay speedlines in compressor coordinates.
    if n_group_lines is not None:
      finite_lines = np.isfinite(n_group_lines)
      unique_groups = np.sort(np.unique(n_group_lines[finite_lines])) if np.any(finite_lines) else np.array([])
      if len(unique_groups) > 0:
        line_colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(1, len(unique_groups))))
        h_list = []
        l_list = []
        for gi, rpm in enumerate(unique_groups):
          idx = np.where(n_group_lines == rpm)[0]
          if len(idx) < 2:
            continue
          order = np.argsort(x[idx])
          x_line = x[idx][order]
          y_line = y[idx][order]
          h, = ax.plot(
            x_line, y_line, '-',
            color = line_colors[gi % len(line_colors)],
            linewidth = 1.8,
            alpha = 0.98,
            zorder = 6
          )
          h_list.append(h)
          l_list.append(rf"$N = {int(round(rpm))}\ \mathrm{{rpm}}$")
        if h_list and (len(h_list) <= 12):
          ax.legend(h_list, l_list, loc = 'best', fontsize = 8, framealpha = 0.9, title = 'Speedlines')

    sc = ax.scatter(
      x, y, c = m, cmap = cmap_bg,
      vmin = float(levels[0]),
      vmax = float(levels[-1]),
      s = 22, edgecolors = '#f2f2f2', linewidths = 0.45, alpha = 0.95, zorder = 5
    )
    if cf is not None:
      cbar = fig.colorbar(cf, ax = ax)
    else:
      cbar = fig.colorbar(sc, ax = ax)
    cbar.set_label('|M_mot| [Nm]')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Normierter Massenstrom $\\dot{m}_1$ [–]")
    ax.set_ylabel("Druckverhältnis $p_2/p_1$ [–]")
    ax.set_title('Torque Contour Map (|M_mot|) on Compressor Coordinates')
    ax.grid(True, linestyle = '--', linewidth = 0.4, alpha = 0.15)

    savepath = str(savepath)
    out_dir = os.path.dirname(savepath) or '.'
    os.makedirs(out_dir, exist_ok = True)
    png_file = savepath
    pdf_file = os.path.splitext(png_file)[0] + '.pdf'

    fig.tight_layout()
    saved = []
    if save_png:
      fig.savefig(png_file, dpi = 300)
      saved.append(png_file)
    if save_pdf:
      fig.savefig(pdf_file)
      saved.append(pdf_file)
    plt.close(fig)

    if saved:
      print(f"  torque contour map saved: {' / '.join(saved)} ({method_txt})")
    else:
      print('  note: torque contour map generated, but all export formats were disabled.')

  except Exception as e:
    print(f'  warning: torque contour map failed: {e}')
    try:
      plt.close('all')
    except Exception:
      pass


def _find_motor_eff_map_columns(df):
  if hasattr(df, 'columns'):
    columns = list(df.columns)
  elif isinstance(df, dict):
    columns = list(df.keys())
  else:
    columns = []

  def _first_match(preferred, candidates):
    if preferred and (preferred in columns):
      return preferred
    lower_map = {str(c).lower(): c for c in columns}
    for c in candidates:
      if c in columns:
        return c
      lc = str(c).lower()
      if lc in lower_map:
        return lower_map[lc]
    return None

  pdc_col = _first_match('P_DC', ['P_DC_mean', 'P_dc', 'P_dc_mean', 'Pdc'])
  pmech_col = _first_match('P_mech', ['P_mech_mean', 'Pmech', 'P_shaft', 'P_out', 'P_motor_mech'])
  speed_col = _first_match('N_korr', ['N', 'n', 'speed', 'rpm'])
  torque_col = _first_match(
    'M_mot',
    ['M_mot_mean', 'm_mot', 'm_mot_mean', 'M', 'M_mech', 'Torque', 'torque', 'Tq', 'T_mech', 'M_est', 'torque_est', 'Torque_est', 'M_motor', 'T_motor', 'M_Nm', 'torque_Nm']
  )

  return pdc_col, pmech_col, speed_col, torque_col


def plot_motor_efficiency_map_torque_speed(
  df,
  out_dir = ".",
  fname = "motor_eff_map_torque_speed_like_ref",
  save_png = True,
  save_pdf = True
):
  try:
    def _get_col(d, col):
      if isinstance(d, dict):
        return d.get(col, [])
      return d[col]

    def _to_float_array(values):
      arr = []
      for v in values:
        sv = _safe_float(v)
        arr.append(np.nan if sv is None else sv)
      return np.asarray(arr, dtype = float)

    pdc_col, pmech_col, speed_col, torque_col = _find_motor_eff_map_columns(df)
    columns = list(df.keys()) if isinstance(df, dict) else (list(df.columns) if hasattr(df, 'columns') else [])
    file_col = None
    for c in ['FILE', 'file', 'File', 'filename', 'file_name']:
      if c in columns:
        file_col = c
        break

    if torque_col is None:
      print('  warning: motor_eff_map_torque_speed skipped, no torque column found.')
      return
    if pdc_col is None:
      print('  warning: motor_eff_map_torque_speed skipped, no P_DC column found.')
      return
    if pmech_col is None:
      print('  warning: motor_eff_map_torque_speed skipped, no P_mech column found.')
      return
    if speed_col is None:
      print('  warning: motor_eff_map_torque_speed skipped, no speed column found.')
      return

    pdc = _to_float_array(_get_col(df, pdc_col))
    pmech = _to_float_array(_get_col(df, pmech_col))
    n = _to_float_array(_get_col(df, speed_col))
    m_raw = _to_float_array(_get_col(df, torque_col))
    m = np.abs(m_raw)
    file_vals = None
    if file_col is not None:
      try:
        file_tmp = np.asarray(_get_col(df, file_col), dtype = object)
        if len(file_tmp) == len(pdc):
          file_vals = file_tmp
      except Exception:
        file_vals = None
    eta = pmech / pdc

    finite_mask = np.isfinite(pdc) & np.isfinite(pmech) & np.isfinite(n) & np.isfinite(m) & np.isfinite(eta)
    valid = finite_mask.copy()
    valid &= (pdc > 0.0) & (pmech > 0.0)
    valid &= (eta > 0.0) & (eta < 1.2)
    valid &= (n != 0.0)
    valid &= (m >= 0.0)

    # Use absolute speed for a classic motor-map view and avoid split clouds by rotation direction.
    x = np.abs(n[valid])
    y = m[valid]
    z = eta[valid]
    file_v = file_vals[valid] if file_vals is not None else None

    blende_ids = None
    if file_v is not None:
      def _extract_blende_id(val):
        m_bl = re.search(r'blende[^0-9+-]*([+-]?\d+)', str(val), flags = re.IGNORECASE)
        return int(m_bl.group(1)) if m_bl else None

      blende_ids = np.asarray([_extract_blende_id(v) for v in file_v], dtype = object)
      has_b0 = np.any(blende_ids == 0)
      has_b185 = np.any(blende_ids == 185)
      if has_b0 and has_b185:
        keep = (blende_ids == 0) | (blende_ids == 185)
        x = x[keep]
        y = y[keep]
        z = z[keep]
        file_v = file_v[keep]
        blende_ids = blende_ids[keep]
      else:
        blende_ids = None

    if len(x) == 0:
      print('  warning: motor_eff_map_torque_speed skipped, no valid points after filtering.')
      print(
        '  debug: '
        f'finite={int(np.count_nonzero(finite_mask))}, '
        f'Pdc>0={int(np.count_nonzero(finite_mask & (pdc > 0.0)))}, '
        f'Pmech>0={int(np.count_nonzero(finite_mask & (pmech > 0.0)))}, '
        f'eta(0..1.2)={int(np.count_nonzero(finite_mask & (eta > 0.0) & (eta < 1.2)))}, '
        f'N!=0={int(np.count_nonzero(finite_mask & (n != 0.0)))}, '
        f'M>=0={int(np.count_nonzero(finite_mask & (m >= 0.0)))}'
      )
      return

    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    l0 = np.floor(zmin * 100.0) / 100.0
    l1 = np.ceil(zmax * 100.0) / 100.0
    if l1 <= l0:
      l1 = l0 + 0.01
    levels = np.arange(l0, l1 + 0.001, 0.01)
    if len(levels) < 2:
      levels = np.array([l0, l1], dtype = float)
    contour_levels = levels[::2] if len(levels) > 2 else levels
    if (len(contour_levels) < 3) and ((zmax - zmin) > 1e-4):
      contour_levels = np.linspace(zmin, zmax, 4)

    fig, ax = plt.subplots(figsize = (9, 7))
    ax.set_facecolor('#f5f6f7')
    cf = None
    cs = None

    # Hybrid strategy for sparse speed lines: contour only if bin occupancy is sufficient.
    enough_for_surface = len(x) >= 30
    drew_surface = False
    if enough_for_surface:
      x_min, x_max = float(np.min(x)), float(np.max(x))
      y_min, y_max = float(np.min(y)), float(np.max(y))
      if x_max <= x_min:
        x_max = x_min + 1.0
      if y_max <= y_min:
        y_max = y_min + 1.0

      speed_step_hint = 50.0
      x_group_probe = np.round(x / speed_step_hint) * speed_step_hint
      n_speed_groups = max(1, len(np.unique(x_group_probe)))
      n_bins_speed = int(np.clip(3 * n_speed_groups, 14, 42))
      n_bins_torque = int(np.clip(max(22, np.sqrt(len(x)) * 3.2), 22, 40))

      x_edges = np.linspace(x_min, x_max, n_bins_speed + 1)
      y_edges = np.linspace(y_min, y_max, n_bins_torque + 1)

      ix = np.digitize(x, x_edges) - 1
      iy = np.digitize(y, y_edges) - 1
      in_range = (ix >= 0) & (ix < n_bins_speed) & (iy >= 0) & (iy < n_bins_torque)
      ix = ix[in_range]
      iy = iy[in_range]
      zv = z[in_range]

      z_grid = np.full((n_bins_torque, n_bins_speed), np.nan, dtype = float)
      bins = {}
      for k in range(len(zv)):
        key = (iy[k], ix[k])
        bins.setdefault(key, []).append(float(zv[k]))
      for (r, c), vals in bins.items():
        z_grid[r, c] = float(np.median(vals))

      valid_mask = np.isfinite(z_grid)
      occupied = int(np.count_nonzero(valid_mask))
      needed = max(40, int(0.14 * n_bins_speed * n_bins_torque))
      if occupied >= needed:
        z_plot = z_grid
        try:
          from scipy.ndimage import gaussian_filter
          sigma = 0.7
          num = gaussian_filter(np.where(valid_mask, z_grid, 0.0), sigma = sigma, mode = 'constant', cval = 0.0)
          den = gaussian_filter(valid_mask.astype(float), sigma = sigma, mode = 'constant', cval = 0.0)
          z_smooth = np.where(den > 1e-8, num / np.maximum(den, 1e-8), np.nan)
          z_plot = np.where(valid_mask, z_smooth, np.nan)
        except Exception:
          z_plot = z_grid

        z_masked = np.ma.masked_invalid(z_plot)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        Xc, Yc = np.meshgrid(xc, yc)

        cf = ax.contourf(Xc, Yc, z_masked, levels = levels, cmap = 'cividis', extend = 'both', zorder = 1)
        cs = ax.contour(Xc, Yc, z_masked, levels = contour_levels, colors = 'black', linewidths = 0.95, alpha = 0.95, zorder = 3)
        ax.clabel(cs, inline = True, fontsize = 8, fmt = '%.2f')
        drew_surface = True
      else:
        print(f'  note: contour surface skipped (occupied bins {occupied}/{n_bins_speed * n_bins_torque}); using line+scatter view.')
    else:
      print(f'  warning: motor_eff_map_torque_speed sparse data ({len(x)} points), creating line+scatter view.')

    # Ensure colored background exists even if binning is too sparse.
    if (not drew_surface) and (len(x) >= 6):
      try:
        import matplotlib.tri as mtri
        tri = mtri.Triangulation(x, y)
        try:
          analyzer = mtri.TriAnalyzer(tri)
          tri.set_mask(analyzer.get_flat_tri_mask(min_circle_ratio = 0.01))
        except Exception:
          pass
        cf = ax.tricontourf(tri, z, levels = levels, cmap = 'cividis', extend = 'both', zorder = 1, alpha = 0.9)
        cs = ax.tricontour(tri, z, levels = contour_levels, colors = 'black', linewidths = 0.95, alpha = 0.95, zorder = 3)
        if cs is not None:
          ax.clabel(cs, inline = True, fontsize = 8, fmt = '%.2f')
        drew_surface = True
        print('  note: using triangulation background for efficiency coloring.')
      except Exception as e_bg:
        print(f'  note: background fill fallback failed ({e_bg}); scatter coloring only.')

    # Draw only Blende 0 and 185 as boundary lines and connect top/bottom speeds to form a closed field.
    used_blende_lines = False
    if blende_ids is not None:
      idx0 = np.where(blende_ids == 0)[0]
      idx185 = np.where(blende_ids == 185)[0]
      if (len(idx0) >= 2) and (len(idx185) >= 2):
        used_blende_lines = True
        o0 = np.argsort(x[idx0])
        o185 = np.argsort(x[idx185])
        x0, y0 = x[idx0][o0], y[idx0][o0]
        x185, y185 = x[idx185][o185], y[idx185][o185]

        h0, = ax.plot(x0, y0, '-', color = '#d7191c', linewidth = 1.9, alpha = 0.98, zorder = 8, label = 'Blende 0')
        h185, = ax.plot(x185, y185, '-', color = '#2c7bb6', linewidth = 1.9, alpha = 0.98, zorder = 8, label = 'Blende 185')

        # Connect the lowest-speed and highest-speed endpoints between both Blenden.
        ax.plot([x0[0], x185[0]], [y0[0], y185[0]], '-', color = '#404040', linewidth = 1.45, alpha = 0.95, zorder = 8)
        ax.plot([x0[-1], x185[-1]], [y0[-1], y185[-1]], '-', color = '#404040', linewidth = 1.45, alpha = 0.95, zorder = 8)

        # Subtle closed-field fill between both boundary lines.
        x_poly = np.concatenate([x0, x185[::-1]])
        y_poly = np.concatenate([y0, y185[::-1]])
        ax.fill(x_poly, y_poly, color = '#9e9e9e', alpha = 0.12, zorder = 2)

        ax.legend([h0, h185], ['Blende 0', 'Blende 185'], loc = 'best', fontsize = 8, framealpha = 0.9, title = 'Blende')

    if not used_blende_lines:
      # Fallback: connect by speed groups when Blende information is unavailable.
      speed_step = 50.0
      n_group = np.round(x / speed_step) * speed_step
      unique_groups = np.sort(np.unique(n_group[np.isfinite(n_group)]))
      for g in unique_groups:
        idx = np.where(n_group == g)[0]
        if len(idx) < 3:
          continue
        order = np.argsort(y[idx])
        x_line = x[idx][order]
        y_line = y[idx][order]
        ax.plot(x_line, y_line, '-', color = '#1f1f1f', linewidth = 0.7, alpha = 0.35, zorder = 4)

    sc = ax.scatter(
      x,
      y,
      c = z,
      cmap = 'cividis',
      vmin = float(levels[0]),
      vmax = float(levels[-1]),
      s = 14,
      edgecolors = '#202020',
      linewidths = 0.25,
      alpha = 0.95,
      zorder = 6
    )

    if drew_surface and (cf is not None):
      cbar = fig.colorbar(cf, ax = ax, ticks = contour_levels)
    else:
      cbar = fig.colorbar(sc, ax = ax, ticks = contour_levels)

    cbar.set_label('η = P_mech / P_DC [-]')
    ax.set_title('Motor Efficiency Map (η) - Torque vs Speed')
    ax.set_xlabel('Speed (rpm)')
    ax.set_ylabel('Torque (Nm)')
    ax.grid(True, linestyle = '--', linewidth = 0.4, alpha = 0.2)

    os.makedirs(out_dir, exist_ok = True)
    png_path = os.path.join(out_dir, f'{fname}.png')
    pdf_path = os.path.join(out_dir, f'{fname}.pdf')
    fig.tight_layout()
    saved = []
    if save_png:
      fig.savefig(png_path, dpi = 250)
      saved.append(png_path)
    if save_pdf:
      fig.savefig(pdf_path)
      saved.append(pdf_path)
    plt.close(fig)
    if saved:
      print(f"  motor efficiency map saved: {' / '.join(saved)}")
    else:
      print('  note: motor efficiency map generated, but all export formats were disabled.')

  except Exception as e:
    print(f'  warning: motor_eff_map_torque_speed failed: {e}')
    try:
      plt.close('all')
    except Exception:
      pass

def main():
  global path_recursive
  try:
    export_csv = True
    create_eff_map = True
    create_motor_map = True
    create_torque_map = True
    export_png = True
    export_pdf = True
    selected_speedlines = None

    gui_cfg = _startup_gui(default_path = default_data_path, recursive_default = path_recursive)
    if isinstance(gui_cfg, dict) and (not gui_cfg.get('cancelled', False)):
      inputfiles = gui_cfg.get('inputfiles', default_data_path)
      path_recursive = bool(gui_cfg.get('path_recursive', path_recursive))
      selected_speedlines = gui_cfg.get('selected_speedlines', None)
      export_csv = bool(gui_cfg.get('export_csv', True))
      create_eff_map = bool(gui_cfg.get('create_eff_map', True))
      create_motor_map = bool(gui_cfg.get('create_motor_map', True))
      create_torque_map = bool(gui_cfg.get('create_torque_map', True))
      export_png = bool(gui_cfg.get('export_png', True))
      export_pdf = bool(gui_cfg.get('export_pdf', True))
    elif isinstance(gui_cfg, dict) and gui_cfg.get('cancelled', False):
      print('Start abgebrochen.')
      return
    else:
      # Console fallback
      inputfiles = input(f'Input data file or path (to process all .csv files): [{default_data_path}] ')
      if not inputfiles:
        inputfiles = default_data_path
      if os.path.isdir(inputfiles):
        if path_recursive:
          tmp = input('Search path recursively? [Y/n] ')
          if (tmp == 'n') or (tmp == 'N'):
            path_recursive = False
        else:
          tmp = input('Search path recursively? [y/N] ')
          if (tmp == 'y') or (tmp == 'Y'):
            path_recursive = True

    data_files = _collect_csv_files(inputfiles, recursive = path_recursive)
    if not data_files:
      print('Error: data file not found.')
      sys.exit()

    if os.path.isdir(inputfiles):
      data_file_path = inputfiles
    else:
      data_file_path = os.path.dirname(inputfiles)

    if isinstance(selected_speedlines, (list, tuple)) and (len(selected_speedlines) > 0):
      sel_set = set(int(abs(v)) for v in selected_speedlines)
      before_n = len(data_files)
      filtered = []
      for fp in data_files:
        rpm = _extract_rpm(fp)
        if rpm is None:
          continue
        if int(abs(rpm)) in sel_set:
          filtered.append(fp)
      data_files = filtered
      print(f'  note: speedline filter active ({len(sel_set)} selected) -> {len(data_files)}/{before_n} files kept.')
      if not data_files:
        print('  warning: no files left after speedline filter.')
        return

    # Read and process all data_files
    for datafile_n in range(len(data_files)):
      # Init variables
      data_tic = []
      data_value = []
      data_unit = []
      data_name = []
      data_tic_col = -1
      data_value_col = -1
      data_unit_col = -1
      data_name_col = -1
      step_data_tic = []
      step_data = []
      step_list = []

      print(f'Process file {datafile_n + 1}/{len(data_files)} : {data_files[datafile_n]}')
      data_file = os.path.basename(data_files[datafile_n])
      data_descriptor = os.path.splitext(data_file)[0]
      file_rows_start = len(avg_data)
      try:
        with open(data_files[datafile_n], mode = 'rt', encoding = 'utf-8') as f:
          filereader = csv.reader(f, delimiter=';')
          for row in filereader:
            # Ignore empty lines and lines with comments
            if row and (row[0][0] != '#'):
              # Get column designation from first row
              if (data_name_col == -1):
                if ('PARAM' not in row):
                  print(' Error: first line misformatted')
                  sys.exit()
                else:
                  data_tic_col = row.index('TIME_TIC')
                  data_value_col = row.index('VALUE')
                  data_unit_col = row.index('UNIT')
                  data_name_col = row.index('PARAM')
              # Process data rows
              else:
                # Add new parameters to array
                if (row[data_name_col] not in data_name):
                  data_tic.append([])
                  data_value.append([])
                  data_unit.append(row[data_unit_col])
                  data_name.append(row[data_name_col])
                # Add timestamp and data
                this_param_idx = data_name.index(row[data_name_col])
                data_tic[this_param_idx].append(float(row[data_tic_col]))
                data_value[this_param_idx].append(float(row[data_value_col]))

        add_empty = True
        if (stepfind_param in data_name):
          # Copy timestamps and data into working variables, for easier reading
          step_data_tic = data_tic[data_name.index(stepfind_param)]
          step_data = data_value[data_name.index(stepfind_param)]

          ## Main function to extract steps from datafile
          # Input: data timestamp list, data list, fixed width in seconds of intervals to find, limit value for (standard) deviation to distinguish step_list, threshold value
          # Output: steps in list of lists [[start_idx, end_idx, mean, std], ...]
          step_list = tfdaq_stepfind.find_steps(step_data_tic, step_data, stepfind_interval, stepfind_std_lim, stepfind_threshold)
          
          # Delete steps of similar value
          step_list = tfdaq_stepfind.del_duplicate_steps(step_list, stepfind_std_lim)

          # If no steps are found, add empty line in output file
          if (len(step_list) > 0):
            # Average all selected parameters
            for step_n in range(len(step_list)):
              roi_start = step_data_tic[step_list[step_n][0]]
              roi_stop = step_data_tic[step_list[step_n][1]]
              if ((roi_start >= 0) and (roi_stop >= 0) and (roi_stop > roi_start)):
                # Add step to output list
                avg_data.append(['' for n in range(len(avg_data_columns))])
                avg_data[-1][avg_data_columns.index('FILE')] = os.path.basename(data_files[datafile_n])
                avg_data[-1][avg_data_columns.index('ROI_START')] = roi_start
                avg_data[-1][avg_data_columns.index('ROI_STOP')] = roi_stop
                # Calculate mean value of selected data in ROI
                for avg_param in avg_params:
                  if (avg_param in data_name):
                    # Find start and end data index
                    roi_start_idx = bisect.bisect_right(data_tic[data_name.index(avg_param)], roi_start)
                    roi_stop_idx = bisect.bisect_left(data_tic[data_name.index(avg_param)], roi_stop)
                    # Save this value
                    if roi_stop_idx > roi_start_idx:
                      add_empty = False
                      vals = np.asarray(data_value[data_name.index(avg_param)][roi_start_idx:roi_stop_idx])
                      avg_data[-1][avg_data_columns.index(avg_param)] = round(np.mean(vals), 4)
                      # SEM uses sample standard deviation when n > 1
                      if len(vals) > 1:
                        sem = np.std(vals, ddof = 1) / np.sqrt(len(vals))
                        avg_data[-1][avg_data_columns.index(f'{avg_param}_SEM')] = round(sem, 4)
                        # Add/overwrite unit to global unit list
                        avg_params_units[avg_params.index(avg_param)] = data_unit[data_name.index(avg_param)]
                        avg_params_sem_units[avg_params.index(avg_param)] = data_unit[data_name.index(avg_param)]
                # Force p_diff_2 = p_diff_1 for all further calculations
                if ('p_diff_1' in avg_params) and ('p_diff_2' in avg_params):
                  p1_idx = avg_data_columns.index('p_diff_1')
                  p2_idx = avg_data_columns.index('p_diff_2')
                  avg_data[-1][p2_idx] = avg_data[-1][p1_idx]
                # p_sta_1 = p_diff_4_mean - p_diff_3_mean (additional output column)
                if ('p_diff_3' in avg_params) and ('p_diff_4' in avg_params):
                  p3_idx = avg_data_columns.index('p_diff_3')
                  p4_idx = avg_data_columns.index('p_diff_4')
                  p_sta_1_idx = avg_data_columns.index('p_sta_1')
                  p_sta_1_sem_idx = avg_data_columns.index('p_sta_1_SEM')
                  p3_val = _safe_float(avg_data[-1][p3_idx])
                  p4_val = _safe_float(avg_data[-1][p4_idx])
                  if (p3_val is not None) and (p4_val is not None):
                    avg_data[-1][p_sta_1_idx] = round(p4_val - p3_val, 4)
                  p3_sem_idx = avg_data_columns.index('p_diff_3_SEM')
                  p4_sem_idx = avg_data_columns.index('p_diff_4_SEM')
                  p3_sem = _safe_float(avg_data[-1][p3_sem_idx])
                  p4_sem = _safe_float(avg_data[-1][p4_sem_idx])
                  if (p3_sem is not None) and (p4_sem is not None):
                    avg_data[-1][p_sta_1_sem_idx] = round(np.sqrt(p3_sem**2 + p4_sem**2), 4)
                # Electrical input power and efficiency
                if ('U_dc' in avg_params) and ('I_dc' in avg_params):
                  u_dc_val = _safe_float(avg_data[-1][avg_data_columns.index('U_dc')])
                  i_dc_val = _safe_float(avg_data[-1][avg_data_columns.index('I_dc')])
                  if (u_dc_val is not None) and (i_dc_val is not None):
                    p_dc = u_dc_val * i_dc_val
                    avg_data[-1][avg_data_columns.index('P_dc_mean')] = round(p_dc, 4)
                    if ('P_mech' in avg_params):
                      p_mech_val = _safe_float(avg_data[-1][avg_data_columns.index('P_mech')])
                      if (p_mech_val is not None) and (p_dc != 0):
                        avg_data[-1][avg_data_columns.index('eta_mech')] = round(p_mech_val / p_dc, 6)
                # Derived values from mean p_diff_1/2 and rho_air (negative p_dyn -> NaN)
                avg_data[-1][avg_data_columns.index('A_impeller')] = round(IMP_AREA, 6)
                if ('rho_air' in avg_params):
                  rho_val = avg_data[-1][avg_data_columns.index('rho_air')]
                else:
                  rho_val = ''
                if rho_val != '':
                  rho_val = float(rho_val)
                p1_val = avg_data[-1][avg_data_columns.index('p_diff_1')] if ('p_diff_1' in avg_params) else ''
                p2_val = avg_data[-1][avg_data_columns.index('p_diff_2')] if ('p_diff_2' in avg_params) else ''
                if p1_val != '' and rho_val != '':
                  p1_unit = avg_params_units[avg_params.index('p_diff_1')] if ('p_diff_1' in avg_params) else None
                  p1_pa = _to_pa(float(p1_val), p1_unit)
                  if p1_pa < 0:
                    p1_pa = np.nan
                  v1, m1 = calc_velocity_and_massflow(p1_pa, rho_val, IMP_AREA)
                  if not np.isnan(v1):
                    avg_data[-1][avg_data_columns.index('v_1_mean')] = round(v1, 4)
                    avg_data[-1][avg_data_columns.index('m_dot_1_mean')] = round(m1, 4)
                if p2_val != '' and rho_val != '':
                  p2_unit = avg_params_units[avg_params.index('p_diff_2')] if ('p_diff_2' in avg_params) else None
                  p2_pa = _to_pa(float(p2_val), p2_unit)
                  if p2_pa < 0:
                    p2_pa = np.nan
                  v2, m2 = calc_velocity_and_massflow(p2_pa, rho_val, IMP_AREA)
                  if not np.isnan(v2):
                    avg_data[-1][avg_data_columns.index('v_2_mean')] = round(v2, 4)
                    avg_data[-1][avg_data_columns.index('m_dot_2_mean')] = round(m2, 4)
                # Normierung der Massenströme (ISA) mit p_abs am Messpunkt
                t_amb_val = avg_data[-1][avg_data_columns.index('T_amb')] if ('T_amb' in avg_params) else ''
                p_amb_val = avg_data[-1][avg_data_columns.index('p_amb')] if ('p_amb' in avg_params) else ''
                p_dyn_1_val = avg_data[-1][avg_data_columns.index('p_diff_1')] if ('p_diff_1' in avg_params) else ''
                p_dyn_2_val = avg_data[-1][avg_data_columns.index('p_diff_2')] if ('p_diff_2' in avg_params) else ''
                p_sta_1_val = avg_data[-1][avg_data_columns.index('p_sta_1')]
                p_sta_2_val = avg_data[-1][avg_data_columns.index('p_diff_4')] if ('p_diff_4' in avg_params) else ''
                m_dot_1_val = avg_data[-1][avg_data_columns.index('m_dot_1_mean')]
                m_dot_2_val = avg_data[-1][avg_data_columns.index('m_dot_2_mean')]

                t_amb = _safe_float(t_amb_val)
                p_amb = _safe_float(p_amb_val)
                p_dyn_1 = _safe_float(p_dyn_1_val)
                p_dyn_2 = _safe_float(p_dyn_2_val)
                p_sta_1 = _safe_float(p_sta_1_val)
                p_sta_2 = _safe_float(p_sta_2_val)
                m_dot_1 = _safe_float(m_dot_1_val)
                m_dot_2 = _safe_float(m_dot_2_val)

                if t_amb is not None:
                  t_amb_k = _to_kelvin(t_amb)
                  avg_data[-1][avg_data_columns.index('T_amb_K_mean')] = round(t_amb_k, 3)
                else:
                  t_amb_k = None

                # Corrected speed using ambient temperature in Kelvin
                n_mot = _safe_float(avg_data[-1][avg_data_columns.index('N_mot')]) if 'N_mot' in avg_params else None
                if (t_amb_k is not None) and (n_mot is not None) and (t_amb_k > 0):
                  n_korr = n_mot * np.sqrt(T_NORM / t_amb_k)
                  avg_data[-1][avg_data_columns.index('N_korr')] = round(n_korr, 4)

                if p_amb is not None:
                  p_amb_unit = avg_params_units[avg_params.index('p_amb')] if ('p_amb' in avg_params) else None
                  p_amb_pa = _to_pa(p_amb, p_amb_unit)
                  avg_data[-1][avg_data_columns.index('p_amb_mean_pa')] = round(p_amb_pa, 1)
                else:
                  p_amb_pa = None

                if (t_amb_k is not None) and (p_amb_pa is not None) and (p_dyn_1 is not None) and (p_sta_1 is not None):
                  p1_abs = p_amb_pa + p_sta_1 + p_dyn_1
                  avg_data[-1][avg_data_columns.index('p1_abs_mean')] = round(p1_abs, 1)
                  k1 = _corr_factor(t_amb_k, p1_abs)
                  avg_data[-1][avg_data_columns.index('k1')] = round(k1, 6)
                  if m_dot_1 is not None:
                    avg_data[-1][avg_data_columns.index('m_dot_1_norm')] = round(m_dot_1 * k1, 6)

                if (t_amb_k is not None) and (p_amb_pa is not None) and (p_dyn_2 is not None) and (p_sta_2 is not None):
                  p2_abs = p_amb_pa + p_sta_2 + p_dyn_2
                  avg_data[-1][avg_data_columns.index('p2_abs_mean')] = round(p2_abs, 1)
                  k2 = _corr_factor(t_amb_k, p2_abs)
                  avg_data[-1][avg_data_columns.index('k2')] = round(k2, 6)
                  if m_dot_2 is not None:
                    avg_data[-1][avg_data_columns.index('m_dot_2_norm')] = round(m_dot_2 * k2, 6)

                if ('p1_abs' in locals()) and ('p2_abs' in locals()):
                  if (p1_abs is not None) and (p2_abs is not None) and (p1_abs > 0):
                    avg_data[-1][avg_data_columns.index('p2_p1_ratio')] = round(p2_abs / p1_abs, 6)

                # Unsicherheiten (SEM-basiert, 1σ) in SI-Einheiten
                rho_sem = _safe_float(avg_data[-1][avg_data_columns.index('rho_air_SEM')]) if 'rho_air' in avg_params else None
                p_dyn_1_sem = _safe_float(avg_data[-1][avg_data_columns.index('p_diff_1_SEM')]) if 'p_diff_1' in avg_params else None
                p_dyn_2_sem = _safe_float(avg_data[-1][avg_data_columns.index('p_diff_2_SEM')]) if 'p_diff_2' in avg_params else None
                p_sta_1_sem = _safe_float(avg_data[-1][avg_data_columns.index('p_sta_1_SEM')])
                p_sta_2_sem = _safe_float(avg_data[-1][avg_data_columns.index('p_diff_4_SEM')]) if 'p_diff_4' in avg_params else None
                t_amb_sem = _safe_float(avg_data[-1][avg_data_columns.index('T_amb_SEM')]) if 'T_amb' in avg_params else None
                p_amb_sem = _safe_float(avg_data[-1][avg_data_columns.index('p_amb_SEM')]) if 'p_amb' in avg_params else None

                u_t_k = t_amb_sem if t_amb_sem is not None else None
                if p_amb_sem is not None:
                  p_amb_unit = avg_params_units[avg_params.index('p_amb')] if ('p_amb' in avg_params) else None
                  u_p_amb_pa = _to_pa(p_amb_sem, p_amb_unit)
                else:
                  u_p_amb_pa = None

                v1_mean = _safe_float(avg_data[-1][avg_data_columns.index('v_1_mean')])
                v2_mean = _safe_float(avg_data[-1][avg_data_columns.index('v_2_mean')])
                if (rho_sem is not None) and (p_dyn_1_sem is not None) and (rho_val != '') and (v1_mean is not None):
                  u_v1 = _u_v_from_pdyn_rho(v1_mean, p_dyn_1_sem, float(rho_val), rho_sem)
                  if u_v1 is not None:
                    avg_data[-1][avg_data_columns.index('u_v_1_mean')] = round(u_v1, 6)
                if (rho_sem is not None) and (p_dyn_2_sem is not None) and (rho_val != '') and (v2_mean is not None):
                  u_v2 = _u_v_from_pdyn_rho(v2_mean, p_dyn_2_sem, float(rho_val), rho_sem)
                  if u_v2 is not None:
                    avg_data[-1][avg_data_columns.index('u_v_2_mean')] = round(u_v2, 6)

                u_v1_val = _safe_float(avg_data[-1][avg_data_columns.index('u_v_1_mean')])
                u_v2_val = _safe_float(avg_data[-1][avg_data_columns.index('u_v_2_mean')])
                m1_mean = _safe_float(avg_data[-1][avg_data_columns.index('m_dot_1_mean')])
                m2_mean = _safe_float(avg_data[-1][avg_data_columns.index('m_dot_2_mean')])
                if (rho_sem is not None) and (v1_mean is not None) and (u_v1_val is not None) and (rho_val != ''):
                  u_m1 = _u_mdot_from_rho_v(float(rho_val), rho_sem, v1_mean, u_v1_val, IMP_AREA)
                  avg_data[-1][avg_data_columns.index('u_m_dot_1_mean')] = round(u_m1, 6)
                if (rho_sem is not None) and (v2_mean is not None) and (u_v2_val is not None) and (rho_val != ''):
                  u_m2 = _u_mdot_from_rho_v(float(rho_val), rho_sem, v2_mean, u_v2_val, IMP_AREA)
                  avg_data[-1][avg_data_columns.index('u_m_dot_2_mean')] = round(u_m2, 6)

                if (u_p_amb_pa is not None) and (t_amb_k is not None) and (u_t_k is not None) and (p_amb_pa is not None):
                  if (p_dyn_1 is not None) and (p_sta_1 is not None):
                    p1_abs = p_amb_pa + p_sta_1 + p_dyn_1
                    u_p1 = np.sqrt(u_p_amb_pa**2 + (p_sta_1_sem or 0.0)**2 + (p_dyn_1_sem or 0.0)**2)
                    k1_val = _safe_float(avg_data[-1][avg_data_columns.index('k1')])
                    if k1_val is not None and p1_abs > 0 and m1_mean is not None:
                      u_k1 = _u_k_from_T_p(k1_val, t_amb_k, u_t_k, p1_abs, u_p1)
                      u_m1 = _safe_float(avg_data[-1][avg_data_columns.index('u_m_dot_1_mean')])
                      if u_m1 is not None:
                        u_m1n = _u_mnorm_from_m_k(m1_mean, u_m1, k1_val, u_k1)
                        avg_data[-1][avg_data_columns.index('u_m_dot_1_norm')] = round(u_m1n, 6)

                  if (p_dyn_2 is not None) and (p_sta_2 is not None):
                    p2_abs = p_amb_pa + p_sta_2 + p_dyn_2
                    u_p2 = np.sqrt(u_p_amb_pa**2 + (p_sta_2_sem or 0.0)**2 + (p_dyn_2_sem or 0.0)**2)
                    k2_val = _safe_float(avg_data[-1][avg_data_columns.index('k2')])
                    if k2_val is not None and p2_abs > 0 and m2_mean is not None:
                      u_k2 = _u_k_from_T_p(k2_val, t_amb_k, u_t_k, p2_abs, u_p2)
                      u_m2 = _safe_float(avg_data[-1][avg_data_columns.index('u_m_dot_2_mean')])
                      if u_m2 is not None:
                        u_m2n = _u_mnorm_from_m_k(m2_mean, u_m2, k2_val, u_k2)
                        avg_data[-1][avg_data_columns.index('u_m_dot_2_norm')] = round(u_m2n, 6)

                  if ('p1_abs' in locals()) and ('p2_abs' in locals()):
                    if (p1_abs is not None) and (p2_abs is not None) and (p1_abs > 0) and (p2_abs > 0):
                      ratio = _safe_float(avg_data[-1][avg_data_columns.index('p2_p1_ratio')])
                      if ratio is not None:
                        u_ratio = ratio * np.sqrt((u_p2 / p2_abs)**2 + (u_p1 / p1_abs)**2)
                        avg_data[-1][avg_data_columns.index('u_p2_p1_ratio')] = round(u_ratio, 6)
          if add_empty:
            print('  no steps found')
        else:
          print(f'  search paramter {stepfind_param} not in file')

        # Conditionally add line with zeros to final output to indicate no steps (above threshold) were found for this file
        if add_empty:
          avg_data.append(['' for n in range(len(avg_data_columns))])
          avg_data[-1][avg_data_columns.index('FILE')] = os.path.basename(data_files[datafile_n])
        else:
          # Text output
          print('  steps found:')
          for step_n in range(len(step_list)):
            print(f'  {step_data_tic[step_list[step_n][0]]}, {step_data_tic[step_list[step_n][1]]}, {round(step_list[step_n][2], 1)}, {round(step_list[step_n][3], 1)}')

        # Keep only one row per file: closest N_mot to rpm in filename; tie -> higher p2_p1_ratio
        if not add_empty:
          file_rows = avg_data[file_rows_start:]
          if len(file_rows) > 1:
            rpm_target = _extract_rpm(data_files[datafile_n])
            n_mot_idx = avg_data_columns.index('N_mot')
            ratio_idx = avg_data_columns.index('p2_p1_ratio')
            if rpm_target is None:
              print('  Hinweis: keine Soll-Drehzahl im Dateinamen gefunden (rpm fehlt).')
            best_idx = None
            best_key = None
            for i, row in enumerate(file_rows):
              n_mot_val = _safe_float(row[n_mot_idx]) if n_mot_idx < len(row) else None
              ratio_val = _safe_float(row[ratio_idx]) if ratio_idx < len(row) else None
              if rpm_target is None or n_mot_val is None:
                diff = float('inf')
              else:
                diff = abs(n_mot_val - rpm_target)
              # minimize diff; tie-breaker: higher ratio
              key = (diff, -(ratio_val if ratio_val is not None else -float('inf')))
              if best_key is None or key < best_key:
                best_key = key
                best_idx = i
            if best_idx is not None and best_key[0] != float('inf'):
              chosen = file_rows[best_idx]
              del avg_data[file_rows_start:]
              avg_data.append(chosen)
            else:
              reason = []
              if rpm_target is None:
                reason.append('rpm im Dateinamen fehlt')
              if n_mot_idx >= len(avg_data_columns):
                reason.append('N_mot-Spalte fehlt')
              else:
                all_n_mot_missing = all(_safe_float(r[n_mot_idx]) is None for r in file_rows)
                if all_n_mot_missing:
                  reason.append('N_mot in ROI leer')
              reason_text = ', '.join(reason) if reason else 'unbekannt'
              print(f'  Hinweis: kein passender Bereich gefunden ({reason_text}).')

        # Plot
        if plot_en:
          fig, ax = plt.subplots()
          plt.title(f'{data_descriptor}, {stepfind_param}')
          ax.plot(step_data_tic, step_data)
          for step_n in range(len(step_list)):
            ax.axvline(step_data_tic[step_list[step_n][0]], 0, 1, color = 'xkcd:green')
            ax.axvline(step_data_tic[step_list[step_n][1]], 0, 1, color = 'xkcd:red')
          plt.show()
        
      except:
        # File reading or processing error, mark in output file
        print('  data error!')
        avg_data.append(['' for n in range(len(avg_data_columns))])
        avg_data[-1][avg_data_columns.index('FILE')] = data_descriptor
        
    # Sort rows by Blende number and rpm before writing
    avg_data.sort(key = _sort_key)

    # Write file with all data combined
    avg_data_units = ['', 's', 's'] + avg_params_units + avg_params_sem_units + derived_params_units
    output_columns = list(avg_data_columns)
    output_columns[output_columns.index('p_diff_1')] = 'p_dyn_1'
    output_columns[output_columns.index('p_diff_2')] = 'p_dyn_2'
    output_columns[output_columns.index('p_diff_4')] = 'p_sta_2'
    output_columns[output_columns.index('p_diff_1_SEM')] = 'p_dyn_1_SEM'
    output_columns[output_columns.index('p_diff_2_SEM')] = 'p_dyn_2_SEM'
    output_columns[output_columns.index('p_diff_4_SEM')] = 'p_sta_2_SEM'
    output_columns = list(output_columns)
    output_units = list(avg_data_units)
    unit_by_col = dict(zip(output_columns, output_units))
    # First: all measured values, then derived with paired mean/uncertainty order
    def _rename_measured(name):
      if name == 'p_diff_1':
        return 'p_dyn_1'
      if name == 'p_diff_2':
        return 'p_dyn_2'
      if name == 'p_diff_4':
        return 'p_sta_2'
      if name == 'p_diff_1_SEM':
        return 'p_dyn_1_SEM'
      if name == 'p_diff_2_SEM':
        return 'p_dyn_2_SEM'
      if name == 'p_diff_4_SEM':
        return 'p_sta_2_SEM'
      return name

    measured_cols = ['FILE', 'ROI_START', 'ROI_STOP']
    measured_cols += [_rename_measured(n) for n in avg_params]
    # Place p_sta_1 directly after p_diff_3 in measured section
    if 'p_diff_3' in avg_params:
      insert_pos = measured_cols.index(_rename_measured('p_diff_3')) + 1
      measured_cols.insert(insert_pos, 'p_sta_1')
    measured_cols += [_rename_measured(n) for n in avg_params_sem]
    if 'p_diff_3_SEM' in avg_params_sem:
      insert_pos_sem = measured_cols.index(_rename_measured('p_diff_3_SEM')) + 1
      measured_cols.insert(insert_pos_sem, 'p_sta_1_SEM')
    measured_order = [c for c in measured_cols if c in output_columns]

    pair_groups = [
      ('v_1_mean', 'u_v_1_mean'),
      ('v_2_mean', 'u_v_2_mean'),
      ('m_dot_1_mean', 'u_m_dot_1_mean'),
      ('m_dot_1_norm', 'u_m_dot_1_norm'),
      ('m_dot_2_mean', 'u_m_dot_2_mean'),
      ('m_dot_2_norm', 'u_m_dot_2_norm'),
      ('p2_p1_ratio', 'u_p2_p1_ratio')
    ]
    remaining = [c for c in output_columns if c not in measured_order]
    derived_order = []
    for a, b in pair_groups:
      if a in remaining:
        derived_order.append(a)
        remaining.remove(a)
      if b in remaining:
        derived_order.append(b)
        remaining.remove(b)
    output_columns = measured_order + derived_order + remaining
    output_units = [unit_by_col.get(c, '') for c in output_columns]
    # Map output columns back to internal column names for row writing
    def _orig_col(name):
      if name == 'p_dyn_1':
        return 'p_diff_1'
      if name == 'p_dyn_2':
        return 'p_diff_2'
      if name == 'p_sta_2':
        return 'p_diff_4'
      if name == 'p_dyn_1_SEM':
        return 'p_diff_1_SEM'
      if name == 'p_dyn_2_SEM':
        return 'p_diff_2_SEM'
      if name == 'p_sta_2_SEM':
        return 'p_diff_4_SEM'
      return name
    order_idx = [avg_data_columns.index(_orig_col(c)) for c in output_columns]
    formula_by_col = {
      'A_impeller': 'A = pi*0.125^2',
      'p_sta_1': 'p_sta_1 = p_diff_4 - p_diff_3',
      'p_sta_1_SEM': 'u = sqrt(u3^2+u4^2)',
      'P_dc_mean': 'P_dc = U_dc*I_dc',
      'eta_mech': 'eta = P_mech/P_dc',
      'v_1_mean': 'v = sqrt(2*p_dyn/rho)',
      'v_2_mean': 'v = sqrt(2*p_dyn/rho)',
      'm_dot_1_mean': 'm_dot = rho*A*v',
      'm_dot_2_mean': 'm_dot = rho*A*v',
      'T_amb_K_mean': 'T_K = T_amb + 273.15',
      'p_amb_mean_pa': 'p_amb_Pa = p_amb_hPa*100',
      'N_korr': 'N_korr = N_mot*sqrt(T_norm/T_K)',
      'p_dyn_2': 'p_dyn_2 = p_dyn_1 (forced)',
      'p1_abs_mean': 'p1 = p_amb + p_sta_1 + p_dyn_1',
      'p2_abs_mean': 'p2 = p_amb + p_sta_2 + p_dyn_2',
      'k1': 'k = sqrt((T_K/T_norm)*(p_norm/p))',
      'k2': 'k = sqrt((T_K/T_norm)*(p_norm/p))',
      'm_dot_1_norm': 'm_dot_norm = m_dot*k',
      'm_dot_2_norm': 'm_dot_norm = m_dot*k',
      'p2_p1_ratio': 'p2/p1',
      'u_v_1_mean': 'u(v)=sqrt((dv/dp*u_p)^2+(dv/dr*u_r)^2)',
      'u_v_2_mean': 'u(v)=sqrt((dv/dp*u_p)^2+(dv/dr*u_r)^2)',
      'u_m_dot_1_mean': 'u(m)=sqrt((A*v*u_r)^2+(A*r*u_v)^2)',
      'u_m_dot_2_mean': 'u(m)=sqrt((A*v*u_r)^2+(A*r*u_v)^2)',
      'u_m_dot_1_norm': 'u(mn)=sqrt((k*u_m)^2+(m*u_k)^2)',
      'u_m_dot_2_norm': 'u(mn)=sqrt((k*u_m)^2+(m*u_k)^2)',
      'u_p2_p1_ratio': 'u(r)=r*sqrt((u_p2/p2)^2+(u_p1/p1)^2)'
    }
    output_formula = [formula_by_col.get(col, '') for col in output_columns]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if export_csv:
      output_file = f'{data_file_path}/data_avg_{timestamp}.csv'
      with open(output_file, 'w', newline = '', encoding = 'utf-8') as f:
        filewriter = csv.writer(f, delimiter = ';')
        # Write legend
        filewriter.writerow([f'step find parameters: {stepfind_param}, averaging interval {stepfind_interval} s, allowed standard deviation {stepfind_std_lim}'])
        filewriter.writerow(output_formula)
        filewriter.writerow(output_columns)
        filewriter.writerow(output_units)
        # Write one line for each step, transpose array for easier write
        for step_n in range(len(avg_data)):
          filewriter.writerow([avg_data[step_n][i] for i in order_idx])
    else:
      print('  note: CSV export disabled by GUI selection.')

    try:
      rows_for_map = [[avg_data[step_n][i] for i in order_idx] for step_n in range(len(avg_data))]
      df = {col: [row[col_n] for row in rows_for_map] for col_n, col in enumerate(output_columns)}
      # Only add case-normalized alias for DC power naming.
      if ('P_DC_mean' not in df) and ('P_dc_mean' in df):
        df['P_DC_mean'] = list(df['P_dc_mean'])
      if create_eff_map:
        plot_efficiency_map(
          df,
          savepath = os.path.join(data_file_path, 'efficiency_map.png'),
          save_png = export_png,
          save_pdf = export_pdf
        )
      else:
        print('  note: efficiency_map export disabled by GUI selection.')

      if create_motor_map:
        plot_motor_efficiency_map_torque_speed(
          df,
          out_dir = data_file_path,
          fname = 'motor_eff_map_torque_speed_like_ref',
          save_png = export_png,
          save_pdf = export_pdf
        )
      else:
        print('  note: motor_eff_map_torque_speed export disabled by GUI selection.')

      if create_torque_map:
        plot_torque_contour_map(
          df,
          savepath = os.path.join(data_file_path, 'torque_contour_map.png'),
          save_png = export_png,
          save_pdf = export_pdf
        )
      else:
        print('  note: torque_contour_map export disabled by GUI selection.')
    except Exception as e:
      print(f'  warning: efficiency map not created ({e})')

  except KeyboardInterrupt:
    sys.exit()

  finally:
    pass
    
if __name__ == '__main__':
    main()
