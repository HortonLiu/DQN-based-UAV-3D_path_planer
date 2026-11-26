"""Rebuild manuscript success plots from the author-supplied binary episodes.

Run with Python, matplotlib, numpy and pypdf. No training is performed.
Input data are preserved byte-for-byte. The plotted values are recomputed,
not inferred from screenshot pixels or rounded manuscript percentages.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
# Original TensorBoard series colors, verified against the pre-vector PNGs.
# In overlapped translucent screenshots some visible pixels are mixtures of
# two series; these are the underlying series colors, not those mixtures.
ORIGINAL_COLORS = {
    18: ['#009988'], 19: ['#BBBBBB'], 20: ['#FF7043'],
    21: ['#0077BB'], 22: ['#009988'], 23: ['#BBBBBB'],
    24: ['#CC3311'], 25: ['#FF7043'],
    26: ['#0077BB', '#FF7043'],
    27: ['#0077BB', '#FF7043', '#EE3377'],
    28: ['#CC3311', '#33BBEE', '#009988'],
    30: ['#FF7043', '#0077BB'],
    33: ['#0077BB', '#CC3311'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def recompute(success, beta):
    accumulator = 0.0
    out = []
    for count, value in enumerate(success, 1):
        accumulator = beta * accumulator + (1.0 - beta) * value
        out.append(accumulator / (1.0 - beta ** count))
    return np.array(out)


def vector_images(page):
    resources = page.get('/Resources', {})
    resources = resources.get_object()
    xobjects = resources.get('/XObject', {})
    total = 0
    for item in xobjects.get_object().values() if xobjects else []:
        obj = item.get_object()
        if obj.get('/Subtype') == '/Image':
            total += 1
        elif obj.get('/Subtype') == '/Form':
            total += vector_images(obj)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=HERE / 'vectors')
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--copy-input', type=Path)
    args = parser.parse_args()
    if args.copy_input:
        args.input.mkdir(parents=True, exist_ok=True)
        for original in args.copy_input.rglob('*'):
            if not original.is_file():
                continue
            target = args.input / original.relative_to(args.copy_input)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == sha(original), target
            else:
                shutil.copy2(original, target)
            assert sha(target) == sha(original)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = csv_rows(args.input / 'manifest.csv')
    assert len(rows) == 20
    datasets = {}
    report = []
    for row in rows:
        samples = csv_rows(args.input / row['Curve_CSV'])
        raw = csv_rows(args.input / row['Raw_CSV'])
        episode = np.array([int(s['Episode']) for s in samples])
        outcome = np.array([float(s['Success']) for s in samples])
        assert np.array_equal(episode, np.arange(1, len(samples) + 1))
        assert set(outcome).issubset({0, 1})
        assert len(raw) == len(samples)
        assert np.array_equal(episode, [int(float(s['Step'])) for s in raw])
        assert np.array_equal(outcome, [float(s['Value']) for s in raw])
        y = recompute(outcome, args.beta)
        if args.beta == .99:
            supplied = [float(s['EMA_0.99_TensorBoard']) for s in samples]
            assert np.max(np.abs(y - supplied)) < 5.1e-13
        elif args.beta == .999:
            supplied = [float(s['EMA_0.999_TensorBoard_Artwork']) for s in samples]
            assert np.max(np.abs(y - supplied)) < 5.1e-13
        datasets[row['Curve_CSV']] = (episode, y, row)
        report.append({
            'Figure': row['Figure'], 'Panel': row['Panel'], 'Curve': row['Curve'],
            'Artwork': row['Artwork'].replace('.png', '.pdf'),
            'Run': row['TensorBoard_Run'], 'First_Episode': int(episode[0]),
            'Last_Episode': int(episode[-1]), 'Episodes': len(samples),
            'Successful_Episodes': int(outcome.sum()), 'Beta': args.beta,
            'Bias_Correction': 'W_m / (1 - beta**m), W_0 = 0',
            'Final_Smoothed_Percent': float(y[-1] * 100),
            'Displayed_Final_Percent': f'{y[-1] * 100:.2f}',
            'Source_Curve_CSV': row['Curve_CSV'],
            'Source_Raw_CSV': row['Raw_CSV'],
            'CSV_SHA256': sha(args.input / row['Curve_CSV']),
        })

    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'font.size': 12.5,
        'axes.labelsize': 12.5, 'xtick.labelsize': 11.5,
        'ytick.labelsize': 11.5, 'legend.fontsize': 11.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': .8, 'axes.edgecolor': '#444444',
        'text.color': '#222222', 'axes.labelcolor': '#222222',
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    # Only the smoothed series is displayed. Binary outcomes are retained in
    # the distributed data, not overlaid as opaque screenshot backgrounds.
    groups = {}
    for row in rows:
        groups.setdefault(row['Artwork'], []).append(row)
    records = []
    for artwork, members in groups.items():
        art_num = int(artwork[7:9])
        has_legend = len(members) > 1
        fig, ax = plt.subplots(figsize=(3.25, 2.60 if has_legend else 2.35))
        fig.subplots_adjust(left=.205, right=.91, bottom=.245,
                            top=.765 if has_legend else .86)
        for number, member in enumerate(members):
            x, y, _ = datasets[member['Curve_CSV']]
            label = None
            style = '-'
            if art_num == 26:
                label = ['Level 2', 'Level 3'][number]
                style = ['-', '--'][number]
            elif art_num in (27, 28):
                label = ['Level 1', 'Level 2', 'Level 3'][number]
                style = ['-', '--', '-.'][number]
            elif art_num == 30:
                label = ['[1] Mixed', '[2] Static, 1 m/s'][number]
                style = ['--', '-'][number]
            elif art_num == 33:
                label = ['[2] Static, 1 m/s', '[3] Static, 4 m/s'][number]
                style = ['-', '--'][number]
            color = ORIGINAL_COLORS[art_num][number]
            ax.plot(x, y, color=color, linewidth=1.65, linestyle=style,
                    label=label, solid_capstyle='round')
        ax.set_xlim(0, 1000)
        ax.set_ylim(-.025, 1.025)
        ax.set_xticks([0, 500, 1000])
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.set_xlabel('Episode', labelpad=4)
        ax.set_ylabel('Success rate', labelpad=5)
        ax.set_axisbelow(True)
        ax.grid(True, color='#D9DDE0', linewidth=.65)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=3.0, width=.7, pad=3)
        if has_legend:
            legend = ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.035),
                               frameon=False, borderaxespad=0, labelspacing=.2,
                               handlelength=1.6, handletextpad=.5,
                               ncol=1 if art_num in (30, 33) else len(members),
                               columnspacing=.8, fontsize=10.2 if art_num in (27, 28) else 11.3)
            if art_num in (27, 28):
                # A single horizontal legend would exceed the printed column.
                legend.remove()
                handles, labels = ax.get_legend_handles_labels()
                order = [0, 2, 1]
                ax.legend([handles[i] for i in order], [labels[i] for i in order],
                          loc='lower center', bbox_to_anchor=(.5, 1.015),
                          frameon=False, borderaxespad=0, labelspacing=.1,
                          handlelength=1.5, handletextpad=.45, ncol=2,
                          columnspacing=.65, fontsize=11.3)
                fig.subplots_adjust(top=.755)
        else:
            level = 'Level 2' if art_num == 21 else 'Level 3'
            ax.text(.98, 1.045, level, ha='right', va='bottom',
                    transform=ax.transAxes, fontsize=11.5, color='#333333')
        target = args.output / artwork.replace('.png', '.pdf')
        fig.savefig(target, metadata={'Title': f'Success-rate artwork {art_num}',
                    'Subject': f'Author-supplied binary outcomes; bias-corrected EMA beta={args.beta}; no resampling',
                    'Creator': 'rebuild_plots.py / matplotlib'})
        fig.savefig(target.with_suffix('.png'), dpi=180)
        plt.close(fig)
        page = PdfReader(target).pages[0]
        count = vector_images(page)
        assert count == 0
        records.append({'File': target.name, 'SHA256': sha(target),
                        'Width_pt': float(page.mediabox.width),
                        'Height_pt': float(page.mediabox.height),
                        'Raster_Image_Objects': count, 'Curves': len(members),
                        'Original_Series_Colors': ORIGINAL_COLORS[art_num]})
    with (args.output / 'recomputed_endpoints.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    validation = {'beta': args.beta, 'bias_corrected': True, 'curves': len(rows),
                  'distinct_raw_csvs': len({r['Raw_CSV'] for r in rows}),
                  'raw_binary_values_verified': True, 'supplied_ema_verified': True,
                  'resampling_or_padding': False,
                  'author_confirmed_Fig15_Level2_old_folder_name': True,
                  'series_colors_restored_from_original_screenshots': ORIGINAL_COLORS,
                  'artwork': records, 'endpoints': report}
    (args.output / 'plot_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps({'beta': args.beta, 'curves': len(rows), 'plots': len(records),
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
