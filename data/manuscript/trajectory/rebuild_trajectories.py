"""Rebuild three vector trajectory panels from author-supplied ECharts HTML.

Parse the option object as JSON without executing HTML or JavaScript. Preserve
every recorded path, obstacle-grid vertex, dynamic-track point and virtual
source. The orthographic camera is fixed to match the original near-top view.
No training, resampling, smoothing or time-base inference is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
MAPPING = [
    ('16a', '95201273-ff28-4956-acb9-8187fd9e8a32.html', 'artwork29.pdf', 'Figure16a_trajectory.pdf', 5.1),
    ('17a', '0f2fb3ad-448a-4c82-91ad-3965703aa3af.html', 'artwork31.pdf', 'Figure17a_trajectory_V1.pdf', 2.65),
    ('17b', '0acd1113-3e42-4835-a4cb-3bdbea617ef7.html', 'artwork32.pdf', 'Figure17b_trajectory_V4.pdf', 2.65),
]
BLUE = '#5070DD'  # Original ECharts 6 series color, verified in old PNGs.
ELEVATION = np.deg2rad(78.0)
COLORS = {'trajectory': BLUE, 'static': '#FF0000', 'dynamic': '#008000',
          'dynamic_track': '#90EE90', 'start': '#00FF00', 'goal': '#FF0000',
          'virtual_surface': '#800080', 'virtual_center': '#8A2BE2', 'callout': '#FF0000'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_html(path):
    raw = path.read_text(encoding='utf-8-sig')
    match = re.search(r'var option_\w+\s*=\s*', raw)
    if not match:
        raise ValueError(f'Missing ECharts option object: {path}')
    return json.JSONDecoder().raw_decode(raw[match.end():])[0]


def project(data):
    a = np.asarray(data, dtype=float)
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    return np.stack(((x+y)/np.sqrt(2), np.sin(ELEVATION)*(y-x)/np.sqrt(2) + np.cos(ELEVATION)*z), axis=-1)


def image_count(page):
    resources = page.get('/Resources', {})
    if hasattr(resources, 'get_object'): resources=resources.get_object()
    xo = resources.get('/XObject', {})
    if hasattr(xo, 'get_object'): xo=xo.get_object()
    return sum(1 if obj.get_object().get('/Subtype') == '/Image' else
               image_count(obj.get_object()) if obj.get_object().get('/Subtype') == '/Form' else 0
               for obj in xo.values())


def render(panel, option, target, width):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.0 if width < 3 else 10.0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'path.simplify': False})
    fig = plt.figure(figsize=(width, width*0.96), facecolor='white')
    ax = fig.add_axes([0.015, 0.025, 0.97, 0.96])
    ax.set_aspect('equal'); ax.set_axis_off()
    # Ground-plane grid, with readable 100-grid tick intervals.
    for value in range(0, 501, 25):
        for points in [[[value,0,0],[value,500,0]], [[0,value,0],[500,value,0]]]:
            q=project(points); ax.plot(q[:,0],q[:,1],color='#D2D2D2',lw=0.32,zorder=0)
    for a,b in [([0,0,0],[500,0,0]),([500,0,0],[500,500,0])]:
        q=project([a,b]); ax.plot(q[:,0],q[:,1],color='#333333',lw=0.7,zorder=1)
    font = 7.8 if width < 3 else 9.8
    for value in range(0,501,100):
        q=project([[value,0,0]])[0]
        ax.text(q[0]-18,q[1]-18,str(value),ha='right',va='top',fontsize=font,color='#333333',zorder=20,bbox={'facecolor':'white','edgecolor':'none','pad':0.1})
        q=project([[500,value,0]])[0]
        offset=78 if panel=='17b' and value==400 else 18
        ax.text(q[0]+offset,q[1]-18,str(value),ha='left',va='top',fontsize=font,color='#333333',zorder=20,bbox={'facecolor':'white','edgecolor':'none','pad':0.1})
    ax.text(70,-335,'X (grid)',ha='center',va='center',fontsize=font,zorder=20)
    ax.text(650,-335,'Y (grid)',ha='center',va='center',fontsize=font,zorder=20)
    series=option['series']; line=series[0]
    assert line['type']=='line3D'
    trajectory=np.asarray(line['data'], dtype=float)
    q=project(trajectory); ax.plot(q[:,0],q[:,1],color=BLUE,lw=0.75 if width<3 else 1.0,zorder=2)
    # The HTML stores dynamic motion as scatter points; retain all positions.
    for s in series:
        if s.get('name','').startswith('Dynamic Obstacle Path'):
            q=project(s['data'])
            ax.scatter(q[:,0],q[:,1],s=7 if width<3 else 17,color=COLORS['dynamic_track'],alpha=s['itemStyle']['opacity'],edgecolors='none',zorder=2.1)
    surfaces=[]
    for s in series:
        if s['type']!='surface': continue
        data=np.asarray(s['data'],dtype=float); n=int(round(np.sqrt(len(data))))
        assert n*n==len(data)
        a=data.reshape(n,n,3)
        faces=np.stack((a[:-1,:-1],a[1:,:-1],a[1:,1:],a[:-1,1:]),axis=2).reshape(-1,4,3)
        # Retain zero-height square-grid regions exactly as encoded in HTML.
        depth=(np.cos(ELEVATION)*(faces[...,0]-faces[...,1])/np.sqrt(2)+np.sin(ELEVATION)*faces[...,2]).mean(axis=1)
        order=np.argsort(depth)
        surfaces.append((float(depth.mean()),project(faces[order]),s['itemStyle']))
    for i,(_,faces,style) in enumerate(sorted(surfaces,key=lambda v:(v[2]['color']=='purple',v[0]))):
        is_virtual=style['color']=='purple'
        if is_virtual:
            # Virtual sources must remain visibly purple over overlapping red
            # physical obstacles. Keep the original purple face opacity while
            # drawing their purple mesh on top of physical-obstacle meshes.
            ax.add_collection(PolyCollection(faces,
                             facecolors=matplotlib.colors.to_rgba(COLORS['virtual_surface'],style['opacity']),
                             edgecolors=matplotlib.colors.to_rgba(COLORS['virtual_surface'],0.95),
                             linewidths=0.23 if width<3 else 0.35,zorder=4+i/100))
        else:
            ax.add_collection(PolyCollection(faces,facecolors=style['color'],alpha=style['opacity'],
                             edgecolors='#555555',linewidths=0.10 if width<3 else 0.16,zorder=3+i/100))
    # Start, goal and source centers are displayed as in the original figures.
    for s in series:
        if s['type']=='scatter3D' and not s.get('name','').startswith('Dynamic Obstacle Path'):
            q=project(s['data'])
            ax.scatter(q[:,0],q[:,1],s=(11 if width<3 else 25),c=s['itemStyle']['color'],edgecolors='none',zorder=5)
    callout=None
    if panel=='17b':
        # The red frame is an original figure annotation, not an obstacle or data.
        # Enclose the same final, oscillatory portion without altering the path.
        q=project(trajectory[trajectory[:,0]>=350])
        low=q.min(axis=0)-[14,20]; high=q.max(axis=0)+[32,26]
        callout=[*low,*(high-low)]
        ax.add_patch(Rectangle(low,*(high-low),fill=False,edgecolor=COLORS['callout'],lw=1.5,zorder=30))
    ax.set_xlim(-68,785); ax.set_ylim(-415,365)
    fig.savefig(target,metadata={'Title':f'Figure {panel}: recorded navigation trajectory',
                               'Subject':'Author experience HTML data; original colors; vector paths and surfaces'})
    fig.savefig(target.with_suffix('.png'),dpi=220)
    plt.close(fig)
    reader=PdfReader(target); assert len(reader.pages)==1 and image_count(reader.pages[0])==0
    obstacles=[s for s in series if s['type']=='surface' and s['itemStyle']['color'] in ('red','green')]
    return {'Panel':panel,'File':target.name,'SHA256':sha(target),'Vector_Image_Objects':0,
            'Trajectory_Points':len(trajectory),'Trajectory_SHA256':hashlib.sha256(trajectory.tobytes()).hexdigest(),
            'Static_Obstacles':sum(s['itemStyle']['color']=='red' for s in obstacles),
            'Dynamic_Obstacles':sum(s['itemStyle']['color']=='green' for s in obstacles),
            'Dynamic_Tracks':sum(s.get('name','').startswith('Dynamic Obstacle Path') for s in series),
            'Virtual_Sources':sum(len(s['data']) for s in series if s.get('name')=='Virtual Center'),
            'Virtual_Surface_Rendering':'Original purple face at original opacity; purple mesh above physical obstacles',
            'All_Encoded_Surface_Vertices':sum(len(s['data']) for s in series if s['type']=='surface'),
            'Start':next(s['data'][0] for s in series if s.get('name')=='Start Point'),
            'Goal':next(s['data'][0] for s in series if s.get('name')=='Goal Point'),
            'Original_Red_Frame':callout,'Width_pt':float(reader.pages[0].mediabox.width),
            'Height_pt':float(reader.pages[0].mediabox.height)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'experience_html')
    parser.add_argument('--output',type=Path,default=HERE/'vectors')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    result={'Original_Colors':COLORS,'Camera':{'Projection':'orthographic','Elevation_deg':78,'Azimuth_deg':-45},
            'No_Smoothing_Or_Resampling':True,'Coordinates_From_HTML':True,'No_Timing_Inferred_From_Point_Counts':True,'Panels':[]}
    for panel,name,asset,remote,width in MAPPING:
        file=args.input/name; option=parse_html(file)
        row=render(panel,option,args.output/asset,width)
        row.update({'Source_HTML':name,'Source_SHA256':sha(file),'Overleaf_Path':'Picture/trajectory/'+remote})
        result['Panels'].append(row)
    (args.output/'trajectory_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
