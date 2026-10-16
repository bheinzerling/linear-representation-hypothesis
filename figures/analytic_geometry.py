#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Text3D
from mpl_toolkits.mplot3d import proj3d

import coefficient_evidence as evidence
from svg_tools import math_font_css, svg

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-05-figure7-geometric-steering-failures"
OUT = ROOT / "data/figures/analytic"
INK, GRAY, LINE = "#25231f", "#696969", "#dedede"
BLUE, TEAL, GOLD, ERROR = "#35618d", "#16766f", "#ac7d00", "#b9602b"



def analytic_cases():
    """Specify and verify exact counterexamples; never infer output from distance."""
    t = np.linspace(0, 1, 101)
    records = {}

    def add(key, manifold, decoder, downstream, source, target, edit, target_probe,
            requested_output, d_fn, f_fn, original, natural_labels, **extra):
        source, target, edit = map(lambda x: np.asarray(x, dtype=float), [source, target, edit])
        assert np.allclose(d_fn(original), natural_labels, atol=1e-12)
        assert np.allclose(f_fn(original), extra.pop("original_outputs", natural_labels), atol=1e-12)
        assert np.allclose(d_fn(edit), target_probe, atol=1e-12)
        assert np.allclose(f_fn(target), requested_output, atol=1e-12)
        assert not np.allclose(f_fn(edit), requested_output, atol=1e-12)
        def value(x):
            return np.asarray(x).tolist()
        records[key] = dict(original_states=manifold, decoder=decoder, downstream=downstream,
            source=value(source), target=value(target), edit=value(edit), target_probe=value(target_probe),
            requested_output=requested_output, edited_probe=value(d_fn(edit)), edited_output=value(f_fn(edit)),
            original_samples_checked=len(original), **extra)

    add("line", "h(t)=(t,t)", "d(x,y)=x; P(x,y)=sqrt(2)*x", "F(x,y)=y",
        [0,0], [1,1], [1,0], 1, 1, lambda h:h[...,0], lambda h:h[...,1],
        np.column_stack([t,t]), t, distance_scaling="P preserves all pairwise distances on y=x")
    assert np.allclose(np.sqrt(2)*np.abs(t[:,None]-t[None,:]),
                       np.linalg.norm(np.column_stack([t,t])[:,None,:]-np.column_stack([t,t])[None,:,:],axis=-1))
    natural = np.column_stack([t,t*t])
    f = lambda h:h[...,0]+h[...,1]-h[...,0]**2
    add("parabola", "h(t)=(t,t^2)", "d(x,y)=x", "F_lambda(x,y)=x+lambda*(y-x^2), lambda=1",
        [0,0], [1,1], [1,0], 1, 1, lambda h:h[...,0], f, natural, t,
        downstream_family="All lambda agree with d on the parabola; F_lambda(1,0)=1-lambda")
    tc=np.linspace(-1,1,201)
    add("chord", "h(t)=(t,t^2), -1<=t<=1", "d(x,y)=x", "F(x,y)=x+y-x^2",
        [-1,1], [0,0], [0,1], 0, 0, lambda h:h[...,0], f,
        np.column_stack([tc,tc*tc]), tc, chord="h_chord(t)=(t,1)",
        chord_output="t+1-t^2", valid_endpoints=[[-1,1],[1,1]])
    assert np.allclose(f(np.array([[-1,1],[1,1]])),[-1,1])
    helix=np.column_stack([np.cos(np.pi*t/2),np.sin(np.pi*t/2),t])
    phase=lambda h:2/np.pi*np.arctan2(h[...,1],h[...,0])
    add("helix_scalar", "h(t)=(cos(pi*t/2),sin(pi*t/2),t), 0<=t<=1", "d(x,y,z)=z",
        "F(x,y,z)=2/pi*atan2(y,x)", [1,0,0], [0,1,1], [1,0,1], 1, 1,
        lambda h:h[...,2], phase, helix,t)
    add("helix_circular", "h(t)=(cos(pi*t/2),sin(pi*t/2),t), 0<=t<=1",
        "P(x,y,z)=(x,y); d=2/pi*atan2(P_2,P_1)", "F(x,y,z)=z",
        [1,0,0], [0,1,1], [0,1,0], 1, 1,phase,lambda h:h[...,2],helix,t,
        edited_circular_code=[0,1],target_circular_code=[0,1])
    assert np.allclose(records['helix_circular']['edit'][:2],records['helix_circular']['target'][:2])
    natural_context=np.concatenate([np.column_stack([t,c*t*t,np.full_like(t,c)]) for c in [1,2]])
    add("context", "h(t,c)=(t,c*t^2,c), c in {1,2}", "d(x,y,c)=x",
        "F(x,y,c)=x+y-c*x^2", [0,0,2], [1,2,2], [1,1,2], 1, 1,
        lambda h:h[...,0],lambda h:h[...,0]+h[...,1]-h[...,2]*h[...,0]**2,
        natural_context,np.tile(t,2), fitted_displacement=[1,1,0], context1_output=1)
    u,v=np.meshgrid(t[::5],t[::5]); original=np.column_stack([u.ravel(),v.ravel(),(u*v).ravel()])
    add("surface", "h(u,v)=(u,v,u*v), 0<=u,v<=1", "P(u,v,z)=(u,v)", "F(u,v,z)=z",
        [0,0,0], [1,1,1], [1,1,0], [1,1], 1,lambda h:h[...,:2],lambda h:h[...,2],
        original,original[:,:2],original_outputs=original[:,2], individual_edits=[[1,0,0],[0,1,0]],
        omitted_interaction=1, independent_displacements=[[1,0,0],[0,1,0]],
        available_edit_basis=[[1,0,0],[0,1,0],[0,0,1]],
        independent_coefficients="(u,v,0)",joint_coefficients="(u,v,u*v)")
    assert np.allclose(np.array(records['surface']['individual_edits']).sum(axis=0),[1,1,0])
    for u,v,z in records['surface']['individual_edits']:
        assert z==u*v
    for lam in [0,.25,1,2]:
        assert np.allclose(natural[:,0]+lam*(natural[:,1]-natural[:,0]**2),t)
    return records


def rel_ax(fig, rect, area, **kwargs):
    x,y,w,h=rect; a,b,c,d=area
    return fig.add_axes([x+a*w,y+b*h,c*w,d*h], **kwargs)


def txt(fig, rect, x, y, text, size=7.5, color=INK, ha="left", **kwargs):
    a,b,w,h=rect
    return fig.text(a+x*w,b+y*h,text,fontsize=size,color=color,ha=ha,va="center",**kwargs)


def axes_style(ax, limits, xticks=(0,1), yticks=(0,1)):
    ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1])
    ax.set_xticks(xticks);ax.set_yticks(yticks)
    ax.tick_params(labelsize=6.2,pad=1.5,length=2,width=.5)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['bottom','left']:ax.spines[side].set_color('#a4a4a4')


def point(ax, p, kind, label=None, offset=(4,4)):
    marker,color,fill={'source':('o',BLUE,BLUE),'target':('D',TEAL,TEAL),'edit':('s',GOLD,'white')}[kind]
    ax.plot(*p,marker=marker,ms=4.7,mec=color,mfc=fill,mew=1,ls='none',zorder=6)
    if label:
        ax.annotate(label,p,xytext=offset,textcoords='offset points',fontsize=7,color=color,zorder=7)


def arrow(ax, source, target, color=GOLD, **kwargs):
    ax.annotate('',xy=target,xytext=source,arrowprops=dict(arrowstyle='->',color=color,lw=1.2,
                shrinkA=4,shrinkB=4,mutation_scale=7,**kwargs),zorder=5)


def result_block(fig, rect, target='1', edited='0', x=.66, decoder='d', y=.83):
    txt(fig,rect,x,y,'Probe-matching edit',size=7)
    txt(fig,rect,x,y-.16,rf'${decoder}(E)={target}$',color=TEAL,size=9)
    txt(fig,rect,x,y-.31,rf'$F(E)={edited}$',color=ERROR,size=9)
    txt(fig,rect,x,y-.44,'Target activation',size=7)
    txt(fig,rect,x,y-.59,rf'$F(T)={target}$',color=TEAL,size=9)


def draw_projection(fig,rect):
    ax=rel_ax(fig,rect,[.095,.27,.47,.63])
    axes_style(ax,[(-.14,1.28),(-.15,1.32)])
    t=np.linspace(0,1.18,100);ax.plot(t,t,color=BLUE,lw=1.5)
    ax.axvline(1,color=LINE,lw=.8,ls=':');ax.plot([1,1],[0,1],color=GRAY,lw=.8,ls=':')
    arrow(ax,(0,0),(1,0))
    point(ax,(0,0),'source','S',(3,6));point(ax,(1,0),'edit','E',(5,4));point(ax,(1,1),'target','T',(4,0))
    txt(fig,rect,.045,.91,'y',size=7);txt(fig,rect,.59,.28,'x',size=7)
    result_block(fig,rect)
    txt(fig,rect,.5,.075,r'$h(t)=(t,t),\quad d=x,\quad F=y$',size=7.8,ha='center')


def draw_parabola(fig,rect,contours=False):
    ax=rel_ax(fig,rect,[.095,.27,.47,.63])
    axes_style(ax,[(-.15,1.32),(-.18,1.43)])
    if contours:
        xx,yy=np.meshgrid(np.linspace(-.15,1.32,100),np.linspace(-.18,1.43,100))
        field=xx+yy-xx*xx
        cs=ax.contour(xx,yy,field,levels=[0,.5,1,1.5],colors=['#c4c4c4']*4,linewidths=.6)
        ax.contour(xx,yy,field,levels=[1],colors=[TEAL],linewidths=1.35,linestyles='--')
        ax.text(.13,1.12,r'$F=1$',fontsize=7,color=TEAL)
    t=np.linspace(0,1.18,100);ax.plot(t,t*t,color=BLUE,lw=1.5)
    ax.axvline(1,color=GRAY,lw=.8,ls=':')
    if not contours:ax.plot([1,1],[0,1],color=GRAY,lw=.8,ls=':')
    arrow(ax,(0,0),(1,0))
    point(ax,(0,0),'source','S',(3,6));point(ax,(1,0),'edit','E',(5,4));point(ax,(1,1),'target','T',(4,0))
    txt(fig,rect,.045,.91,'y',size=7);txt(fig,rect,.59,.28,'x',size=7)
    result_block(fig,rect)
    txt(fig,rect,.5,.075,r'$h(t)=(t,t^2),\quad d=x,\quad F=x+y-x^2$',size=7.6,ha='center')


def chord_geometry(ax):
    axes_style(ax,[(-1.25,1.23),(-.16,1.38)],xticks=[-1,0,1])
    t=np.linspace(-1,1,151);ax.plot(t,t*t,color=BLUE,lw=1.6)
    ax.plot([-1,1],[1,1],color=GOLD,lw=1.3,ls='--')
    ax.plot([0,0],[0,1],color=GRAY,lw=.8,ls=':')
    point(ax,(-1,1),'source','S',(-10,5))
    ax.plot(1,1,marker='o',ms=4.2,color=GRAY,ls='none',zorder=6)
    ax.annotate(r'$h(1)$',(1,1),xytext=(-17,6),textcoords='offset points',fontsize=6.5,color=GRAY)
    point(ax,(0,1),'edit','E',(4,5));point(ax,(0,0),'target','T',(5,0))


def draw_chord(fig,rect):
    ax=rel_ax(fig,rect,[.09,.28,.47,.60]);chord_geometry(ax)
    ax.text(.50,.47,'curve',color=BLUE,fontsize=6.8,rotation=35)
    ax.text(-.57,1.16,'chord',color=GOLD,fontsize=7)
    txt(fig,rect,.045,.91,'y',size=7);txt(fig,rect,.59,.28,'x',size=7)
    result_block(fig,rect,target='0',edited='1')
    txt(fig,rect,.5,.08,r'$d=x,\quad F=x+y-x^2$',size=7.8,ha='center')


def draw_profiles(fig,rect):
    ax=rel_ax(fig,rect,[.085,.40,.33,.45]);chord_geometry(ax)
    ax.tick_params(labelsize=6)
    bx=rel_ax(fig,rect,[.60,.40,.34,.45])
    axes_style(bx,[(-1.25,1.23),(-1.2,1.45)],xticks=[-1,0,1],yticks=[-1,0,1])
    t=np.linspace(-1,1,151)
    bx.plot(t,t,color=BLUE,lw=1.5)
    bx.plot(t,t+1-t*t,color=GOLD,lw=1.25,ls='--')
    point(bx,(0,1),'edit',None);point(bx,(0,0),'target',None)
    txt(fig,rect,.25,.94,'Activation plane',size=7.2,ha='center')
    txt(fig,rect,.77,.94,'Model output',size=7.2,ha='center')
    ax.set_xlabel('x',fontsize=7,labelpad=0)
    bx.set_xlabel('Decoded target t',fontsize=7,labelpad=0)
    txt(fig,rect,.05,.065,'Curve: F = d = t',color=BLUE,size=7.2)
    txt(fig,rect,.54,.065,r'Chord: $F=t+1-t^2$',color=GOLD,size=7.2)


def helix_axes(fig,rect):
    ax=rel_ax(fig,rect,[.005,.11,.62,.83],projection='3d')
    t=np.linspace(0,1,150)
    ax.plot(np.cos(np.pi*t/2),np.sin(np.pi*t/2),t,color=BLUE,lw=1.8)
    q=np.linspace(0,np.pi/2,80)
    ax.plot(np.cos(q),np.sin(q),np.zeros_like(q),color=LINE,lw=.7)
    for v in [0,1]:
        ax.plot([0,1.2],[0,0],[v,v],color=LINE,lw=.55)
        ax.plot([0,0],[0,1.2],[v,v],color=LINE,lw=.55)
    ax.plot([0,0],[0,0],[0,1.22],color=GRAY,lw=.6)
    ax.text(1.2,0,0,'x',fontsize=6.5,color=GRAY)
    ax.text(0,1.2,0,'y',fontsize=6.5,color=GRAY)
    ax.text(0,0,1.23,'z',fontsize=6.5,color=GRAY)
    ax.set_xlim(-.1,1.3);ax.set_ylim(-.1,1.3);ax.set_zlim(-.1,1.35)
    ax.set_box_aspect([1,1,1]);ax.view_init(elev=22,azim=38);ax.set_axis_off()
    return ax


def point3(ax,p,kind,label=None,offset=(0,0,.05)):
    marker,color,fill={'source':('o',BLUE,BLUE),'target':('D',TEAL,TEAL),'edit':('s',GOLD,'white')}[kind]
    ax.plot(*[[v] for v in p],marker=marker,ms=4.7,mec=color,mfc=fill,mew=1,ls='none',zorder=10)
    if label:ax.text(*(np.array(p)+np.array(offset)),label,fontsize=7,color=color,zorder=11)


def arrow3(ax,a,b):
    a,b=np.array(a),np.array(b)
    ax.quiver(*a,*(b-a),color=GOLD,linewidth=1.1,arrow_length_ratio=.13,normalize=False)


def draw_helix(fig,rect,circular=False):
    ax=helix_axes(fig,rect);s=[1,0,0];target=[0,1,1];edit=[0,1,0] if circular else [1,0,1]
    point3(ax,s,'source','S',(.04,-.02,.06));point3(ax,target,'target','T',(.02,.02,.05));point3(ax,edit,'edit','E',(.03,0,.07))
    arrow3(ax,s,edit)
    ax.plot(*np.column_stack([edit,target]),color=GRAY,ls=':',lw=.9)
    result_block(fig,rect,x=.66)
    if circular:
        txt(fig,rect,.29,.97,'Circular components match',size=7.2,ha='center')
        txt(fig,rect,.29,.05,r'$P(E)=P(T)=(0,1),\quad F=z$',size=7.5,ha='center')
    else:
        txt(fig,rect,.29,.97,'Scalar value matches',size=7.2,ha='center')
        txt(fig,rect,.29,.05,r'$d=z,\quad F=\frac{2}{\pi}\mathrm{atan2}(y,x)$',size=7.3,ha='center')


def draw_context(fig,rect):
    t=np.linspace(0,1.05,101)
    for left,c in [(.085,1),(.59,2)]:
        ax=rel_ax(fig,rect,[left,.40,.33,.45])
        axes_style(ax,[(-.1,1.25),(-.15,2.38)],xticks=[0,1],yticks=[0,1,2])
        ax.set_xlabel('x',fontsize=7,labelpad=0)
        ax.set_ylabel('y',fontsize=7,labelpad=1)
        ax.plot(t,c*t*t,color=BLUE,lw=1.5)
        arrow(ax,(0,0),(1,1))
        point(ax,(0,0),'source','S',(3,6))
        point(ax,(1,c),'target','T',(5,1))
        if c==1:point(ax,(1,1),'edit',None)
        else:
            point(ax,(1,1),'edit','E',(4,-8));ax.plot([1,1],[1,2],color=GRAY,lw=.8,ls=':')
        txt(fig,rect,left+.16,.95,f'Context c = {c}',size=7.4,ha='center')
        txt(fig,rect,left+.16,.13,f'Probe: 1    Output: {2-c}',color=TEAL if c==1 else ERROR,size=7.2,ha='center')
    txt(fig,rect,.5,.035,r'$\Delta h=(1,1,0),\quad F=x+y-cx^2$',size=7.6,ha='center')


def draw_surface(fig,rect):
    ax=rel_ax(fig,rect,[.005,.06,.65,.94],projection='3d')
    u,v=np.meshgrid(np.linspace(0,1,6),np.linspace(0,1,6))
    ax.plot_wireframe(u,v,u*v,color=BLUE,lw=.55,alpha=.62)
    source=[0,0,0]; a=[1,0,0];b=[0,1,0];edit=[1,1,0];target=[1,1,1]
    arrow3(ax,source,a);arrow3(ax,a,edit)
    ax.plot(*np.column_stack([source,b,edit]),color=GOLD,lw=.9,ls='--')
    ax.plot([1,1],[1,1],[0,1],color=GRAY,ls=':',lw=.9)
    for p,kind,name in [(source,'source','S'),(edit,'edit','E'),(target,'target','T')]:
        point3(ax,p,kind,name,(.03,.01,.055))
    point3(ax,a,'target','A',(.03,0,.06));point3(ax,b,'target','B',(.03,0,.06))
    ax.set_xlim(-.1,1.2);ax.set_ylim(-.1,1.2);ax.set_zlim(-.1,1.3)
    ax.set_box_aspect([1,1,.9]);ax.view_init(elev=20,azim=-48);ax.set_axis_off()
    txt(fig,rect,.34,.96,'Original states: z = uv',size=7.2,ha='center')
    txt(fig,rect,.68,.78,'Both properties match',size=6.9)
    txt(fig,rect,.68,.60,r'$P(E)=(1,1)$',size=8.3,color=TEAL)
    txt(fig,rect,.68,.42,r'$F(E)=0$',size=9,color=ERROR)
    txt(fig,rect,.68,.22,r'$F(T)=1$',size=9,color=TEAL)
    txt(fig,rect,.34,.04,r'$h(u,v)=(u,v,uv),\quad F=z$',size=7.3,ha='center')


def draw_slices(fig,rect):
    for left,v in [(.085,0),(.59,1)]:
        ax=rel_ax(fig,rect,[left,.415,.33,.44])
        axes_style(ax,[(-.12,1.3),(-.18,1.35)])
        u=np.linspace(0,1.12,100);ax.plot(u,v*u,color=BLUE,lw=1.5)
        arrow(ax,(0,0),(1,0))
        point(ax,(0,0),'source','S' if v==0 else 'B',(3,6))
        point(ax,(1,v),'target','A' if v==0 else 'T',(5,2))
        if v==0:point(ax,(1,0),'edit',None)
        else:
            point(ax,(1,0),'edit','E',(5,2));ax.plot([1,1],[0,1],color=GRAY,ls=':',lw=.8)
        ax.set_xlabel('u',fontsize=7,labelpad=1)
        ax.set_ylabel('z',fontsize=7,labelpad=1)
        txt(fig,rect,left+.16,.95,f'Section v = {v}',size=7.4,ha='center')
    txt(fig,rect,.5,.135,r'$h(u,v)=(u,v,uv),\quad F=z$',size=7.3,ha='center')
    txt(fig,rect,.5,.035,r'$P(E)=(1,1),\quad F(E)=0,\quad F(T)=1$',size=7.7,ha='center')


DRAWERS = {
    '01_projection':draw_projection,'02_parabola':draw_parabola,'03_chord':draw_chord,
    '04_level_sets':lambda f,r:draw_parabola(f,r,contours=True),'05_chord_profiles':draw_profiles,
    '06_axial_helix':draw_helix,'07_circular_helix':lambda f,r:draw_helix(f,r,circular=True),
    '08_context':draw_context,'09_surface':draw_surface,'10_surface_slices':draw_slices,
}


















