"""Render the PC-MDE comparison from aggregated distance-sensitive scores."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .figure_style import TWO_COLUMN, apply_style
from .plot_data import load_data

CONDITIONS = ("Clinical", "Symbolic")
COLORS = {"Clinical": "#0072B2", "Symbolic": "#D55E00"}
LABELS = {"Clinical": "Medical semantics", "Symbolic": "No medical semantics"}
MARKERS = {"Clinical": "o", "Symbolic": "s"}
MODEL_LABELS = {
    "gpt_gpt-5.6-sol-ca": "GPT 5.6 Sol",
    "gpt_claude-sonnet-5": "Claude Sonnet 5",
    "vllm_deepseek_v4_flash": "DeepSeek V4 Flash 0731",
    "gpt_qwen3.5-397b-a17b": "Qwen3.5 397B A17B",
    "vllm_Qwen3_6_35B_A3B": "Qwen3.6 35B A3B",
    "vllm_Qwen3_8_27B": "Qwen3.8 27B",
    "vllm_Mistral_4": "Mistral Small 4 119B 2603",
    "vllm_gpt_oss_120b": "GPT OSS 120B",
}
MODEL_ORDER = list(MODEL_LABELS)


def show_axis_lines(ax):
    """Keep the left/bottom coordinate axes visible without a full plot frame."""
    for side in ('left','bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('#333333')
        ax.spines[side].set_linewidth(1.0)


def save(fig,path,extra_bottom=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.08)
    if extra_bottom:
        from matplotlib.transforms import Bbox
        bbox = Bbox.from_extents(bbox.x0, bbox.y0-extra_bottom, bbox.x1, bbox.y1)
    for ext in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+ext),dpi=600 if ext=='png' else None,
                    facecolor='white',edgecolor='none',transparent=False,
                    bbox_inches=bbox)
    plt.close(fig)


def draw_level_means(ax,levels):
    """Draw the same level panel in standalone and composite exports."""
    lt={(r['level'],r['condition']):r for r in levels};means=[]
    for cond,offset in [('Clinical',13),('Symbolic',-13)]:
        data=[lt[i,cond]['mean_sd'] for i in range(1,11)];means.extend(v for v in data if v is not None)
        ax.plot(range(1,11),data,color=COLORS[cond],marker='o',markersize=4.5,linewidth=1.8,label=LABELS[cond])
        for level,value in enumerate(data,1):
            if value is not None:ax.annotate(f'{value:.3f}',(level,value),xytext=(0,offset),textcoords='offset points',ha='center',va='center',fontsize=8.5,color=COLORS[cond],
                                     bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.3))
    low,high=min(means),max(means);pad=max(.015,(high-low)*.3)
    ax.set(xticks=range(1,11),xlim=(.6,10.4),ylim=(max(0,low-pad),min(1,high+pad)),ylabel='Mean distance-aware score $S_d$',xlabel='Difficulty level')
    ax.grid(axis='y',color='#E8EDF2',linewidth=.7);ax.set_axisbelow(True);ax.tick_params(length=0,labelsize=10)
    ax.xaxis.label.set_size(11);ax.yaxis.label.set_size(11)
    show_axis_lines(ax)


def draw_pcmde_distribution(ax,ci_rows):
    """Draw the same model means and intervals in every export."""
    table={(r['model'],r['condition']):r for r in ci_rows}
    ys=np.arange(len(MODEL_ORDER))[::-1]
    for y in ys[:-1]:
        ax.axhline(y-.5,color='#e8edf2',linewidth=.55,zorder=0)
    for model,y in zip(MODEL_ORDER,ys):
        counts=[]
        for condition,offset in [('Clinical',.19),('Symbolic',-.19)]:
            r=table.get((model,condition));counts.append(str(r['n']) if r else '–')
            if not r:continue
            yy=y+offset;color=COLORS[condition]
            ax.plot([r['ci_lower'],r['ci_upper']],[yy,yy],color=color,lw=1.5,solid_capstyle='butt')
            ax.plot([r['ci_lower'],r['ci_upper']],[yy,yy],linestyle='none',marker='|',markersize=5,color=color)
            ax.plot(r['mean_sd'],yy,marker=MARKERS[condition],color=color,markersize=4,
                    markeredgecolor='white',markeredgewidth=.45,zorder=4)
            ax.annotate(f"{r['mean_sd']:.3f}",(r['ci_upper'],yy),xytext=(4,0),
                        textcoords='offset points',ha='left',va='center',fontsize=8,color=color)
        ax.text(1.025,y,' / '.join(counts),transform=ax.get_yaxis_transform(),fontsize=8.5,color='#444444',va='center',clip_on=False)
    ax.set_yticks(ys,[MODEL_LABELS[m] for m in MODEL_ORDER]);ax.set_ylim(-.65,len(MODEL_ORDER)-.28)
    bounds=[x for r in ci_rows for x in (r['ci_lower'],r['ci_upper'])];pad=max(.008,(max(bounds)-min(bounds))*.1)
    ax.set_xlim(max(0,min(bounds)-pad),min(1,max(bounds)+max(pad,(max(bounds)-min(bounds))*.19)))
    ax.set_xlabel('Mean distance-aware score $S_d$')
    ax.grid(axis='x',color='#E8EDF2',lw=.7);ax.set_axisbelow(True)
    ax.tick_params(axis='both',labelsize=10,length=0,pad=5);ax.xaxis.label.set_size(11);ax.text(1.025,1.04,'n: Medical / No medical',transform=ax.transAxes,fontsize=8.5,color='#444444',ha='left')
    show_axis_lines(ax)


def plot_comparison(ci_rows,levels,out):
    """Export both original panels and their shared legend on one white canvas."""
    apply_style()
    bottom_padding=.5
    width,height=2*TWO_COLUMN-1.0,5.4+bottom_padding
    fig=plt.figure(figsize=(width,height))
    # Preserve each standalone panel's physical plot size and center its canvas
    # in the same row; only the positions change in the assembled figure.
    model_ax=fig.add_axes([.31*TWO_COLUMN/width,(bottom_padding+.5+.14*4.8)/height,
                           .49*TWO_COLUMN/width,.79*4.8/height])
    level_ax=fig.add_axes([(TWO_COLUMN+.13*TWO_COLUMN-1.0)/width,(bottom_padding+1+.19*3.8)/height,
                           .84*TWO_COLUMN/width,.78*3.8/height])
    draw_pcmde_distribution(model_ax,ci_rows)
    draw_level_means(level_ax,levels)
    handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in CONDITIONS]
    fig.legend(handles=handles,frameon=False,fontsize=11,loc='lower center',
               bbox_to_anchor=(.5,(bottom_padding+.025*5.4)/height),
               ncol=2,columnspacing=1.5,handlelength=1.6)
    save(fig,out/'disease_diagnosis_sd_comparison',extra_bottom=.45)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="aggregated pc_mde_semantics_sd.json")
    parser.add_argument("--output-dir", type=Path, default=Path("plot"))
    args = parser.parse_args()
    data = load_data("pc_mde_semantics_sd", args.data)
    plot_comparison(data["model_ci"], data["levels"], args.output_dir)


if __name__ == "__main__":
    main()
