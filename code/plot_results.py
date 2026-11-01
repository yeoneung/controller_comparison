"""Regenerate the two comparison figures from the supplied observations."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def draw(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42})
    rows=list(csv.DictReader((ROOT/'recorded_data/sensitivity_examples.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(8,2.9),layout='constrained')
    for ax,system,title in zip(axes[:2],['mixing','transient'],
                               ['(a) Mixing plant','(b) Transient amplification']):
        data=[r for r in rows if r['system']==system and r['nonlinearity']=='tanh']
        x=[int(r['L']) for r in data]
        for column,label,style in [('actual_tangent_difference','Sensitivity error','k-'),
            ('common_defect_bound','Common-defect bound','C0--'),
            ('separate_lipschitz_bound','Separate-Lipschitz bound','C1:')]:
            ax.plot(x,[float(r[column]) for r in data],style,lw=1.6,label=label)
        ax.set_yscale('log');ax.set_xlabel('Retained length L')
        ax.set_title(title,fontsize=9)
        ax.legend(loc='lower left',fontsize=7);ax.grid(alpha=.2)
    axes[0].set_ylabel('Sensitivity difference or bound')

    structural=json.loads((ROOT/'structural_extension/evidence/structural_checks.json').read_text())
    matrix=sorted(structural['matrix_gain_characterization'],key=lambda r:r['gain'])
    ax=axes[2]
    ax.semilogx([r['gain'] for r in matrix],[r['log_gain_constant'] for r in matrix],
        'o-',color='C2',lw=1.6,ms=4,label='Finite-length maximum')
    ax.axhline(np.exp(-1),color='black',lw=1,ls='--',label=r'$e^{-1}$')
    ax.set_xlabel(r'Gain $a$');ax.set_ylabel(r'$a\widehat C(a)$')
    ax.set_title('(c) Near-zero gain scaling',fontsize=9)
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.set_yticks([.3674,.3676,.3678])
    ax.legend(loc='lower left',fontsize=7);ax.grid(alpha=.2)
    fig.savefig(output/'Figure_1.pdf');fig.savefig(output/'Figure_1.png',dpi=170);plt.close(fig)

    records=json.loads((ROOT/'evidence/thermal_application.json').read_text())['records']
    small=[r for r in records if r['increment']==.01]
    fig,axes=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    shown=[r for r in small if r['zones']==6];x=np.arange(len(shown));ax=axes[0]
    ax.bar(x-.18,[r['generic_memory'] for r in shown],.36,label='Conventional bound',color='#b9c5cd')
    ax.bar(x+.18,[r['combined_memory'] for r in shown],.36,label='With common defect',color='#357d9b')
    ax.set_xticks(x,[f"{int(r['wall_capacity']/1e6)} / {r['theta']}" for r in shown])
    ax.set_xlabel('Wall capacity (MJ/K) / nominal gain');ax.set_ylabel('Retained time steps')
    ax.set_title('(a) 10 W changes, 0.001 K bias budget')
    ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2)
    ax=axes[1];x=np.arange(1,len(small)+1)
    for field,label,style in [('median_sampling_time_ratio','Input generation + simulation','o-'),
        ('median_cost_ratio_with_compiled_kernel','Also memory selection','s--'),
        ('median_total_cost_ratio','Also one-time compilation','^:')]:
        ax.plot(x,[r[field] for r in small],style,lw=1.2,ms=4,label=label)
    ax.axhline(1,color='black',lw=.8);ax.axvline(6.5,color='gray',lw=.7,ls='--')
    ax.text(3.5,.793,'6 zones',ha='center',fontsize=8)
    ax.text(9.5,.793,'18 zones',ha='center',fontsize=8)
    ax.set_ylim(.78,1.015);ax.set_xticks([1,3,6,7,9,12]);ax.set_xlabel('Condition')
    ax.set_ylabel('New / conventional cost');ax.set_title('(b) Five paired timing repetitions')
    ax.legend(loc='center right');ax.grid(alpha=.2)
    fig.savefig(output/'Figure_2.pdf');fig.savefig(output/'Figure_2.png',dpi=170);plt.close(fig)
    print(json.dumps({'passed':True,'figures':2,'figure_1_panels':3,
        'matrix_gain_values':len(matrix),
        'source':'Recorded deterministic, matrix gain-scaling, and thermal observations.'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'figures')
    draw(parser.parse_args().out)
