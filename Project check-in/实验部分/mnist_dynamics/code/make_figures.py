"""Plot only measured local dynamics; missing thresholds remain censored."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results')
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=read(a.input/'metrics.csv');runs=read(a.input/'runs.csv');summary=read(a.input/'summary.csv')
    qa=json.loads((a.input/'qa.json').read_text())
    tag='' if qa['complete'] else ' | PARTIAL'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(a.output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    def trace(run_id):
        r=[r for r in rows if r['run_id']==run_id and r.get('test_accuracy','')!='']
        return {key:np.array([float(x[key]) for x in r]) for key in ['step','train_accuracy','test_accuracy','train_mse_element_mean','test_mse_element_mean','global_norm','layer1_norm','layer2_norm','layer3_norm']}
    alphas=sorted(set(float(r['alpha']) for r in runs if r['phase']=='fig3d'))
    colors=plt.cm.viridis(np.linspace(.05,.9,max(1,len(alphas))))
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    legend=[]
    for alpha,color in zip(alphas,colors):
        group=[r for r in runs if r['phase']=='fig3d' and float(r['alpha'])==alpha]
        ts=[trace(r['run_id']) for r in group]
        assert all(np.array_equal(t['step'],ts[0]['step']) for t in ts)
        x=ts[0]['step'];mask=x>0
        for metric,style in [('train_accuracy','--'),('test_accuracy','-')]:
            y=np.array([t[metric] for t in ts])
            mean=y.mean(0);sd=y.std(0,ddof=1) if len(y)>1 else np.zeros_like(mean)
            ax.plot(x[mask],mean[mask],style,color=color,lw=1.6)
            if len(y)>1:ax.fill_between(x[mask],np.clip((mean-sd)[mask],0,1),np.clip((mean+sd)[mask],0,1),color=color,alpha=.08)
        legend.append(Line2D([0],[0],color=color,label=f'alpha={alpha:g} (n={len(ts)})'))
    legend += [Line2D([0],[0],ls='--',color='black',label='Train'),Line2D([0],[0],ls='-',color='black',label='Test')]
    ax.set(xscale='log',xlim=(1,100000),ylim=(-.02,1.02),xlabel='Optimizer updates',ylabel='Accuracy',
           title='Figure 3d reproduction | MNIST N=1,000 | AdamW wd=1e-4'+tag)
    ax.grid(alpha=.18);ax.legend(handles=legend,fontsize=8,ncol=2,loc='lower right')
    fig.text(.5,-.025,'FP64 | mean ± sample SD across available seeds | project initialization grid',ha='center',fontsize=9)
    save(fig,'figure3d')
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,alpha in zip(axes.flat,alphas):
        for run in [r for r in runs if r['phase']=='fig3d' and float(r['alpha'])==alpha]:
            t=trace(run['run_id']);mask=t['step']>0
            ax.plot(t['step'][mask],t['train_accuracy'][mask],'--',color='#1464a0',alpha=.65,lw=.8)
            ax.plot(t['step'][mask],t['test_accuracy'][mask],color='#dc6500',alpha=.65,lw=1)
        ax.set(xscale='log',title=f'alpha={alpha:g}',xlabel='Updates',ylabel='Accuracy',ylim=(-.02,1.02));ax.grid(alpha=.15)
    fig.suptitle('Figure 3d | individual seeds | blue dashed: train; orange: test'+tag)
    save(fig,'figure3d_by_seed')
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    diagnostic_metrics=[('train_mse_element_mean','Train MSE'),('test_mse_element_mean','Test MSE'),
                        ('global_norm','Global norm'),('layer1_norm','Layer 1 norm'),
                        ('layer2_norm','Layer 2 norm'),('layer3_norm','Layer 3 norm')]
    for alpha,color in zip(alphas,colors):
        ts=[trace(r['run_id']) for r in runs if r['phase']=='fig3d' and float(r['alpha'])==alpha]
        for ax,(metric,label) in zip(axes.flat,diagnostic_metrics):
            x=ts[0]['step'];mask=x>0
            y=np.array([t[metric] for t in ts]).mean(0)
            ax.plot(x[mask],y[mask],color=color,lw=1.2)
    for ax,(metric,label) in zip(axes.flat,diagnostic_metrics):
        ax.set(xscale='log',xlim=(1,100000),title=label,xlabel='Optimizer updates',
               ylabel='Element-mean MSE' if 'mse' in metric else 'L2 parameter norm')
        if 'mse' in metric:ax.set_yscale('log')
        ax.grid(alpha=.15)
    fig.suptitle('Figure 3d diagnostics | N=1,000 | wd=1e-4'+tag)
    fig.legend(handles=[Line2D([0],[0],color=c,label=f'alpha={a:g}') for a,c in zip(alphas,colors)],
               ncol=6,loc='upper center',bbox_to_anchor=(.5,-.015),frameon=False)
    fig.text(.5,-.08,'Means across available seeds at measured evaluation steps | norms include weights and biases',ha='center',fontsize=9)
    save(fig,'figure3d_diagnostics')
    fig,ax=plt.subplots(figsize=(9,5.2),constrained_layout=True)
    e=[r for r in runs if r['phase']=='fig3e']
    seedcolors=['#1f77b4','#ee8a24','#279a68']
    for seed in (0,1,2):
        for run in [r for r in e if int(r['seed'])==seed and r.get('scope_role','primary')=='primary']:
            n=int(run['n'])
            if run['event_observed']=='True':
                t=float(run['t60'])
                if t>0:ax.scatter(n,t,color=seedcolors[seed],s=35,alpha=.75,zorder=3)
                else:ax.annotate(f'N={n}: t=0',(n,100),fontsize=7)
            else:ax.scatter(n,100000,marker='^',s=65,facecolors='none',edgecolors=seedcolors[seed],zorder=4)
    supplemental=[r for r in e if r.get('scope_role','primary')!='primary']
    for run in supplemental:
        if run['event_observed']=='True':ax.scatter(int(run['n']),float(run['t60']),marker='D',color='#666666',s=45,zorder=4)
    means=[r for r in summary if r['phase']=='fig3e' and r['mean_t60']!='']
    # Do not bridge N values where the mean is unavailable due to censoring.
    e_summary=sorted([r for r in summary if r['phase']=='fig3e'],key=lambda r:int(r['n']))
    if e_summary:
        ax.plot([int(r['n']) for r in e_summary],[float(r['mean_t60']) if r['mean_t60'] else np.nan for r in e_summary],color='black',lw=1,label='Mean (all 3 seeds reached)')
    if means:
        ax.errorbar([int(r['n']) for r in means],[float(r['mean_t60']) for r in means],
                    yerr=[float(r['sample_sd_t60']) for r in means],fmt='none',ecolor='black',capsize=3,lw=.8,zorder=2)
    for row in e_summary:
        if int(row['censored']):ax.annotate(f"{row['reached']}/{row['planned_seeds']} reached",(int(row['n']),100000),xytext=(10,10),textcoords='offset points',fontsize=8)
    reduced=qa.get('cancelled_by_user',0)>0
    ax.set(yscale='log',ylim=(100,220000),xlabel='Training samples N',ylabel='First observed update with test accuracy > 60%',
           title=('Figure 3e | user-retained sample range\n' if reduced else 'Figure 3e reproduction | ')+ 'alpha=8 (project choice) | wd=5e-3'+tag)
    ax.grid(alpha=.18)
    handles=[Line2D([0],[0],marker='o',ls='',color=c,label=f'Seed {s}') for s,c in enumerate(seedcolors)]
    handles += [Line2D([0],[0],marker='^',mfc='none',ls='',color='black',label='Not reached by 100,000'),Line2D([0],[0],color='black',label='Mean: all 3 reached')]
    if supplemental:handles.append(Line2D([0],[0],marker='D',ls='',color='#666666',label='N=8,000: seed 0 only (supplement)'))
    ax.legend(handles=handles,fontsize=8,loc='lower right')
    footer='N=200-4,000: 3 seeds each | N=8,000: seed 0 only | Whiskers: sample SD' if reduced else 'Whiskers: sample SD | counts in results/summary.csv | censored runs excluded from ordinary means'
    fig.text(.5,-.025,footer,ha='center',fontsize=9)
    save(fig,'figure3e')
    reference='fig3d_n1000_a8_s0'
    if any(r['run_id']==reference for r in runs):
        refrow=next(r for r in runs if r['run_id']==reference)
        t=trace(reference);mask=t['step']>0
        norm=np.load(ROOT/'runs'/reference/'norms.npy',mmap_mode='r');steps=np.arange(len(norm))
        fig,axes=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
        for metric,color in [('train_accuracy','#1464a0'),('test_accuracy','#dc6500')]:axes[0].plot(t['step'][mask],t[metric][mask],color=color,label=metric.split('_')[0])
        for metric,color in [('train_mse_element_mean','#1464a0'),('test_mse_element_mean','#dc6500')]:axes[1].plot(t['step'][mask],t[metric][mask],color=color,label=metric.split('_')[0])
        norm_indices=np.unique(np.r_[np.arange(1,len(norm),100),len(norm)-1])
        for i,name in enumerate(['Global','Layer 1','Layer 2','Layer 3']):axes[2].plot(steps[norm_indices],norm[norm_indices,i],label=name,lw=1)
        for ax in axes:
            ax.set_xscale('log');ax.set_xlabel('Optimizer updates');ax.grid(alpha=.15)
            if refrow['fit_step']:ax.axvline(float(refrow['fit_step']),color='#777777',ls=':',lw=.8)
            if refrow['t60']:ax.axvline(float(refrow['t60']),color='#4b8541',ls=':',lw=.8)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('Accuracy');axes[1].set(yscale='log',ylabel='Element-mean MSE');axes[2].set_ylabel('L2 parameter norm')
        fig.suptitle('Reference | N=1,000, alpha=8, wd=1e-4, seed=0'+tag)
        fig.text(.5,-.025,f"Vertical guides: training fit = {refrow['fit_step']} (gray); first test > 60% = {refrow['t60']} (green)",ha='center',fontsize=9)
        save(fig,'reference_diagnostics')
    print('Figures written:',a.output)


if __name__=='__main__':main()
