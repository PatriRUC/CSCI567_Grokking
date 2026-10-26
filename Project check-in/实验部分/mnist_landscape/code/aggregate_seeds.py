"""CPU aggregation and plots from portable CSV data; no torch or server paths."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACKAGE=Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def edges(values):
    logs=np.log10(values); middle=(logs[:-1]+logs[1:])/2
    return 10**np.r_[logs[0]-(middle[0]-logs[0]),middle,logs[-1]+logs[-1]-middle[-1]]

def aggregate(root,input_csv,configs_dir):
    root.mkdir(parents=True,exist_ok=True)
    configs=[json.loads((configs_dir/('seed%d.json'%s)).read_text(encoding='utf-8')) for s in (0,1,2)]
    invariant=['official_commit','original_data_sizes','all_data_sizes','alphas','steps','batch_size','lr','weight_decay','dtype','optimizer','foreach','tf32','evaluation_every_steps','sampled_evaluation_size_max','method','checkpoint_selection']
    for key in invariant: assert all(c[key]==configs[0][key] for c in configs), key
    assert [c['seed'] for c in configs]==[0,1,2]
    ns=np.array(configs[0]['all_data_sizes']); alphas=np.array(configs[0]['alphas'])
    orig_ns=np.array(configs[0]['original_data_sizes'])
    orig_indices=[int(np.where(ns==n)[0][0]) for n in orig_ns]
    all_rows=read_csv(input_csv)
    assert len(all_rows)==1890
    rows_by_seed=[[r for r in all_rows if int(r['seed'])==s] for s in (0,1,2)]
    expected={(int(n),i) for n in ns for i in range(len(alphas))}
    lookups=[]
    for seed,rows in enumerate(rows_by_seed):
        assert len(rows)==630 and set(int(r['seed']) for r in rows)=={seed}
        lookup={(int(r['n']),int(r['alpha_index'])):r for r in rows}
        assert set(lookup)==expected
        for (n,i),r in lookup.items():
            assert np.isclose(float(r['alpha']),alphas[i],rtol=0,atol=1e-12)
            assert (r['original_grid']=='True') == (n in set(orig_ns))
        assert all(np.isfinite(float(v)) for r in rows for k,v in r.items() if k!='original_grid')
        assert max(float(r['max_norm_relative_error']) for r in rows)<2e-6
        lookups.append(lookup)
    metrics=['paper_best_train_accuracy','paper_best_test_accuracy','paper_min_train_loss','paper_min_test_loss',
             'selected_full_train_accuracy','selected_full_test_accuracy','selected_full_train_loss','selected_full_test_loss',
             'final_full_train_accuracy','final_full_test_accuracy','final_full_train_loss','final_full_test_loss']
    arrays={field:np.array([[[float(lookup[(int(n),i)][field]) for i in range(len(alphas))] for n in ns] for lookup in lookups]) for field in metrics}
    assert all(np.isfinite(v).all() for v in arrays.values())
    combined=[r for rows in rows_by_seed for r in rows]
    write_csv(root/'all_seeds_results.csv',combined)
    summary=[]
    for n_index,n in enumerate(ns):
        for i,alpha in enumerate(alphas):
            row=dict(n=int(n),alpha_index=i,alpha=float(alpha),seed_count=3)
            for metric in metrics:
                values=arrays[metric][:,n_index,i]
                row[metric+'_mean']=float(values.mean()); row[metric+'_std']=float(values.std(ddof=1))
            for variant in ('paper','selected_full'):
                train_key='paper_best_train_accuracy' if variant=='paper' else 'selected_full_train_accuracy'
                test_key='paper_best_test_accuracy' if variant=='paper' else 'selected_full_test_accuracy'
                gaps=arrays[train_key][:,n_index,i]-arrays[test_key][:,n_index,i]
                row[variant+'_generalization_gap_mean']=float(gaps.mean()); row[variant+'_generalization_gap_std']=float(gaps.std(ddof=1))
            summary.append(row)
    write_csv(root/'summary_by_condition.csv',summary)
    np.savez_compressed(root/'seeds012_grid.npz',seeds=np.array([0,1,2]),ns=ns,alphas=alphas,original_ns=orig_ns,**arrays)
    for n in (200,8000): write_csv(root/('slice_N%d_mean_std.csv'%n),[r for r in summary if r['n']==n])

    contrasts=[]
    checks={}
    def error(seed,n,i,variant,split='test'):
        key=('paper_best_'+split+'_accuracy') if variant=='paper' else ('selected_full_'+split+'_accuracy')
        return 1-float(lookups[seed][(n,i)][key])
    for variant in ('paper','selected_full'):
        values_by_name={name:[] for name in ('smallN_test_error_increase_alpha1_to10','largeN_test_error_increase_alpha1_to10','test_error_reduction_N200_to8000_at_alpha10','smallN_minus_largeN_alpha_effect')}
        for seed in (0,1,2):
            small=error(seed,200,20,variant)-error(seed,200,10,variant)
            large=error(seed,8000,20,variant)-error(seed,8000,10,variant)
            reduction=error(seed,200,20,variant)-error(seed,8000,20,variant)
            values=[small,large,reduction,small-large]
            for (name,items),value in zip(values_by_name.items(),values):
                items.append(value)
                contrasts.append(dict(variant=variant,contrast=name,seed=seed,difference=value,difference_percentage_points=100*value))
        checks[variant]={}
        for name,values in values_by_name.items():
            a=np.array(values)
            checks[variant][name]=dict(per_seed=a.tolist(),mean=float(a.mean()),std=float(a.std(ddof=1)),all_positive=bool((a>0).all()))
        checks[variant]['smallN_both_sides_worse_than_alpha1']=all(error(s,200,0,variant)>error(s,200,10,variant) and error(s,200,20,variant)>error(s,200,10,variant) for s in (0,1,2))
    write_csv(root/'paired_contrasts_by_seed.csv',contrasts)

    figdir=root/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def combined_figure(variant):
        train_key='paper_best_train_accuracy' if variant=='paper' else 'selected_full_train_accuracy'
        test_key='paper_best_test_accuracy' if variant=='paper' else 'selected_full_test_accuracy'
        train_err=1-arrays[train_key]; test_err=1-arrays[test_key]
        fig=plt.figure(figsize=(16,6.2),constrained_layout=True)
        spec=fig.add_gridspec(2,3,width_ratios=[1.1,1.1,1])
        for col,(err,title) in enumerate([(train_err,'(a) Mean training error'),(test_err,'(b) Mean test error')]):
            ax=fig.add_subplot(spec[:,col]); z=err.mean(0)[orig_indices]
            mesh=ax.pcolormesh(edges(alphas),edges(orig_ns),z,cmap='Reds',vmin=0,vmax=1,shading='flat',rasterized=True)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.1,10); ax.set_ylim(60000,10)
            ax.set_xticks([.1,1,10]); ax.set_xticklabels(['0.1','1','10'])
            ax.set_yticks([10,100,1000,10000,60000]); ax.set_yticklabels(['10','100','1,000','10,000','60,000'])
            ax.set_xlabel('Relative weight norm alpha'); ax.set_ylabel('Training samples N (increases downward)'); ax.set_title(title)
            levels=[v for v in (.1,.2,.3,.4) if z.min()<v<z.max()]
            if levels:
                contour=ax.contour(alphas,orig_ns,z,levels=levels,colors='#716500',linewidths=.7,linestyles='dashed'); ax.clabel(contour,fontsize=7,fmt='%.1f')
            fig.colorbar(mesh,ax=ax,label='Classification error',fraction=.046,pad=.04)
        for row,n in enumerate((200,8000)):
            ax=fig.add_subplot(spec[row,2]); idx=int(np.where(ns==n)[0][0])
            for err,label,color in [(train_err,'Train','#1f77b4'),(test_err,'Test','#ff7f0e')]:
                means=err[:,idx,:].mean(0); stds=err[:,idx,:].std(0,ddof=1)
                ax.plot(alphas,means,'-o',color=color,markersize=3,label=label)
                ax.fill_between(alphas,np.clip(means-stds,0,1),np.clip(means+stds,0,1),color=color,alpha=.18)
            ax.set_xscale('log'); ax.set_xlim(.1,10); ax.set_ylim(-.02,1.02); ax.grid(alpha=.2)
            ax.set_xticks([.1,1,10]); ax.set_xticklabels(['0.1','1','10']); ax.legend(loc='best')
            ax.set_title(('(c) ' if row==0 else '')+'N = {:,}; mean +/- sample SD'.format(n)); ax.set_ylabel('Classification error')
            if row==1: ax.set_xlabel('Relative weight norm alpha')
        label='Independent best accuracies over scheduled evaluations' if variant=='paper' else 'Training-loss-selected checkpoints; full-split evaluation'
        fig.suptitle('MNIST fixed-norm landscape | seeds 0, 1, 2\n'+label,fontsize=12)
        for ext in ('png','pdf'): fig.savefig(figdir/('figure3abc_seeds012_'+variant+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
        return train_err,test_err
    train_err,test_err=combined_figure('paper')
    combined_figure('selected_full')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for s,ax in enumerate(axes):
        mesh=ax.pcolormesh(edges(alphas),edges(orig_ns),test_err[s,orig_indices,:],cmap='Reds',vmin=0,vmax=1,shading='flat',rasterized=True)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.1,10); ax.set_ylim(60000,10); ax.set_title('Seed %d'%s)
        ax.set_xlabel('Relative weight norm alpha'); ax.set_ylabel('Training samples N')
    fig.colorbar(mesh,ax=axes,label='Test classification error',fraction=.025,pad=.02)
    for ext in ('png','pdf'): fig.savefig(figdir/('test_landscapes_by_seed.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    z=test_err.std(0,ddof=1)[orig_indices]
    mesh=ax.pcolormesh(edges(alphas),edges(orig_ns),z,cmap='viridis',shading='flat',rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.1,10); ax.set_ylim(60000,10)
    ax.set_xlabel('Relative weight norm alpha'); ax.set_ylabel('Training samples N'); ax.set_title('Test error sample SD | seeds 0, 1, 2')
    fig.colorbar(mesh,ax=ax,label='Sample standard deviation')
    for ext in ('png','pdf'): fig.savefig(figdir/('test_error_std.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    max_idx=np.unravel_index(np.argmax(test_err.std(0,ddof=1)),test_err.shape[1:])
    qa=dict(seeds=[0,1,2],conditions_per_seed=630,total_result_rows=len(combined),same_config_except_seed=True,
            max_norm_relative_error=max(float(r['max_norm_relative_error']) for r in combined),
            all_numeric_fields_finite=True,standard_deviation_definition='sample SD, ddof=1, across seeds; not confidence interval',
            maximum_test_error_std=dict(n=int(ns[max_idx[0]]),alpha=float(alphas[max_idx[1]]),std=float(test_err.std(0,ddof=1)[max_idx])),contrasts=checks)
    (root/'aggregate_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('Verified 1,890 rows / 630 conditions / seeds 0,1,2; output:',root)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=PACKAGE/'results/all_seeds_results.csv')
    parser.add_argument('--configs-dir',type=Path,default=PACKAGE/'configs')
    parser.add_argument('--output',type=Path,default=PACKAGE/'generated/summary')
    args=parser.parse_args()
    aggregate(args.output,args.input,args.configs_dir)
