from pathlib import Path
import sys,json
from lavu.runtime import context, load_json
H, BUNDLE = context()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=load_json(BUNDLE)['results']
out=H/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(out/(name+'.'+ext),dpi=600)
    plt.close(fig)
def rows(key,mode):return d[key]['splits'][mode]['rows']
def wer(key,mode):
    r=rows(key,mode);return 100*sum(x['errors'] for x in r)/sum(x['words'] for x in r)
labels=['Baseline','No alignment','Median/MAD','Repair + 5-tap','5-tap filter','Earlier pipeline']
variants=[None,'f_nosync_g','f_norm_mad_g','f_clean_ours_g','f_smooth_5tap_g','ours_full_g']
fig,axes=plt.subplots(2,1,figsize=(3.35,2.958),sharex=True)
for ax,model,title in zip(axes,['GMM+fMLLR','DNN+fMLLR'],['(a) GMM-SAT + fMLLR','(b) DNN + fMLLR']):
    a=np.array([[wer(f'factor_{v}_{model}_s1' if v else f'shared_32_{model}',m) for m in ['test_modal','test_silent']] for v in variants]);y=np.arange(len(labels))
    ax.scatter(a[:,0],y-.1,s=14,color='#303030',label='Voiced',zorder=3);ax.scatter(a[:,1],y+.1,s=14,color='#d55e00',label='Silent',zorder=3)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,65);ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_title(title,loc='left',fontweight='bold');ax.tick_params(axis='y',length=0);ax.axhspan(-.45,.45,color='#f1f1f1',zorder=0)
axes[1].set_xlabel('WER (%)');fig.subplots_adjust(left=.33,right=.98,hspace=.43,top=.94,bottom=.14);save(fig,'factors')
def per_speaker(model,mode):
    v={}
    for fold in range(9):
        for r in rows(f'disjoint_32_{model}_fold{fold}',mode):
            a=v.setdefault(r['utt'].split('_')[0],[0,0]);a[0]+=r['errors'];a[1]+=r['words']
    return {s:100*e/n for s,(e,n) in v.items()}
fig,axes=plt.subplots(1,2,figsize=(3.5,1.85),sharex=True,sharey=True)
for ax,mode,title,color in zip(axes,['test_modal','test_silent'],['(a) Voiced','(b) Silent'],['#303030','#d55e00']):
    x=per_speaker('DNN_transform_withheld',mode);y=per_speaker('DNN+fMLLR',mode);s=sorted(x);ax.scatter([x[k] for k in s],[y[k] for k in s],s=7,color=color,alpha=.85);ax.plot([0,100],[0,100],'--',lw=.7,color='#888');ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_xticks([0,50,100]);ax.set_yticks([0,50,100]);ax.set_aspect('equal');ax.set_title(title,loc='left');ax.set_xlabel('Transform withheld (%)')
axes[0].set_ylabel('fMLLR supplied (%)');fig.subplots_adjust(left=.15,right=.98,bottom=.30,top=.82,wspace=.18);save(fig,'adaptation')
fig,axes=plt.subplots(1,2,figsize=(6.5,1.581575),sharey=True)
for ax,mode,title in zip(axes,['test_modal','test_silent'],['(a) Voiced','(b) Silent']):
    allrows=[]
    for units,color,shift in [(32,'#0072b2',-.12),(40,'#888888',.12)]:
        values=[wer(f'disjoint_{units}_DNN+fMLLR_fold{i}',mode) for i in range(9)];ax.plot(np.arange(1,10)+shift,values,'o',ms=3,color=color,label=str(units));rr=[r for i in range(9) for r in rows(f'disjoint_{units}_DNN+fMLLR_fold{i}',mode)];pooled=100*sum(r['errors'] for r in rr)/sum(r['words'] for r in rr);ax.axhline(pooled,ls='--',lw=.8,color=color);allrows.append(values)
    for i in range(9):ax.plot([i+1-.12,i+1+.12],[allrows[0][i],allrows[1][i]],color='#bcbcbc',lw=.8,zorder=0)
    ax.set_title(title,loc='left');ax.set_xlabel('Fold');ax.set_xticks([1,3,5,7,9]);ax.set_ylim(0,40);ax.grid(axis='y',color='#ddd',lw=.5)
axes[0].set_ylabel('WER (%)');axes[1].legend(frameon=False,loc='lower right');fig.subplots_adjust(left=.075,right=.99,bottom=.27,top=.84,wspace=.18);save(fig,'folds')
print('Corrected figures generated')
