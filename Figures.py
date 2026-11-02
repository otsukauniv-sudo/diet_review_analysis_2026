"""Figures 1-6 for the journal and figures 1-4 for the conference paper."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


def _save(fig, directory, stem):
    fig.savefig(directory/(stem+'.png'), dpi=300)
    fig.savefig(directory/(stem+'.svg'))
    plt.close(fig)


def _upper(values, baseline, padding):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    return max(baseline,float(values.max())+padding) if len(values) else baseline


def _lower(values, baseline, padding):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    return min(baseline,float(values.min())-padding) if len(values) else baseline


def _japanese_font(explicit=None):
    families=['Noto Serif CJK JP','Noto Serif JP','Noto Sans CJK JP','Meiryo','MS Mincho','Yu Mincho','MS Gothic','Yu Gothic','Hiragino Mincho ProN','IPAexMincho']
    candidates=[]
    if explicit:
        font_manager.fontManager.addfont(str(explicit))
        candidates.append(str(explicit))
    candidates.extend(font.fname for family in families for font in font_manager.fontManager.ttflist if font.name==family)
    required=set(map(ord,'口コミ減量体重評価解約困難副作用腹痛症状発生価格効果否定'))
    for file in candidates:
        if required.issubset(FT2Font(file).get_charmap()):
            return font_manager.FontProperties(fname=file)
    raise RuntimeError('Japanese figures require an installed Japanese font, such as Meiryo or Noto Serif CJK JP.')


def _inputs(run_dir):
    dest=run_dir/'summary_tables'
    g=pd.read_csv(dest/'group_counts_and_weight_change.csv').set_index('group')
    f=pd.read_csv(dest/'group_text_expression_comparison.csv')
    for name,position in [('first_group',0),('second_group',1)]:
        f[name]=f.comparison.map(lambda x: {'B_vs_A':('B','A'),'C_vs_D':('C','D')}[x][position])
    diagnostics=dest/'weight_change_breakdown.csv'
    d=pd.read_csv(diagnostics)
    file=run_dir/'regression_results/low_rating_associations_5_percent_loss.csv'
    names=['cancellation_mention','x_cancel_difficulty_expression','symptom_mention','x_symptom_experience_expression','x_price_negative_expression','x_effect_positive_expression','x_effect_negative_expression']
    m=pd.read_csv(file).set_index('term').reindex(names) if file.exists() else pd.DataFrame(np.nan,index=names,columns=['coefficient','ci_low','ci_high'])
    audit=json.loads((run_dir/'data_checks/input_data_summary.json').read_text(encoding='utf-8'))
    flow=pd.read_csv(dest/'weight_rating_sample_selection.csv').set_index('stage').n_remaining
    status=json.loads((run_dir/'regression_results/regression_details.json').read_text(encoding='utf-8'))['N1_A_success_low_FE']
    status_file=run_dir/'run_status.json'
    config=json.loads(status_file.read_text(encoding='utf-8')).get('config',{}) if status_file.exists() else {}
    return dict(g=g,f=f,d=d,m=m,config=config,input_n=int(audit['input_review_rows']),
        target_n=int(audit['target_category_rows']),rated_n=int(flow['valid_star_rating']),
        parsed_n=int(flow['both_weights_parsed']),eligible_n=int(flow['finite_usable_weight_and_rating']),
        model_input_n=int(status.get('n_input_reviews',0)),model_n=int(status.get('n_reviews',0)),model_products=int(status.get('n_products',0)))


def journal_figures(run_dir, data, font, examples):
    OUT=Path(run_dir)/'figures/en';OUT.mkdir(parents=True,exist_ok=True)
    g,f,d,m=(data[key] for key in ['g','f','d','m'])
    n_success=int(g.loc[['A','B','neutral_success'],'n_reviews'].sum())
    n_below=int(g.loc[['C','D','neutral_non_success'],'n_reviews'].sum())
    b_rate=100*int(g.loc['B','n_reviews'])/n_success if n_success else np.nan
    c_rate=100*int(g.loc['C','n_reviews'])/n_below if n_below else np.nan
    plt.rcParams.update({'font.family':['STIXGeneral',font.get_name()], 'font.size':11.5,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10.5,'ytick.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','figure.facecolor':'white','savefig.facecolor':'white'})
    JP=font.copy(); JP.set_size(11)
    NAMES=[('cancellation_mention','Cancellation mention'),('x_cancel_difficulty_expression','Cancellation difficulty'),('symptom_mention','Symptom mention'),('x_symptom_experience_expression','Symptom occurrence'),('x_price_negative_expression','Price dissatisfaction'),('x_effect_positive_expression','Positive effect wording'),('x_effect_negative_expression','Negative effect wording')]
    def save(fig,name):
     _save(fig,OUT,name)
    def box(ax,x,y,w,h,text,shade='white',size=11.5,weight='normal'):
     p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',linewidth=.9,edgecolor='.15',facecolor=shade)
     ax.add_patch(p);ax.text(x,y,text,ha='center',va='center',fontsize=size,weight=weight,linespacing=1.25)
     return p
    def arrow(ax,a,b):
     ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=.9,color='.2',connectionstyle='arc3'))
    fig,ax=plt.subplots(figsize=(7,4.4));fig.subplots_adjust(left=.025,right=.975,bottom=.02,top=.98);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.5,.935,.75,.09,f"Review and product files: {data['input_n']:,} reviews",shade='.95')
    box(ax,.5,.80,.75,.10,f"Match product IDs and select four categories\n{data['target_n']:,} reviews",shade='.95')
    arrow(ax,(.5,.88),(.5,.86))
    box(ax,.245,.59,.44,.21,f"Structured fields\nValid ratings: {data['rated_n']:,}\nBoth weights parsed: {data['parsed_n']:,}\nUsable weights and rating: {data['eligible_n']:,}",size=11)
    box(ax,.755,.59,.44,.21,'Japanese review text\nNormalize characters and HTML\nApply explicit wording rules\nCreate binary text indicators',size=11)
    arrow(ax,(.40,.74),(.245,.705));arrow(ax,(.60,.74),(.755,.705))
    box(ax,.245,.355,.44,.13,'Weight change × star rating\nA, B, C, D and two neutral groups',size=11)
    box(ax,.755,.355,.44,.13,'Seven indicators for regression\nAdditional descriptive indicators',size=11)
    arrow(ax,(.245,.477),(.245,.43));arrow(ax,(.755,.477),(.755,.43))
    box(ax,.5,.125,.80,.19,f"Combine both measurements\nGroup frequencies and feature comparisons\nMain regression: {data['model_input_n']:,} reviews with loss ≥5%\n{data['model_n']:,} reviews in {data['model_products']:,} products after singleton removal",shade='.94',size=11)
    arrow(ax,(.245,.28),(.37,.23));arrow(ax,(.755,.28),(.63,.23))
    save(fig,'figure1_analysis_flow')
    fig,ax=plt.subplots(figsize=(7,4.4));fig.subplots_adjust(left=.025,right=.975,bottom=.015,top=.985);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.16,.93,.28,.105,'Normalize text\nCharacters and HTML',size=10.5)
    box(ax,.50,.93,.28,.105,'Apply phrase rules\nMentions and specific flags',size=10.5)
    box(ax,.84,.93,.28,.105,'Record 0 or 1\nSeveral flags may equal 1',size=10.5)
    arrow(ax,(.31,.93),(.35,.93));arrow(ax,(.65,.93),(.69,.93))
    ax.text(.5,.842,'Specific wording rules check local negation and hypothetical context.',ha='center',va='center',fontsize=10.8)
    xs=[.025,.20,.72,.98]; ytop=.755;rowh=.132
    headers=['Domain','Invented Japanese sentence','Mention / specific']
    for j in range(3):
     ax.add_patch(Rectangle((xs[j],ytop),xs[j+1]-xs[j],.065,facecolor='.92',edgecolor='.65',lw=.7))
     ax.text((xs[j]+xs[j+1])/2,ytop+.0325,headers[j],ha='center',va='center',fontsize=10.6,weight='bold')
    
    for i,(domain,jp,en,value) in enumerate(examples):
     y=ytop-(i+1)*rowh
     for j in range(3):ax.add_patch(Rectangle((xs[j],y),xs[j+1]-xs[j],rowh,facecolor='white',edgecolor='.7',lw=.6))
     ax.text((xs[0]+xs[1])/2,y+rowh/2,domain,ha='center',va='center',fontsize=10.8)
     ax.text(xs[1]+.018,y+rowh*.65,jp,fontproperties=JP,ha='left',va='center')
     ax.text(xs[1]+.018,y+rowh*.29,en,fontsize=10.7,ha='left',va='center')
     ax.text((xs[2]+xs[3])/2,y+rowh/2,value,ha='center',va='center',fontsize=12.8)
    ax.text(.025,.185,'Specific = cancellation difficulty or symptom occurrence, respectively.',fontsize=10.8,ha='left')
    ax.text(.025,.12,'Seven regression indicators',fontsize=11.5,weight='bold')
    ax.text(.025,.078,'Cancellation mention / difficulty; symptom mention / occurrence;',fontsize=10.8)
    ax.text(.025,.035,'price dissatisfaction; positive effect wording; negative effect wording.',fontsize=10.8)
    save(fig,'figure2_text_classification_rules')
    fig,ax=plt.subplots(figsize=(7,3.55));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.98);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    cols=[('Low rating','1–2 stars'),('Neutral rating','3 stars'),('High rating','4–5 stars')]
    left=.28;right=.98;cw=(right-left)/3
    for j,(s,t) in enumerate(cols):ax.text(left+(j+.5)*cw,.925,s+'\n'+t,ha='center',va='center',fontsize=11.5,linespacing=1.35)
    rows=[(f'Loss ≥5%\nn = {n_success:,}' ,['B','neutral_success','A']),(f'Loss <5%\nn = {n_below:,}' ,['D','neutral_non_success','C'])]
    for i,(label,codes) in enumerate(rows):
     y=.73-i*.30
     ax.text(.01,y,label,ha='left',va='center',fontsize=12,linespacing=1.45)
     total=sum(int(g.loc[k,'n_reviews']) for k in codes)
     for j,k in enumerate(codes):
      highlight=k in ['B','C'];x=left+j*cw
      ax.add_patch(Rectangle((x,y-.13),cw,.26,facecolor='.90' if highlight else 'white',edgecolor='.25',lw=1.3 if highlight else .7))
      label2=k if len(k)==1 else 'Neutral'
      ax.text(x+cw/2,y+.065,label2,ha='center',va='center',fontsize=12,weight='bold')
      count=int(g.loc[k,'n_reviews'])
      ax.text(x+cw/2,y-.01,f'{count:,}',ha='center',va='center',fontsize=14)
      ax.text(x+cw/2,y-.080,(f'{count/total*100:.2f}%' if total else 'NA'),ha='center',va='center',fontsize=11)
    ax.text(.01,.135,f"B rate: {int(g.loc['B','n_reviews']):,} / {n_success:,} = {b_rate:.2f}%",ha='left',va='center',fontsize=11.6)
    ax.text(.52,.135,f"C rate: {int(g.loc['C','n_reviews']):,} / {n_below:,} = {c_rate:.2f}%",ha='left',va='center',fontsize=11.6)
    ax.text(.5,.035,'Each row includes all ratings in its weight-loss category.',ha='center',fontsize=10.8)
    save(fig,'figure3_weight_loss_rating_groups')
    fig=plt.figure(figsize=(7,3.8));gs=fig.add_gridspec(1,2,width_ratios=[.85,1.4]);a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1]);fig.subplots_adjust(left=.11,right=.98,bottom=.28,top=.83,wspace=.50)
    vals=[g.loc['A','mean_loss_percent'],g.loc['B','mean_loss_percent']]
    a.barh([1,0],vals,color=['white','.65'],edgecolor='black',height=.48)
    a.set_yticks([1,0],['A: high','B: low']);a.set_xlim(0,_upper(vals,13.5,2));a.set_xticks([0,5,10]);a.set_xlabel('Mean reported loss (%)');a.set_title('(a) Reviews with loss ≥5%',pad=17)
    for y,v in zip([1,0],vals):a.text(v+.25,y,f'{v:.2f}',va='center',fontsize=11)
    a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0)
    colors=['.2','.72','white'];hatches=['','','///'];labels=['Positive loss <5%','No change','Gain']
    for y,code in zip([1,0],['C','D']):
     vals2=[int(d[(d.original_group==code)&(d.weight_change==k)].iloc[0]['n']) for k in ['positive_loss_below_5pct','unchanged','gain']]
     p=np.array(vals2)/sum(vals2)*100 if sum(vals2) else np.full(3,np.nan);start=0
     for j,v in enumerate(p):
      b.barh(y,v,left=start,height=.48,color=colors[j],edgecolor='black',hatch=hatches[j],lw=.65,label=labels[j] if code=='C' else None)
      if v>=5:b.text(start+v/2,y,f'{v:.1f}%',ha='center',va='center',fontsize=10.8,color='white' if j==0 else 'black',bbox=dict(facecolor=colors[j],edgecolor='none',pad=.4))
      else:b.text(100,y+.34,f'{v:.1f}%',ha='right',va='center',fontsize=10.5)
      start+=v
    b.set_yticks([1,0],['C: high','D: low']);b.set_xlim(0,100);b.set_xticks([0,50,100]);b.set_xlabel('Share within group (%)');b.set_title('(b) Reviews with loss <5%',pad=17);b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False)
    handles,labs=b.get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,frameon=False,fontsize=10.7,handlelength=1.3,columnspacing=1.0)
    save(fig,'figure4_weight_change_comparison')
    fig,axes=plt.subplots(1,2,figsize=(7,4.0),sharey=True);fig.subplots_adjust(left=.31,right=.98,bottom=.29,top=.83,wspace=.28)
    y=np.arange(len(NAMES))[::-1]
    rows=[]
    for ax,comparison,first,second,title in zip(axes,['B_vs_A','C_vs_D'],['B','C'],['A','D'],['(a) Loss ≥5%','(b) Loss <5%']):
     firstv=[];secondv=[]
     for k,label in NAMES:
      z=f[(f.comparison==comparison)&(f.feature==k)].iloc[0]
      v1=z.n_first_flagged/z.n_first_with_text*100 if z.n_first_with_text else np.nan;v2=z.n_second_flagged/z.n_second_with_text*100 if z.n_second_with_text else np.nan
      firstv.append(v1);secondv.append(v2);rows.append({'comparison':comparison,'indicator':k,'first_group':first,'second_group':second,'first_pct':v1,'second_pct':v2})
     for yy,v1,v2 in zip(y,firstv,secondv):ax.plot([v1,v2],[yy,yy],color='.6',lw=.85,zorder=1)
     ax.scatter(firstv,y,c='black',s=33,label=first+': '+('low rating' if first=='B' else 'high rating'),zorder=3)
     ax.scatter(secondv,y,facecolors='white',edgecolors='black',s=33,label=second+': '+('high rating' if second=='A' else 'low rating'),zorder=3)
     limit=_upper(firstv+secondv,40,2);ax.set_xlim(-1,limit);ax.set_xticks([0,20,40] if limit==40 else [0,limit/2,limit]);ax.grid(axis='x',color='.9',lw=.65);ax.set_axisbelow(True);ax.set_title(title,pad=15);ax.set_xlabel('Matched expression (%)')
     ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.40),frameon=False,fontsize=10,handletextpad=.35,labelspacing=.2)
    axes[0].set_yticks(y,[label for k,label in NAMES]);axes[0].set_ylim(-.7,6.7)
    
    save(fig,'figure5_text_expression_comparison')
    fig,ax=plt.subplots(figsize=(7,4.0));fig.subplots_adjust(left=.31,right=.98,bottom=.20,top=.94)
    y=np.arange(7)[::-1];x=[];lo=[];hi=[]
    for k,label in NAMES:
     z=m.loc[k];x.append(z.coefficient*100);lo.append(z.ci_low*100);hi.append(z.ci_high*100)
    ax.errorbar(x,y,xerr=[np.array(x)-lo,np.array(hi)-x],fmt='o',color='black',ms=4.5,lw=1.2,capsize=3)
    ax.axvline(0,color='.4',lw=.9,ls='--');ax.set_yticks(y,[label for k,label in NAMES]);ax.set_xlim(min(-16,_lower(lo,-16,2)),_upper(hi,39,2));ax.set_xticks([-10,0,10,20,30]);ax.set_xlabel('Conditional difference in low-rating probability\n(percentage points; coefficient and 95% confidence interval)');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x',color='.92',lw=.6);ax.set_axisbelow(True)
    ax.set_ylim(-.65,6.65)
    for yy,value in zip(y,x):
     if not np.isfinite(value): ax.text(.98,yy,'Not identified',ha='right',va='center',transform=ax.get_yaxis_transform(),fontsize=9)
    save(fig,'figure6_regression_coefficients')


def nlp_figures(run_dir, data, font, examples):
    out = Path(run_dir) / 'figures' / 'ja'
    out.mkdir(parents=True, exist_ok=True)
    g, f, d = data['g'], data['f'], data['d']
    def number(value):
        return f'{int(value):,}' if pd.notna(value) else 'N/A'
    def rect(ax, x, y, w, h, text, face='white', fs=9.5):
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor=face, edgecolor='.25', lw=.6))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, linespacing=1.10)
    def arrow(ax, start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7, lw=.65, color='.2'))
    settings = {'font.family': font.get_name(), 'font.size': 9.5, 'axes.labelsize': 9.5,
        'axes.titlesize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9.5,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'path',
        'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(settings):
        fig, ax = plt.subplots(figsize=(3.2, 2.3))
        fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.015)
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        rect(ax, .5, .94, .95, .10, f"口コミ {number(data['input_n'])}件", face='.95')
        rect(ax, .5, .785, .95, .10, f"対象4カテゴリ {number(data['target_n'])}件", face='.95')
        arrow(ax, (.5, .885), (.5, .84))
        rect(ax, .245, .565, .46, .18, f"体重・星評価\n有効{number(data['eligible_n'])}件")
        rect(ax, .755, .565, .46, .18, '本文の正規化\n表現ルールの適用')
        arrow(ax, (.37, .73), (.245, .665)); arrow(ax, (.63, .73), (.755, .665))
        rect(ax, .245, .345, .46, .12, 'A〜D群・中立群')
        rect(ax, .755, .345, .46, .12, '7つの0/1指標')
        arrow(ax, (.245, .47), (.245, .415)); arrow(ax, (.755, .47), (.755, .415))
        rect(ax, .5, .12, .95, .17,
             f"群比較 {number(data['eligible_n'])}件\n商品内回帰 {number(data['model_n'])}件・{number(data['model_products'])}商品", face='.95')
        arrow(ax, (.245, .28), (.38, .215)); arrow(ax, (.755, .28), (.62, .215))
        _save(fig, out, 'figure1_analysis_flow')

        fig, ax = plt.subplots(figsize=(3.2, 1.65))
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.99)
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        xs, head, row, top = [0, .70, .845, 1], .16, .16, .995
        for j, text in enumerate(['人工例文', '言及', '具体']):
            ax.add_patch(Rectangle((xs[j], top-head), xs[j+1]-xs[j], head, facecolor='.93', edgecolor='.5', lw=.5))
            ax.text((xs[j]+xs[j+1])/2, top-head/2, text, ha='center', va='center', fontsize=9.5)
        for i, (_, jp, _, value) in enumerate(examples):
            y = top-head-(i+1)*row
            for j, text in enumerate([jp, *[part.strip() for part in value.split('/')]]):
                ax.add_patch(Rectangle((xs[j], y), xs[j+1]-xs[j], row, facecolor='white', edgecolor='.6', lw=.45))
                ax.text(xs[j]+.012 if j == 0 else (xs[j]+xs[j+1])/2, y+row/2, text,
                        ha='left' if j == 0 else 'center', va='center', fontsize=9.5)
        ax.text(0, .08, '具体＝上2行は解約困難，下2行は症状発生', fontsize=8.7, ha='left', va='center')
        _save(fig, out, 'figure2_text_classification_rules')

        fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.2, 2.25), gridspec_kw={'height_ratios': [1, .9]})
        fig.subplots_adjust(left=.23, right=.925, top=.91, bottom=.20, hspace=.95)
        values = [float(g.loc[key, 'mean_loss_percent']) for key in ['A', 'B']]
        ax.barh([1, 0], values, height=.55, facecolor=['white', '.65'], edgecolor='.2', lw=.65)
        limit = max([12.2] + [value*1.15 for value in values if np.isfinite(value)])
        ax.set_yticks([1, 0], ['A 高評価', 'B 低評価']); ax.set_xlim(0, limit); ax.set_xticks([0, 5, 10])
        for y, value in zip([1, 0], values):
            ax.text(value+.15 if np.isfinite(value) else .15, y, f'{value:.2f}' if np.isfinite(value) else 'N/A', va='center', fontsize=9)
        ax.set_title('5%以上群の平均減量率（%）', loc='left', pad=3)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        order = ['positive_loss_below_5pct', 'unchanged', 'gain']
        styles = [('.28', ''), ('.72', ''), ('white', '////')]
        for y, group in zip([1, 0], ['C', 'D']):
            rows = d[d.original_group.eq(group)]
            denominator = float(rows.group_denominator.iloc[0]) if len(rows) else 0
            if denominator <= 0:
                bx.text(50, y, 'N/A', ha='center', va='center', fontsize=9)
                continue
            left = 0
            for key, (color, hatch) in zip(order, styles):
                count = float(rows.loc[rows.weight_change.eq(key), 'n'].iloc[0])
                value = 100*count/denominator
                bx.barh(y, value, left=left, height=.55, facecolor=color, edgecolor='.2', hatch=hatch, lw=.4)
                if value >= 10:
                    bx.text(left+value/2, y, f'{value:.1f}', ha='center', va='center',
                            color='white' if key == order[0] else 'black', fontsize=8.5,
                            bbox=dict(facecolor='white', edgecolor='none', pad=.2) if key == 'gain' else None)
                elif group == 'C':
                    bx.text(100, y+.35, f'{value:.1f}', ha='right', va='bottom', fontsize=8.3)
                left += value
        bx.set_yticks([1, 0], ['C 高評価', 'D 低評価']); bx.set_xlim(0, 100); bx.set_xticks([0, 50, 100])
        bx.tick_params(axis='y', length=0); bx.spines['left'].set_visible(False)
        bx.set_title('5%未満群の体重変化内訳（%）', loc='left', pad=4)
        handles = [Rectangle((0, 0), 1, 1, facecolor=c, hatch=h, edgecolor='.2', lw=.5) for c, h in styles]
        fig.legend(handles, ['少量減量', '変化なし', '増量'], loc='lower center', bbox_to_anchor=(.5, .005),
                   ncol=3, frameon=False, fontsize=8.7, handlelength=1.1, columnspacing=.7, handletextpad=.25)
        _save(fig, out, 'figure3_weight_change_comparison')

        features = [('cancellation_mention', '解約言及'), ('x_cancel_difficulty_expression', '解約困難'),
            ('symptom_mention', '症状言及'), ('x_symptom_experience_expression', '症状発生'),
            ('x_price_negative_expression', '価格不満'), ('x_effect_positive_expression', '効果肯定'),
            ('x_effect_negative_expression', '効果否定')]
        fig, axes = plt.subplots(2, 1, figsize=(3.2, 3.8))
        fig.subplots_adjust(left=.295, right=.965, top=.90, bottom=.17, hspace=.66)
        for ax, comparison, title in zip(axes, ['B_vs_A', 'C_vs_D'], ['(a) 減量率5%以上：B対A', '(b) 減量率5%未満：C対D']):
            maximum = 40.
            for y, (key, label) in zip(range(6, -1, -1), features):
                row = f.loc[f.comparison.eq(comparison) & f.feature.eq(key)].iloc[0]
                value = {row.first_group: 100*row.n_first_flagged/row.n_first_with_text if row.n_first_with_text else np.nan,
                         row.second_group: 100*row.n_second_flagged/row.n_second_with_text if row.n_second_with_text else np.nan}
                low, high = ('B', 'A') if comparison == 'B_vs_A' else ('D', 'C')
                ax.plot([value[low], value[high]], [y, y], lw=.65, color='.55', zorder=1)
                ax.scatter([value[low]], [y], s=20, c='black', edgecolors='black', linewidths=.7, zorder=3)
                ax.scatter([value[high]], [y], s=20, c='white', edgecolors='black', linewidths=.7, zorder=4)
                maximum = max([maximum] + [v for v in value.values() if np.isfinite(v)])
                if any(not np.isfinite(v) for v in value.values()):
                    ax.text(39, y, 'N/A', ha='right', va='center', fontsize=8)
            limit = 40 if maximum <= 40 else 10*np.ceil(maximum/10)
            ax.set_yticks(range(6, -1, -1), [label for _, label in features]); ax.set_ylim(-.55, 6.55)
            ax.set_xlim(-.6, limit); ax.set_xticks([0, 20, 40] if limit == 40 else [0, limit/2, limit])
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', length=0, pad=4); ax.grid(axis='x', color='.90', lw=.5); ax.set_axisbelow(True)
            ax.spines['left'].set_visible(False); ax.set_title(title, loc='right', pad=5)
        axes[1].set_xlabel('群内出現率（%）', labelpad=2)
        _save(fig, out, 'figure4_text_expression_comparison')


def make_figures(run_dir, classifier, mention_patterns):
    plt.rcdefaults()
    run_dir=Path(run_dir)
    data=_inputs(run_dir)
    font=_japanese_font(data['config'].get('discordance_font_file'))
    templates=[('Cancellation','解約しました。','I cancelled.','cancellation_mention','x_cancel_difficulty_expression'),
               ('Cancellation','解約できませんでした。','I could not cancel.','cancellation_mention','x_cancel_difficulty_expression'),
               ('Symptoms','副作用はありませんでした。','I had no side effects.','symptom_mention','x_symptom_experience_expression'),
               ('Symptoms','腹痛が出ました。','I developed abdominal pain.','symptom_mention','x_symptom_experience_expression')]
    examples=[(domain,jp,en,f'{int(bool(re.search(mention_patterns[broad],jp)))} / {classifier(jp)[specific]}')
              for domain,jp,en,broad,specific in templates]
    journal_figures(run_dir,data,font,examples)
    nlp_figures(run_dir,data,font,examples)


if __name__=='__main__':
    import argparse
    from Analysis import discordance_text_flags,PATTERNS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    make_figures(args.run_dir,discordance_text_flags,PATTERNS)
