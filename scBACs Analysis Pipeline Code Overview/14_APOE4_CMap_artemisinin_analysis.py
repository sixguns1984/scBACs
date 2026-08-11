"""
APOE4/artemisinin mouse intervention analysis.

Final scope
-----------
Primary manuscript-result input:
    dataset/meta_Apoe4_cmap_artemisinin_mouse_20260705.csv

Retained analyses:
1. Analyze the released human-scBAC prediction metadata directly.
2. Compare APOE3+DMSO, APOE4+DMSO and APOE4+artemisinin at cell level.
3. Aggregate Ensemble_Adult to mouse/sample-level medians and repeat the comparisons.
4. Apply the fixed published Buckley mouse brain clock to the raw mouse h5ad as an
   independent species-matched sensitivity analysis (Ast, Oli and Mic only).


The Buckley clock is fixed: no refitting, recalibration or intercept estimation is
performed. Predictions follow the released coefficient table exactly.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

THIS_DIR=Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0,str(THIS_DIR))

    
from mouse_clock_utils import predict_mouse_clock_celltypes

PROJECT_ROOT=Path("/public/labdata/luojunfeng/project_data/spatial_pvm/tool/scMerge/Cell_Brain_age/Total_cell_analysis/prepare_for_paper_submit/code/revision1_final_code_and_data")
DATA_DIR=PROJECT_ROOT/"dataset"
MODEL_DIR=Path('./')
RESULT_DIR=PROJECT_ROOT/"results_"/"CMAP_drug_repurposing_APOE4_aging"
FIGURE_DIR=PROJECT_ROOT/"figures_"/"CMAP_drug_repurposing_APOE4_aging"
RESULT_DIR.mkdir(parents=True,exist_ok=True)
FIGURE_DIR.mkdir(parents=True,exist_ok=True)

MOUSE_PREDICTION_METADATA=DATA_DIR/"meta_Apoe4_cmap_artemisinin_mouse_20260705.csv"
MOUSE_H5AD=DATA_DIR/"CMAP_mouse_data"/"Apoe4_cmap_artemisinin_mouse.h5ad"
BUCKLEY_MOUSE_CLOCK=MODEL_DIR/"mouse_brain_clock"/"scMouseBrainAgeClock.csv"
RUN_BUCKLEY_MOUSE_CLOCK=True
BUCKLEY_MOUSE_CLOCK_CELLTYPES=["Ast","Oli","Mic"]
BUCKLEY_PREDICTION_OUTPUT=RESULT_DIR/"meta_Apoe4_cmap_artemisinin_mouse_Buckley_clock.csv"

CELLTYPE_ORDER=["Exc","Inh","Ast","Oli","OPC","Mic"]
GROUP_ORDER=["APOE3+DMSO","APOE4+DMSO","APOE4+artemisinin"]

COMPARISONS=[
    ("APOE4+DMSO","APOE3+DMSO","APOE4 effect"),
    ("APOE4+artemisinin","APOE4+DMSO","Artemisinin rescue"),
]

def bh_fdr(pvalues):
    p=np.asarray(pvalues,dtype=float)
    out=np.full(len(p),np.nan,dtype=float)
    valid=np.isfinite(p)
    if valid.sum()==0:
        return out
    pv=p[valid]
    order=np.argsort(pv)
    ranked=pv[order]
    adjusted=ranked*len(ranked)/np.arange(1,len(ranked)+1)
    adjusted=np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted=np.clip(adjusted,0,1)
    restored=np.empty(len(pv),dtype=float)
    restored[order]=adjusted
    out[valid]=restored
    return out

def load_released_meta():
    if not MOUSE_PREDICTION_METADATA.exists():
        raise FileNotFoundError(MOUSE_PREDICTION_METADATA)
    meta=pd.read_csv(MOUSE_PREDICTION_METADATA,index_col=0)
    if "celltype" not in meta.columns:
        raise KeyError("Required column 'celltype' is missing from released metadata")
    meta["celltype"]=meta["celltype"].astype(str).replace({"Ex":"Exc","Olig":"Oli"})
    if "group" not in meta.columns:
        inverse={"APOE3+DMSO":"APOE3","APOE4+DMSO":"APOE4D","APOE4+artemisinin":"APOE4"}
        meta["group"]=meta["Group"].map(inverse)
    if "Ensemble_Adult" not in meta.columns:
        components=["elasticnet_Adult","clm_Adult","transf_Adult"]
        if all(c in meta.columns for c in components):
            meta["Ensemble_Adult"]=meta[components].apply(pd.to_numeric,errors="coerce").mean(axis=1)
        else:
            raise KeyError("Released metadata must contain Ensemble_Adult or its three adult component predictions")
    meta["Ensemble_Adult"]=pd.to_numeric(meta["Ensemble_Adult"],errors="coerce")
    meta=meta[meta["celltype"].isin(CELLTYPE_ORDER)&meta["Group"].isin(GROUP_ORDER)].copy()
    meta=meta.dropna(subset=["Ensemble_Adult"])
    if meta.empty:
        raise ValueError("No analyzable rows remain in released metadata")
    return meta

def infer_mouse_col(df):
    for col in ["orig.ident","donor_id","mouse_id","sample","Sample"]:
        if col in df.columns:
            return col
    raise KeyError("No mouse/sample identifier found; expected orig.ident, donor_id, mouse_id, sample or Sample")

def summarize_groups(df,value_col,level):
    return df.groupby(["celltype","Group"],observed=True)[value_col].agg(
        N="count",
        Mean="mean",
        Median="median",
        SD="std",
        SEM="sem",
        Q25=lambda x:x.quantile(0.25),
        Q75=lambda x:x.quantile(0.75),
    ).reset_index().assign(Level=level)

def run_pairwise_tests(df,value_col,level):
    rows=[]
    for ct in CELLTYPE_ORDER:
        sub=df[df["celltype"]==ct]
        for case,reference,label in COMPARISONS:
            x=pd.to_numeric(sub.loc[sub["Group"]==case,value_col],errors="coerce").dropna()
            y=pd.to_numeric(sub.loc[sub["Group"]==reference,value_col],errors="coerce").dropna()
            if len(x)==0 or len(y)==0:
                continue
            stat,p=mannwhitneyu(x,y,alternative="two-sided")
            rows.append({
                "Level":level,
                "CellType":ct,
                "Comparison":label,
                "Case":case,
                "Reference":reference,
                "Median_Case":float(x.median()),
                "Median_Reference":float(y.median()),
                "Median_Difference":float(x.median()-y.median()),
                "MannWhitney_U":float(stat),
                "P_value":float(p),
                "N_Case":int(len(x)),
                "N_Reference":int(len(y)),
            })
    out=pd.DataFrame(rows)
    if not out.empty:
        out["FDR"]=out.groupby("Comparison",observed=True)["P_value"].transform(bh_fdr)
    return out

def fdr_star(p):
    if not np.isfinite(p):
        return "ns"
    if p<0.001:
        return "***"
    if p<0.01:
        return "**"
    if p<0.05:
        return "*"
    return "ns"

def add_stat_annotations(ax,stats_df,plot_df,value_col,celltype_order):
    if stats_df.empty:
        return
    yvals=pd.to_numeric(plot_df[value_col],errors="coerce").dropna()
    if yvals.empty:
        return
    ymin,ymax=float(yvals.min()),float(yvals.max())
    span=max(ymax-ymin,1.0)
    offsets={GROUP_ORDER[0]:-0.27,GROUP_ORDER[1]:0.0,GROUP_ORDER[2]:0.27}
    top=ymax+0.05*span
    step=0.07*span
    for ct_i,ct in enumerate(celltype_order):
        ct_stats=stats_df[stats_df["CellType"]==ct]
        for level,(case,reference,label) in enumerate(COMPARISONS):
            row=ct_stats[ct_stats["Comparison"]==label]
            if row.empty:
                continue
            fdr=float(row["FDR"].iloc[0])
            x1=ct_i+offsets[reference]
            x2=ct_i+offsets[case]
            y=top+level*step
            ax.plot([x1,x1,x2,x2],[y,y+0.01*span,y+0.01*span,y],color="black",linewidth=0.6)
            ax.text((x1+x2)/2,y+0.012*span,fdr_star(fdr),ha="center",va="bottom",fontsize=7)
    ax.set_ylim(ymin-0.04*span,top+2.0*step)

def plot_group_boxplot(df,stats_df,value_col,out_pdf,title,ylabel,show_points=False,celltype_order=None):
    celltype_order=celltype_order or [ct for ct in CELLTYPE_ORDER if ct in df["celltype"].astype(str).unique()]
    if not celltype_order:
        return
    fig,ax=plt.subplots(figsize=(10,4.8))
    sns.boxplot(data=df,x="celltype",y=value_col,hue="Group",order=celltype_order,hue_order=GROUP_ORDER,showfliers=False,linewidth=1.0,ax=ax)
    if show_points:
        sns.stripplot(data=df,x="celltype",y=value_col,hue="Group",order=celltype_order,hue_order=GROUP_ORDER,dodge=True,color="black",alpha=0.45,size=3,ax=ax)
        handles,labels=ax.get_legend_handles_labels()
        ax.legend(handles[:len(GROUP_ORDER)],labels[:len(GROUP_ORDER)],frameon=False,title="")
    else:
        ax.legend(frameon=False,title="")
    add_stat_annotations(ax,stats_df,df,value_col,celltype_order)
    ax.set_xlabel("Cell type")
    ax.set_ylabel(ylabel)
    ax.set_title(title,fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_pdf,format="pdf",bbox_inches="tight",dpi=300)
    plt.close(fig)

def analyze_released_human_scbac_meta(meta):
    mouse_col=infer_mouse_col(meta)
    summary_cell=summarize_groups(meta,"Ensemble_Adult","Cell level")
    tests_cell=run_pairwise_tests(meta,"Ensemble_Adult","Cell level")
    summary_cell.to_csv(RESULT_DIR/"artemisinin_human_scBAC_celllevel_group_summary.csv",index=False)
    tests_cell.to_csv(RESULT_DIR/"artemisinin_human_scBAC_celllevel_MannWhitney_tests.csv",index=False)
    plot_group_boxplot(meta,tests_cell,"Ensemble_Adult",FIGURE_DIR/"cmap_drug_mouse_cellage_cellLevelboxplot.pdf","Human scBAC-equivalent cellular age: cell level","Ensemble Adult predicted cellular age (years)",show_points=False)


def load_mouse_h5ad_aligned_to_released_meta(meta):
    if not MOUSE_H5AD.exists():
        raise FileNotFoundError(MOUSE_H5AD)
    mouse=sc.read_h5ad(MOUSE_H5AD)
    meta_index=meta.index.astype(str)
    direct=mouse.obs_names.intersection(meta_index)
    if len(direct)==0:
        raise ValueError("No cell IDs from meta_Apoe4_cmap_artemisinin_mouse_20260705.csv match the raw mouse h5ad")
    mouse=mouse[direct].copy()
    aligned=meta.loc[direct].copy()
    for col in aligned.columns:
        mouse.obs[col]=aligned[col].values
    mouse.obs["celltype"]=mouse.obs["celltype"].astype(str).replace({"Ex":"Exc","Olig":"Oli"})
    mouse.obs["Group"]=normalize_group(mouse.obs["Group"])
    mouse_col=infer_mouse_col(mouse.obs)
    mouse.obs["donor_id"]=mouse.obs[mouse_col].astype(str)
    mouse.var_names_make_unique()
    return mouse

def run_buckley_mouse_clock(meta):
    if not RUN_BUCKLEY_MOUSE_CLOCK:
        return
    if not BUCKLEY_MOUSE_CLOCK.exists():
        raise FileNotFoundError(BUCKLEY_MOUSE_CLOCK)
    mouse=load_mouse_h5ad_aligned_to_released_meta(meta)
    buckley=predict_mouse_clock_celltypes(
        mouse,
        param_file=BUCKLEY_MOUSE_CLOCK,
        cell_types=BUCKLEY_MOUSE_CLOCK_CELLTYPES,
        celltype_col="celltype",
        norm=True,
        extra_obs_cols=["orig.ident","donor_id","group","Group"],
    )
    if buckley.empty:
        raise ValueError("Buckley mouse clock returned no predictions")
    buckley["Group"]=normalize_group(buckley["Group"])
    buckley["Pred_age"]=pd.to_numeric(buckley["Pred_age"],errors="coerce")
    buckley=buckley[buckley["Group"].isin(GROUP_ORDER)].dropna(subset=["Pred_age"]).copy()
    buckley.to_csv(BUCKLEY_PREDICTION_OUTPUT,index=True)
    summary_cell=summarize_groups(buckley,"Pred_age","Cell level")
    tests_cell=run_pairwise_tests(buckley,"Pred_age","Cell level")
    summary_cell.to_csv(RESULT_DIR/"Buckley_mouse_clock_celllevel_group_summary.csv",index=False)
    tests_cell.to_csv(RESULT_DIR/"Buckley_mouse_clock_celllevel_MannWhitney_tests.csv",index=False)
    buckley_ct=[ct for ct in BUCKLEY_MOUSE_CLOCK_CELLTYPES if ct in buckley["celltype"].astype(str).unique()]
    plot_group_boxplot(buckley,tests_cell,"Pred_age",FIGURE_DIR/"Buckley_mouse_clock_cellLevelboxplot.pdf","Published Buckley mouse clock: cell level","Buckley mouse-clock predicted age",show_points=False,celltype_order=buckley_ct)
    mouse_level=buckley.groupby(["celltype","donor_id","Group"],observed=True)["Pred_age"].median().reset_index()
    mouse_level.to_csv(RESULT_DIR/"Buckley_mouse_clock_mouse_medians.csv",index=False)
    summary_mouse=summarize_groups(mouse_level,"Pred_age","Mouse level")
    tests_mouse=run_pairwise_tests(mouse_level,"Pred_age","Mouse level")
    summary_mouse.to_csv(RESULT_DIR/"Buckley_mouse_clock_mouselevel_group_summary.csv",index=False)
    tests_mouse.to_csv(RESULT_DIR/"Buckley_mouse_clock_mouselevel_MannWhitney_tests.csv",index=False)
    plot_group_boxplot(mouse_level,tests_mouse,"Pred_age",FIGURE_DIR/"Buckley_mouse_clock_mouseLevelboxplot.pdf","Published Buckley mouse clock: mouse level","Mouse-level median Buckley predicted age",show_points=True,celltype_order=buckley_ct)
    buckley.groupby(["celltype","Group"],observed=True)["Pred_age"].median().unstack("Group").to_csv(RESULT_DIR/"Buckley_mouse_clock_group_medians.csv")

meta=load_released_meta()
analyze_released_human_scbac_meta(meta)
run_buckley_mouse_clock(meta)
print("APOE4/artemisinin analysis complete.")
print(f"Primary metadata: {MOUSE_PREDICTION_METADATA}")
print(f"Results: {RESULT_DIR}")
print(f"Figures: {FIGURE_DIR}")
