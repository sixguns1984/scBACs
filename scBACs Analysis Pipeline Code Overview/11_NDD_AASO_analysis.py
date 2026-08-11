"""
Cell-type-specific age at accelerated-aging state onset (AASO) in NDDs.

This script implements only the final manuscript definition:
- cell-level RAA from an OLS control reference: CellAge ~ Age + Sex + donor_cell_count
- cell-type-specific accelerated-aging threshold = adult-control RAA 75th percentile
- donor AASO = first from-below crossing of the smoothed RAA trajectory along the
  scBAC-predicted cellular-age axis
- AD discovery/replication, extension to FTD/PD/FTLD/ALS, and molecular correlates

AASO is an inferred position on the predicted cellular-age axis, not donor chronological age.
"""

from pathlib import Path
import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from downstream_analysis_utils import (
    MAIN_CELLTYPES, add_scbac_adult, compare_aaso_across_celltypes,
    run_aaso_disease_ols, run_pathology_associations, normalize_log1p,
    bh_fdr, encode_sex,
)
import statsmodels.formula.api as smf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path('./')
DATA_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "NDD_AASO"
FIGURE_DIR = PROJECT_ROOT / "figures" / "NDD_AASO"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLS=["Benchmarking","Ensemble_Adult","Ensemble_Full"]
AASO_MODEL_COL="Ensemble_Adult"
AASO_RAA_COL="Ensemble_Adult_RAA"
AASO_QUANTILE=75
AASO_QUANTILE_COL="q75"
RAA_MODEL_ROOT=Path('./')
AASO_THRESHOLD_FILE=Path('./thresholds_20260602_103912.json')
REBUILD_THRESHOLD_JSON=False
EXPORT_FIXED_DONOR_RAA=True
MAIN_METADATA=DATA_DIR/"meta_human_cortex_scrna_atlas_CT_NDDs.csv"
RAA_REFERENCE_FILES=[MAIN_METADATA,DATA_DIR/"meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv"]
ALS_FTLD_METADATA=DATA_DIR/"meta_ALS_FTLD.csv"
SEAAD_METADATA=DATA_DIR/"meta_SEAAD_PFC_replication.csv"
GSE263468_METADATA=DATA_DIR/"meta_GSE263468_PFC_AD_replication.csv"
RUN_GENE_ANALYSIS=False
NDD_EXPRESSION_H5AD=DATA_DIR/"sce_NDD_expression.h5ad"
ROSMAP_INH_METADATA=DATA_DIR/"meta_ROSMAC_PFC_Inh_Subclass_discovery.csv"
SEAAD_INH_METADATA=DATA_DIR/"meta_SEAAD_PFC_Inh_Subclass_replication.csv"
GSE263468_INH_METADATA=DATA_DIR/"meta_GSE263468_Inh_Subclass_replication.csv"
SUBTYPE_COL="sub_celltype_celltypist_raw"

# =============================================================================
# 2. FIXED PRETRAINED RAA MODELS + SAVED JSON THRESHOLDS
# =============================================================================

def ensure_model_columns(df):
    d=df.copy()
    if "Ensemble_Adult" not in d.columns:
        cols=[c for c in ["elasticnet_Adult","clm_Adult","transf_Adult"] if c in d.columns]
        if len(cols)==3:
            d["Ensemble_Adult"]=d[cols].apply(pd.to_numeric,errors="coerce").mean(axis=1)
        elif "scBAC_adult" in d.columns:
            d["Ensemble_Adult"]=pd.to_numeric(d["scBAC_adult"],errors="coerce")
    if "Ensemble_Full" not in d.columns:
        cols=[c for c in ["elasticnet_Full","clm_Full","transf_Full"] if c in d.columns]
        if len(cols)==3:
            d["Ensemble_Full"]=d[cols].apply(pd.to_numeric,errors="coerce").mean(axis=1)
    if "Benchmarking" not in d.columns and "Muralidharan_age" in d.columns:
        d["Benchmarking"]=pd.to_numeric(d["Muralidharan_age"],errors="coerce")
    return d

def make_original_raa_donor_uid(df):
    d=df.copy()
    donor=d["donor_id"].astype(str)
    if "dataset" in d.columns:
        donor=d["dataset"].astype(str)+"_"+donor
    if "sub_tissue" in d.columns:
        tissue=d["sub_tissue"].fillna("NA").astype(str)
        donor=donor+"_"+tissue
    d["donor_uid"]=donor
    return d

def _encode_sex_fixed(series):
    return series.astype(str).map({"Female":0,"female":0,"F":0,"f":0,"Male":1,"male":1,"M":1,"m":1})

def calculate_celllevel_raa_from_saved_model(data,model_file,donor_id_col="donor_uid",celltype_col="celltype",age_pred_col="Ensemble_Adult",age_death_col="Age_at_death",sex_col="Sex",verbose=True):
    model_file=Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Missing fixed cell-level RAA model: {model_file}")
    with open(model_file,"rb") as f:
        saved=pickle.load(f)
    models=saved["models"] if isinstance(saved,dict) and "models" in saved else saved
    result=data.copy()
    result["sex_encoded"]=_encode_sex_fixed(result[sex_col])
    result["donor_cell_count"]=result.groupby(donor_id_col)[donor_id_col].transform("size")
    result["RAA"]=np.nan
    if verbose:
        print(f"Loading fixed cell-level RAA model: {model_file}")
    for celltype in result[celltype_col].dropna().unique():
        mask=result[celltype_col]==celltype
        if celltype not in models:
            if verbose:
                print(f"  Warning: no RAA model for {celltype}")
            continue
        model_info=models[celltype]
        required=[age_pred_col,age_death_col,"sex_encoded","donor_cell_count"]
        valid=mask & result[required].notna().all(axis=1)
        if valid.sum()==0:
            continue
        d=result.loc[valid].copy()
        if model_info.get("type")=="mean":
            raa=pd.to_numeric(d[age_pred_col],errors="coerce").to_numpy(float)-float(model_info["mean"])
        elif model_info.get("type")=="ols":
            features=list(model_info["features"])
            coef=model_info["coef"]
            X=pd.DataFrame(index=d.index)
            for feature in features:
                if feature=="const":
                    X[feature]=1.0
                elif feature in d.columns:
                    X[feature]=pd.to_numeric(d[feature],errors="coerce")
                else:
                    raise KeyError(f"Feature {feature!r} required by {model_file} is missing")
            predicted=np.zeros(len(X),dtype=float)
            for feature in features:
                predicted+=float(coef.get(feature,0.0))*X[feature].to_numpy(float)
            raa=pd.to_numeric(d[age_pred_col],errors="coerce").to_numpy(float)-predicted
        else:
            raise ValueError(f"Unsupported cell-level RAA model type for {celltype}: {model_info.get('type')}")
        result.loc[valid,"RAA"]=raa
    return result.drop(columns=["sex_encoded","donor_cell_count"],errors="ignore")

def calculate_donor_level_celltype_raa(data,model_file,donor_id_col="donor_uid",celltype_col="celltype",age_pred_col="Ensemble_Adult",age_death_col="Age_at_death",sex_col="Sex",return_model=False):
    model_file=Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Missing fixed donor-level RAA model: {model_file}")
    with open(model_file,"rb") as f:
        saved=pickle.load(f)
    models=saved["models"] if isinstance(saved,dict) and "models" in saved else saved
    result=data.copy()
    result["sex_encoded"]=_encode_sex_fixed(result[sex_col])
    result["donor_RAA"]=np.nan
    print(f"Loading fixed donor-level RAA model: {model_file}")
    for celltype in result[celltype_col].dropna().unique():
        mask=result[celltype_col]==celltype
        if celltype not in models:
            print(f"  Warning: no donor RAA model for {celltype}")
            continue
        info=models[celltype]
        d=result.loc[mask].copy()
        if info.get("type")=="median":
            result.loc[mask,"donor_RAA"]=pd.to_numeric(d[age_pred_col],errors="coerce")-float(info["value"])
        elif info.get("type")=="ols":
            features=list(info["features"])
            X=d[features].apply(pd.to_numeric,errors="coerce")
            X=sm.add_constant(X,has_constant="add")
            expected=info["model"].predict(X)
            result.loc[mask,"donor_RAA"]=pd.to_numeric(d[age_pred_col],errors="coerce").to_numpy(float)-np.asarray(expected,float)
        else:
            raise ValueError(f"Unsupported donor RAA model type for {celltype}: {info.get('type')}")
    result=result.drop(columns=["sex_encoded"],errors="ignore")
    return (result,models) if return_model else result

def compute_and_save_thresholds_from_controls(ctrl_data,raa_col="Ensemble_Adult_RAA",celltype_col="celltype",quantiles=[5,25,50,75,95],save_dir="./thresholds",save_format="json",verbose=True):
    if isinstance(ctrl_data,(str,Path)):
        ctrl_data=pd.read_csv(ctrl_data)
    elif isinstance(ctrl_data,pd.DataFrame):
        ctrl_data=ctrl_data.copy()
    else:
        raise ValueError("ctrl_data must be DataFrame or file path")
    for col in [raa_col,celltype_col]:
        if col not in ctrl_data.columns:
            raise ValueError(f"Required column {col!r} not found in control data")
    thresholds={}
    rows=[]
    for celltype in ctrl_data[celltype_col].dropna().unique():
        values=pd.to_numeric(ctrl_data.loc[ctrl_data[celltype_col]==celltype,raa_col],errors="coerce").dropna()
        if len(values)<10:
            if verbose:
                print(f"  Warning: {celltype} has only {len(values)} values, skipping")
            continue
        qvals=np.percentile(values,quantiles)
        thresholds[celltype]={f"q{q}":float(qvals[i]) for i,q in enumerate(quantiles)}
        thresholds[celltype]["n_cells"]=int(len(values))
        thresholds[celltype]["mean"]=float(values.mean())
        thresholds[celltype]["std"]=float(values.std())
        rows.append({"celltype":celltype,**thresholds[celltype]})
    thresholds_df=pd.DataFrame(rows)
    save_dir=Path(save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_format in ["json","both"]:
        path=save_dir/f"thresholds_{timestamp}.json"
        with open(path,"w") as f:
            json.dump({"thresholds":thresholds,"metadata":{"created_date":timestamp,"quantiles":quantiles,"raa_col":raa_col,"n_cell_types":len(thresholds)}},f,indent=2)
        if verbose:
            print(f"Thresholds saved to: {path}")
    if save_format in ["pickle","both"]:
        path=save_dir/f"thresholds_{timestamp}.pkl"
        with open(path,"wb") as f:
            pickle.dump({"thresholds":thresholds,"thresholds_df":thresholds_df,"metadata":{"created_date":timestamp,"quantiles":quantiles,"raa_col":raa_col,"celltype_col":celltype_col}},f)
    return thresholds,thresholds_df

def load_thresholds(threshold_file=None,threshold_dir="./thresholds",quantile_of_interest=75,verbose=True):
    if threshold_file is None:
        threshold_dir=Path(threshold_dir)
        json_files=list(threshold_dir.glob("thresholds_*.json"))
        pickle_files=list(threshold_dir.glob("thresholds_*.pkl"))
        if json_files:
            threshold_file=max(json_files,key=lambda p:p.stat().st_mtime)
        elif pickle_files:
            threshold_file=max(pickle_files,key=lambda p:p.stat().st_mtime)
        else:
            raise FileNotFoundError(f"No threshold files found in {threshold_dir}")
    threshold_file=Path(threshold_file)
    if threshold_file.suffix.lower()==".json":
        with open(threshold_file,"r") as f:
            data=json.load(f)
        thresholds=data["thresholds"]
        thresholds_df=None
        metadata=data.get("metadata",{})
    elif threshold_file.suffix.lower()==".pkl":
        with open(threshold_file,"rb") as f:
            data=pickle.load(f)
        thresholds=data["thresholds"]
        thresholds_df=data.get("thresholds_df")
        metadata=data.get("metadata",{})
    else:
        raise ValueError(f"Unsupported threshold file: {threshold_file}")
    quantile_col=f"q{quantile_of_interest}"
    missing=[ct for ct,v in thresholds.items() if quantile_col not in v]
    if missing:
        raise KeyError(f"{quantile_col} missing for cell types: {missing}")
    if verbose:
        print(f"Loaded fixed thresholds: {threshold_file}")
        print(f"  Created: {metadata.get('created_date','unknown')}; cell types: {len(thresholds)}; using {quantile_col}")
    return thresholds,thresholds_df,quantile_col

def threshold_dict_to_dataframe(thresholds,quantile_col="q75"):
    return pd.DataFrame([{"celltype":ct,"threshold":float(values[quantile_col])} for ct,values in thresholds.items() if quantile_col in values])

def fast_fit_donor_smooth_curve_batch(donor_cell_data,age_pred_col="Ensemble_Adult",age_gap_col="Ensemble_Adult_RAA",n_points=100):
    if len(donor_cell_data)<8:
        return None,None
    try:
        d=donor_cell_data[[age_pred_col,age_gap_col]].apply(pd.to_numeric,errors="coerce").dropna().sort_values(age_pred_col)
        if len(d)<8:
            return None,None
        age_pred=d[age_pred_col].to_numpy(float)
        age_gap=d[age_gap_col].to_numpy(float)
        q1,q3=np.percentile(age_gap,[5,95])
        iqr=q3-q1
        mask=(age_gap>=q1-1.5*iqr)&(age_gap<=q3+1.5*iqr)
        age_pred_clean=age_pred[mask]
        age_gap_clean=age_gap[mask]
        if len(age_pred_clean)<6:
            return None,None
        age_pred_smooth=np.linspace(age_pred_clean.min(),age_pred_clean.max(),n_points)
        distances=np.abs(age_pred_smooth[:,None]-age_pred_clean)
        window_masks=distances<=2.0
        age_gap_smooth=np.full(n_points,np.nan)
        for i in range(n_points):
            if np.any(window_masks[i]):
                age_gap_smooth[i]=np.mean(age_gap_clean[window_masks[i]])
        valid=np.isfinite(age_gap_smooth)
        if valid.sum()>=4:
            f=interp1d(age_pred_smooth[valid],age_gap_smooth[valid],kind="linear",bounds_error=False,fill_value="extrapolate")
            return age_pred_smooth,f(age_pred_smooth)
    except Exception:
        pass
    return None,None

def fast_find_threshold_from_curve(age_pred_smooth,age_gap_smooth,target_val=0):
    indices=np.where(age_gap_smooth>target_val)[0]
    if len(indices)==0:
        return None
    first_idx=int(indices[0])
    if first_idx==0:
        return "Onset < Earliest Observed Age"
    x1,x2=age_pred_smooth[first_idx-1],age_pred_smooth[first_idx]
    y1,y2=age_gap_smooth[first_idx-1],age_gap_smooth[first_idx]
    if y2==y1:
        return float(x1)
    return float(x1+(target_val-y1)*(x2-x1)/(y2-y1))

def fast_calculate_donor_curve_thresholds(df,thresholds,quantile_col="q75",donor_id_col="donor_uid",celltype_col="celltype",age_pred_col="Ensemble_Adult",age_gap_col="Ensemble_Adult_RAA",min_cells=10,n_points=100,verbose=False):
    rows=[]
    curves={}
    for (donor,celltype),group in df.groupby([donor_id_col,celltype_col],observed=True):
        if len(group)<min_cells or celltype not in thresholds or quantile_col not in thresholds[celltype]:
            continue
        target=float(thresholds[celltype][quantile_col])
        x,y=fast_fit_donor_smooth_curve_batch(group,age_pred_col=age_pred_col,age_gap_col=age_gap_col,n_points=n_points)
        if x is None:
            continue
        onset=fast_find_threshold_from_curve(x,y,target_val=target)
        if onset is None or isinstance(onset,str):
            continue
        rows.append({"donor_id":donor,"celltype":celltype,"threshold_age":onset,"threshold_reference":target,"threshold_quantile":quantile_col,"n_cells":len(group)})
        curves[f"{donor}_{celltype}"]={"age_pred":x,"age_gap":y,"threshold":onset,"target_value":target}
    return pd.DataFrame(rows),curves

def compute_aaso(cells,thresholds,donor_col="donor_uid",celltype_col="celltype",predicted_age_col="Ensemble_Adult",raa_col="Ensemble_Adult_RAA",min_cells=10,metadata_cols=None):
    metadata_cols=metadata_cols or []
    if isinstance(thresholds,pd.DataFrame):
        threshold_dict={str(r["celltype"]):{"q75":float(r["threshold"])} for _,r in thresholds.iterrows()}
        quantile_col="q75"
    else:
        threshold_dict=thresholds
        quantile_col=AASO_QUANTILE_COL
    raw,_=fast_calculate_donor_curve_thresholds(cells,threshold_dict,quantile_col=quantile_col,donor_id_col=donor_col,celltype_col=celltype_col,age_pred_col=predicted_age_col,age_gap_col=raa_col,min_cells=min_cells,n_points=100,verbose=False)
    if raw.empty:
        cols=[donor_col,celltype_col,"AASO","N_Cells"]+metadata_cols
        return pd.DataFrame(columns=list(dict.fromkeys(cols)))
    out=raw.rename(columns={"donor_id":donor_col,"threshold_age":"AASO","n_cells":"N_Cells"})
    if celltype_col!="celltype":
        out=out.rename(columns={"celltype":celltype_col})
    meta=cells.groupby([donor_col,celltype_col],observed=True).first().reset_index()
    keep=[donor_col,celltype_col]+[c for c in metadata_cols if c in meta.columns]
    out=out.merge(meta[keep],on=[donor_col,celltype_col],how="left")
    return out

def prepare_fixed_raa_cells(df):
    d=ensure_model_columns(df)
    d=add_scbac_adult(d)
    d=make_original_raa_donor_uid(d)
    for model_col in MODEL_COLS:
        if model_col not in d.columns:
            continue
        model_file=RAA_MODEL_ROOT/model_col/"adult_linear_celllevel.pkl"
        if not model_file.exists():
            if model_col==AASO_MODEL_COL:
                raise FileNotFoundError(model_file)
            continue
        res=calculate_celllevel_raa_from_saved_model(d,model_file,donor_id_col="donor_uid",celltype_col="celltype",age_pred_col=model_col,age_death_col="Age_at_death",sex_col="Sex",verbose=False)
        d[model_col+"_RAA"]=res["RAA"].to_numpy()
    if AASO_MODEL_COL not in d.columns or AASO_RAA_COL not in d.columns:
        raise KeyError(f"AASO requires {AASO_MODEL_COL} and {AASO_RAA_COL}")
    d["scBAC_adult"]=pd.to_numeric(d[AASO_MODEL_COL],errors="coerce")
    d["RAA"]=pd.to_numeric(d[AASO_RAA_COL],errors="coerce")
    return d

def build_fixed_donor_raa_table(cells):
    meta_cols=[c for c in ["status","dataset","sub_tissue","Sex","Age_at_death","PMI"] if c in cells.columns]
    agg={c:"first" for c in meta_cols}
    for model_col in MODEL_COLS:
        if model_col in cells.columns:
            agg[model_col]="median"
    donor=cells.groupby(["donor_uid","celltype"],observed=True).agg(agg).reset_index()
    for model_col in MODEL_COLS:
        if model_col not in donor.columns:
            continue
        model_file=RAA_MODEL_ROOT/model_col/"adult_linear_DonorRAA.pkl"
        if not model_file.exists():
            if model_col==AASO_MODEL_COL:
                raise FileNotFoundError(model_file)
            continue
        res=calculate_donor_level_celltype_raa(donor.copy(),model_file,donor_id_col="donor_uid",celltype_col="celltype",age_pred_col=model_col,age_death_col="Age_at_death",sex_col="Sex")
        donor[model_col+"_RAA"]=res["donor_RAA"].to_numpy()
    return donor


def rebuild_threshold_json_from_controls():
    parts=[]
    for path in RAA_REFERENCE_FILES:
        if path.exists():
            x=pd.read_csv(path,index_col=0)
            x=x[(x["status"]=="CT")&(pd.to_numeric(x["Age_at_death"],errors="coerce")>=18)&x["celltype"].isin(MAIN_CELLTYPES)].copy()
            parts.append(prepare_fixed_raa_cells(x))
    if not parts:
        raise FileNotFoundError("No adult-control prediction resources found for threshold rebuilding")
    controls=pd.concat(parts,axis=0)
    for model_col in MODEL_COLS:
        raa_col=model_col+"_RAA"
        if raa_col in controls.columns:
            compute_and_save_thresholds_from_controls(controls,raa_col=raa_col,celltype_col="celltype",quantiles=[5,25,50,75,95],save_dir=RAA_MODEL_ROOT/model_col,save_format="json",verbose=True)

if REBUILD_THRESHOLD_JSON:
    rebuild_threshold_json_from_controls()


    
thresholds_dict,_,AASO_QUANTILE_COL=load_thresholds(threshold_file=AASO_THRESHOLD_FILE,threshold_dir=RAA_MODEL_ROOT/AASO_MODEL_COL,quantile_of_interest=AASO_QUANTILE,verbose=True)
thresholds=threshold_dict_to_dataframe(thresholds_dict,AASO_QUANTILE_COL)
# =============================================================================
# VISUALIZATION HELPERS: RESTORED FROM THE ORIGINAL WORKING ANALYSIS
# =============================================================================
CELLTYPE_ORDER=["Mic","OPC","Oli","Ast","Inh","Exc"]
CELLTYPE_COLORS={"Exc":"#377EB8","Inh":"#E41A1C","Ast":"#4DAF4A","Oli":"#FF7F00","OPC":"#A65628","Mic":"#984EA3"}
STATUS_COLORS={"CT":"#3C5488","MCI":"#8491B4","PD":"#00A087","AD":"#E64B35","Dementia":"#F39B7F","FTD":"#7E6148","FTLD":"#B09C85","ALS":"#DC0000"}
STATUS_MARKERS={"CT":"o","MCI":"s","PD":"^","AD":"D","Dementia":"v","FTD":"P","FTLD":"X","ALS":"h"}

def _safe_savefig(path,dpi=300):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(path,format="pdf",bbox_inches="tight",dpi=dpi)
    plt.close()

def plot_raa_status_bins(cells,out_pdf,raa_col=AASO_RAA_COL,age_col="Age_at_death",bin_size=10):
    d=cells.copy()
    if d.empty or raa_col not in d.columns or age_col not in d.columns or "status" not in d.columns:
        return
    d[age_col]=pd.to_numeric(d[age_col],errors="coerce")
    d[raa_col]=pd.to_numeric(d[raa_col],errors="coerce")
    d=d.dropna(subset=[age_col,raa_col,"celltype","status"])
    if d.empty:
        return
    min_age=int(np.floor(d[age_col].min()/bin_size)*bin_size)
    max_age=int(np.ceil(d[age_col].max()/bin_size)*bin_size)
    if max_age<=min_age:
        return
    bins=np.arange(min_age,max_age+bin_size,bin_size)
    d["Age_bin"]=pd.cut(d[age_col],bins=bins,right=False)
    d["Age_bin_center"]=d["Age_bin"].apply(lambda x:x.mid if pd.notnull(x) else np.nan).astype(float)
    celltypes=[ct for ct in ["Exc","Inh","Ast","Oli","OPC","Mic"] if ct in d["celltype"].astype(str).unique()]
    statuses=[st for st in ["CT","MCI","PD","AD","FTD","FTLD","ALS","Dementia"] if st in d["status"].astype(str).unique()]
    if not celltypes or not statuses:
        return
    fig,axes=plt.subplots(2,3,figsize=(11,7),sharex=True,sharey=True,dpi=300)
    axes=np.asarray(axes).flatten()
    offsets=np.linspace(-1.5,1.5,len(statuses)) if len(statuses)>1 else [0]
    for i,ct in enumerate(celltypes):
        ax=axes[i]
        sub=d[d["celltype"].astype(str)==ct]
        ax.axhline(0,color="#666666",linestyle="--",linewidth=0.9,alpha=0.8,zorder=1)
        for j,st in enumerate(statuses):
            g=sub[sub["status"].astype(str)==st]
            summary=g.groupby("Age_bin_center",observed=True)[raa_col].agg(["mean","sem","count"]).reset_index()
            summary=summary[summary["count"]>0]
            if summary.empty:
                continue
            ax.errorbar(summary["Age_bin_center"].to_numpy(float)+offsets[j],summary["mean"],yerr=summary["sem"],fmt="-"+STATUS_MARKERS.get(st,"o"),color=STATUS_COLORS.get(st,"#555555"),ecolor=STATUS_COLORS.get(st,"#555555"),elinewidth=1.0,capsize=2,markersize=4,markerfacecolor="white",markeredgewidth=1,label=st,zorder=3)
        ax.set_title(ct,fontsize=10,fontweight="bold")
        ax.set_xlabel(f"Age at death ({bin_size}-yr bins)",fontsize=8.5)
        ax.set_ylabel("Ensemble Adult RAA",fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True,linestyle="--",alpha=0.22,color="gray")
        ax.tick_params(labelsize=8)
    for j in range(len(celltypes),len(axes)):
        axes[j].axis("off")
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,title="Status",loc="upper center",bbox_to_anchor=(0.5,1.02),ncol=max(1,len(statuses)),frameon=False,fontsize=8)
    plt.tight_layout()
    _safe_savefig(out_pdf,300)

def plot_aaso_sequence(aaso_df,disease_label,out_pdf,celltype_col="celltype"):
    d=aaso_df.copy()
    if d.empty or celltype_col not in d.columns or "AASO" not in d.columns:
        return
    if "status" in d.columns and disease_label is not None:
        d=d[d["status"].astype(str)==str(disease_label)].copy()
    d["AASO"]=pd.to_numeric(d["AASO"],errors="coerce")
    counts=d.groupby(celltype_col,observed=True)["AASO"].count()
    valid=counts[counts>=3].index
    d=d[d[celltype_col].isin(valid)].dropna(subset=["AASO"])
    if d.empty or d[celltype_col].nunique()<1:
        return
    stats_df=d.groupby(celltype_col,observed=True)["AASO"].agg(["mean","std","count","min","max"]).sort_values("mean")
    order=stats_df.index.astype(str).tolist()
    anova,pairwise=compare_aaso_across_celltypes(d.rename(columns={celltype_col:"celltype"}) if celltype_col!="celltype" else d,celltype_col="celltype" if celltype_col!="celltype" else celltype_col)
    if celltype_col!="celltype":
        pairwise=compare_aaso_across_celltypes(d.rename(columns={celltype_col:"celltype"}),celltype_col="celltype")[1]
    fig=plt.figure(figsize=(12,9))
    gs=fig.add_gridspec(2,2)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax4=fig.add_subplot(gs[1,1])
    heat_gs=gs[1,0].subgridspec(1,2,width_ratios=[0.2,1],wspace=0)
    ax_dendro=fig.add_subplot(heat_gs[0]); ax_heat=fig.add_subplot(heat_gs[1])
    box_data=[d.loc[d[celltype_col].astype(str)==ct,"AASO"].dropna().to_numpy(float) for ct in order]
    bp=ax1.boxplot(box_data,labels=order,patch_artist=True,showfliers=False)
    for patch,ct in zip(bp["boxes"],order):
        patch.set_facecolor(CELLTYPE_COLORS.get(ct,"#AAAAAA")); patch.set_alpha(0.65)
    for pos,ct in enumerate(order,1):
        vals=d.loc[d[celltype_col].astype(str)==ct,"AASO"].dropna().to_numpy(float)
        if len(vals):
            jitter=np.linspace(-0.10,0.10,len(vals))
            ax1.scatter(np.full(len(vals),pos)+jitter,vals,s=8,color="#333333",alpha=0.45,zorder=3)
    earliest=order[0]
    y_all=d["AASO"].to_numpy(float); y_min=np.nanmin(y_all); y_max=np.nanmax(y_all); span=max(y_max-y_min,1.0); current_y=y_max+0.06*span
    if not pairwise.empty:
        for _,r in pairwise.iterrows():
            ct1=str(r["CellType_1"]); ct2=str(r["CellType_2"])
            if earliest not in [ct1,ct2] or not np.isfinite(r["FDR"]) or r["FDR"]>=0.05:
                continue
            other=ct2 if ct1==earliest else ct1
            if other not in order:
                continue
            p1=order.index(earliest)+1; p2=order.index(other)+1
            ax1.plot([p1,p1,p2,p2],[current_y,current_y+0.015*span,current_y+0.015*span,current_y],"k-",linewidth=0.8)
            star="***" if r["FDR"]<0.001 else ("**" if r["FDR"]<0.01 else "*")
            ax1.text((p1+p2)/2,current_y+0.02*span,star,ha="center",va="bottom",fontsize=8)
            current_y+=0.07*span
    ax1.set_ylabel("Age at aging acceleration onset")
    ax1.set_title(f"{disease_label}: donor AASO by cell type\nEarliest: {earliest}",fontweight="bold")
    ax1.tick_params(axis="x",rotation=45); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    y_pos=np.arange(len(stats_df))
    ax2.barh(y_pos,stats_df["mean"],xerr=stats_df["std"],color=[CELLTYPE_COLORS.get(str(ct),"#AAAAAA") for ct in stats_df.index],alpha=0.7,capsize=3)
    ax2.set_yticks(y_pos,stats_df.index.astype(str)); ax2.invert_yaxis()
    ax2.set_xlabel("Mean AASO (years)"); ax2.set_title("Cell-type aging sequence",fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for i,(mean,std) in enumerate(zip(stats_df["mean"],stats_df["std"])):
        ax2.text(mean+(0 if pd.isna(std) else std)+0.3,i,f"{mean:.1f}",va="center",fontsize=7)
    donor_col="donor_uid" if "donor_uid" in d.columns else ("donor_id" if "donor_id" in d.columns else None)
    if donor_col is not None and d[celltype_col].nunique()>=2:
        pivot=d.pivot_table(index=donor_col,columns=celltype_col,values="AASO",aggfunc="first")
        col_order=[ct for ct in order if ct in pivot.columns.astype(str).tolist()]
        pivot.columns=pivot.columns.astype(str); pivot=pivot[col_order]
        cluster=pivot.fillna(pivot.mean())
        if len(cluster)>=2 and cluster.shape[1]>=1 and np.isfinite(cluster.to_numpy()).all():
            try:
                dist=pdist(cluster.to_numpy(float),metric="euclidean")
                linkage=hierarchy.linkage(dist,method="ward")
                dend=hierarchy.dendrogram(linkage,ax=ax_dendro,orientation="right",no_labels=True,color_threshold=None)
                row_order=dend["leaves"]; pivot=pivot.iloc[row_order]
                ax_dendro.invert_xaxis(); ax_dendro.set_xticks([]); ax_dendro.set_yticks([])
            except Exception:
                ax_dendro.axis("off")
        else:
            ax_dendro.axis("off")
        im=ax_heat.imshow(pivot.to_numpy(float),cmap="RdBu_r",aspect="auto")
        ax_heat.set_xticks(np.arange(len(pivot.columns)),pivot.columns,rotation=45,ha="right")
        ax_heat.set_yticks([]); ax_heat.set_ylabel("Donors"); ax_heat.set_title("Donor-level AASO")
        cbar=fig.colorbar(im,ax=ax_heat,shrink=0.75); cbar.set_label("AASO (years)")
    else:
        ax_dendro.axis("off"); ax_heat.axis("off")
    if not pairwise.empty and "FDR" in pairwise.columns:
        sig=pairwise[pairwise["FDR"]<0.05].sort_values("FDR").head(8).copy()
    else:
        sig=pd.DataFrame()
    if sig.empty:
        ax4.text(0.5,0.5,"No significant pairwise differences",ha="center",va="center",transform=ax4.transAxes)
        ax4.set_axis_off()
    else:
        labels=[f"{a}\nvs\n{b}" for a,b in zip(sig["CellType_1"],sig["CellType_2"])]
        vals=-np.log10(np.clip(sig["FDR"].to_numpy(float),1e-300,None))
        ax4.barh(np.arange(len(sig)),vals,color="#C83C3C",alpha=0.75)
        ax4.set_yticks(np.arange(len(sig)),labels,fontsize=7); ax4.invert_yaxis()
        ax4.set_xlabel("-log10(FDR)"); ax4.set_title("Pairwise AASO differences",fontweight="bold")
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    plt.tight_layout()
    _safe_savefig(out_pdf,300)

def plot_disease_effect_forest(effect_df,out_pdf,title,celltype_col="CellType",effect_col="Effect_Size_Years"):
    d=effect_df.copy()
    if d.empty or effect_col not in d.columns:
        return
    d=d.dropna(subset=[effect_col,"CI95_Low","CI95_High"]).copy()
    if d.empty:
        return
    order=[ct for ct in CELLTYPE_ORDER if ct in d[celltype_col].astype(str).unique()]
    order+=sorted([str(ct) for ct in d[celltype_col].dropna().unique() if str(ct) not in order])
    d[celltype_col]=pd.Categorical(d[celltype_col].astype(str),categories=order,ordered=True)
    d=d.sort_values(celltype_col)
    y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(5.5,max(2.5,0.45*len(d)+1.5)))
    for i,r in enumerate(d.itertuples()):
        effect=float(getattr(r,effect_col)); low=float(r.CI95_Low); high=float(r.CI95_High)
        color="#2b5c8f" if effect<0 else "#d95f02"
        ax.errorbar(effect,i,xerr=[[effect-low],[high-effect]],fmt="o",color=color,ecolor=color,elinewidth=1.2,capsize=3,markersize=5,markerfacecolor="white",markeredgewidth=1.2)
        p=float(r.P_value)
        fdr=float(r.FDR) if hasattr(r,"FDR") and pd.notna(r.FDR) else np.nan
        label=f"P={p:.2e}" if not np.isfinite(fdr) else f"P={p:.2e}, FDR={fdr:.2e}"
        ax.text(high+0.02*max(abs(d["CI95_High"]).max()-d["CI95_Low"].min(),1),i,label,va="center",fontsize=7)
    ax.axvline(0,color="#999999",linestyle="--",linewidth=0.8)
    ax.set_yticks(y,d[celltype_col].astype(str)); ax.set_xlabel("Disease effect on AASO (years, 95% CI)")
    ax.set_title(title,fontweight="bold"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); _safe_savefig(out_pdf,300)

def plot_pathology_signed_p(assoc_df,out_pdf,title,celltype_col="CellType"):
    d=assoc_df.copy()
    if d.empty or not {"Phenotype",celltype_col,"Effect_Size","P_value"}.issubset(d.columns):
        return
    d["signed_logP"]=-np.log10(np.clip(pd.to_numeric(d["P_value"],errors="coerce"),1e-300,None))*np.sign(pd.to_numeric(d["Effect_Size"],errors="coerce"))
    phenotypes=d["Phenotype"].dropna().astype(str).unique().tolist()
    if not phenotypes:
        return
    cell_order=[ct for ct in CELLTYPE_ORDER if ct in d[celltype_col].astype(str).unique()]
    cell_order+=sorted([str(ct) for ct in d[celltype_col].dropna().unique() if str(ct) not in cell_order])
    fig,axes=plt.subplots(1,len(phenotypes),figsize=(1.8*len(phenotypes)+1.5,max(2.2,0.33*len(cell_order)+1.0)),squeeze=False,gridspec_kw={"wspace":0.10})
    for i,pheno in enumerate(phenotypes):
        ax=axes[0,i]
        sub=d[d["Phenotype"].astype(str)==pheno].copy()
        sub[celltype_col]=sub[celltype_col].astype(str)
        sub=sub.set_index(celltype_col).reindex(cell_order).reset_index()
        vals=sub["signed_logP"].to_numpy(float)
        colors=["#C57D58" if np.isfinite(v) and v>0 else "#4A729A" for v in vals]
        ax.barh(np.arange(len(cell_order)),vals,color=colors,height=0.65,alpha=0.9)
        lim=-np.log10(0.05); ax.axvline(lim,color="#B0B0B0",linestyle=":",linewidth=0.6); ax.axvline(-lim,color="#B0B0B0",linestyle=":",linewidth=0.6); ax.axvline(0,color="black",linewidth=0.5)
        for j,row in sub.iterrows():
            if "FDR" in sub.columns and pd.notna(row.get("FDR")) and row["FDR"]<0.05 and np.isfinite(row["signed_logP"]):
                ax.text(row["signed_logP"],j,"*",ha="center",va="center",fontsize=8,fontweight="bold")
        ax.set_title(pheno,fontsize=8,fontweight="bold"); ax.set_xlabel("signed -log10(P)",fontsize=7)
        ax.set_yticks(np.arange(len(cell_order)))
        if i==0: ax.set_yticklabels(cell_order,fontsize=7)
        else: ax.set_yticklabels([]); ax.tick_params(axis="y",length=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(title,fontsize=9,fontweight="bold",y=1.02)
    plt.tight_layout(); _safe_savefig(out_pdf,600)

def plot_donor_curve_examples(cells,thresholds_dict,out_pdf,disease_label=None,n_examples=3):
    d=cells.copy()
    if disease_label is not None and "status" in d.columns:
        d=d[d["status"].astype(str)==str(disease_label)].copy()
    if d.empty:
        return
    threshold_df,curves=fast_calculate_donor_curve_thresholds(d,thresholds_dict,quantile_col=AASO_QUANTILE_COL,donor_id_col="donor_uid",celltype_col="celltype",age_pred_col=AASO_MODEL_COL,age_gap_col=AASO_RAA_COL,min_cells=10,n_points=100,verbose=False)
    if threshold_df.empty or not curves:
        return
    donor_counts=threshold_df.groupby("donor_id",observed=True)["celltype"].nunique().sort_values(ascending=False)
    donors=donor_counts.index.astype(str).tolist()[:n_examples]
    fig,axes=plt.subplots(1,len(donors),figsize=(3.0*len(donors),3.0),squeeze=False,sharex=True,sharey=True)
    for j,donor in enumerate(donors):
        ax=axes[0,j]; has=False
        for ct in ["Exc","Inh","Ast","Oli","OPC","Mic"]:
            key=f"{donor}_{ct}"
            if key not in curves:
                continue
            has=True; info=curves[key]; color=CELLTYPE_COLORS.get(ct,"#777777"); cutoff=thresholds_dict.get(ct,{}).get(AASO_QUANTILE_COL,np.nan)
            ax.plot(info["age_pred"],info["age_gap"],color=color,linewidth=0.9,label=f"{ct}: {info['threshold']:.1f} y")
            ax.axvline(info["threshold"],color=color,linestyle=":",linewidth=0.5,alpha=0.65)
            if np.isfinite(cutoff): ax.axhline(cutoff,color=color,linestyle="--",linewidth=0.45,alpha=0.35)
        ax.axhline(0,color="#666666",linewidth=0.4,alpha=0.4)
        ax.set_title(str(donor),fontsize=7,fontweight="bold"); ax.set_xlabel("Predicted age (years)")
        if j==0: ax.set_ylabel("Relative age acceleration (RAA)")
        if has: ax.legend(frameon=False,fontsize=5.5,loc="best")
        else: ax.axis("off")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); _safe_savefig(out_pdf,600)

def plot_gene_aaso_scatter(gene_df,out_pdf,title):
    d=gene_df.copy()
    if d.empty or not {"Disease_log2FC","Beta_Gene","AASO_P"}.issubset(d.columns):
        return
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    groups=d.groupby("CellType",observed=True) if "CellType" in d.columns else [("Genes",d)]
    for ct,g in groups:
        ax.scatter(g["Disease_log2FC"],g["Beta_Gene"],s=18,alpha=0.75,label=str(ct))
    ax.axhline(0,color="#777777",linestyle="--",linewidth=0.7); ax.axvline(0,color="#777777",linestyle="--",linewidth=0.7)
    top=d.sort_values("AASO_P").head(min(12,len(d)))
    for _,r in top.iterrows():
        ax.text(r["Disease_log2FC"],r["Beta_Gene"],str(r["gene"]),fontsize=6)
    ax.set_xlabel("Disease vs CT log2 fold change"); ax.set_ylabel("Gene association with AASO (beta)")
    ax.set_title(title,fontweight="bold"); ax.legend(frameon=False,fontsize=6,bbox_to_anchor=(1.02,1),loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); _safe_savefig(out_pdf,300)

# =============================================================================
# 3. LOAD NDD CELL-LEVEL DATA
# =============================================================================
meta = pd.read_csv(MAIN_METADATA, index_col=0)
if ALS_FTLD_METADATA.exists():
    meta = pd.concat([meta, pd.read_csv(ALS_FTLD_METADATA, index_col=0)], axis=0)


    
meta = meta[meta["celltype"].isin(MAIN_CELLTYPES)].copy()
meta = prepare_fixed_raa_cells(meta)
if EXPORT_FIXED_DONOR_RAA:
    build_fixed_donor_raa_table(meta).to_csv(RESULT_DIR / "NDD_fixed_donor_RAA.csv", index=False)
plot_raa_status_bins(meta,FIGURE_DIR/"CellType_RAA_vs_Age_Status_bins_all_NDDs.pdf")
AASO_META = ["status", "dataset", "sub_tissue", "Sex", "Age_at_death", "PMI"]

# =============================================================================
# 4. AD DISCOVERY AND REPLICATION
# =============================================================================
def prepare_ad_cells(df):
    d = df[df["celltype"].isin(MAIN_CELLTYPES)].copy()
    d = d[d["status"].isin(["CT", "AD"])]
    d = d[d["sub_tissue"].astype(str) == "Prefrontal cortex"]
    return d

dis_cells = prepare_ad_cells(meta[meta["dataset"].astype(str) == "ROSMAP.MIT"])
dis_aaso = compute_aaso(dis_cells, thresholds_dict, predicted_age_col=AASO_MODEL_COL, raa_col=AASO_RAA_COL, metadata_cols=AASO_META)
dis_aaso.to_csv(RESULT_DIR / "AD_discovery_AASO.csv", index=False)

rep_parts = []
for path in [SEAAD_METADATA, GSE263468_METADATA]:
    if path.exists():
        x = pd.read_csv(path, index_col=0)
        x = prepare_fixed_raa_cells(x)
        rep_parts.append(x)



        
rep_cells = prepare_ad_cells(pd.concat(rep_parts, axis=0)) if rep_parts else pd.DataFrame()
rep_aaso = compute_aaso(rep_cells, thresholds_dict, predicted_age_col=AASO_MODEL_COL, raa_col=AASO_RAA_COL, metadata_cols=AASO_META) if not rep_cells.empty else pd.DataFrame()
if not rep_aaso.empty:
    rep_aaso.to_csv(RESULT_DIR / "AD_replication_AASO.csv", index=False)


# Rank AASO within AD donors and compare cell types.
for name,aaso,cells in [("discovery",dis_aaso,dis_cells),("replication",rep_aaso,rep_cells)]:
    if aaso.empty:
        continue
    ad_only=aaso[aaso["status"]=="AD"].copy()
    ad_only.groupby("celltype",observed=True)["AASO"].agg(["mean","median","count"]).sort_values("mean").to_csv(RESULT_DIR/f"AD_{name}_AASO_celltype_ranking.csv")
    anova,pairwise=compare_aaso_across_celltypes(ad_only)
    anova.to_csv(RESULT_DIR/f"AD_{name}_AASO_ANOVA.csv",index=False)
    pairwise.to_csv(RESULT_DIR/f"AD_{name}_AASO_pairwise_Welch.csv",index=False)
    effect=run_aaso_disease_ols(aaso,disease_label="AD")
    effect.to_csv(RESULT_DIR/f"AD_{name}_AASO_disease_effect.csv",index=False)
    plot_aaso_sequence(aaso,"AD",FIGURE_DIR/f"AD_{name}_AASO_celltype_sequence.pdf")
    plot_disease_effect_forest(effect,FIGURE_DIR/f"AD_{name}_AASO_disease_effect.pdf",f"AD vs CT AASO effect ({name})")
    plot_donor_curve_examples(cells,thresholds_dict,FIGURE_DIR/f"AD_{name}_donor_AASO_q75_examples.pdf",disease_label="AD",n_examples=3)


# =============================================================================
# 5. AD PATHOLOGY / COGNITION WITH AASO AS OUTCOME
# =============================================================================
def attach_donor_fields(aaso, cells, fields):
    if aaso.empty:
        return aaso
    first = cells.drop_duplicates("donor_uid").set_index("donor_uid")
    out = aaso.copy()
    for col in fields:
        if col in first.columns:
            out[col] = out["donor_uid"].map(first[col])
    return out

# Reuse the donor-level pathology model by temporarily naming AASO as the outcome.
dis_ph = attach_donor_fields(dis_aaso, dis_cells, ["Braak_stage", "ceradsc", "MMSE"])
dis_ph = dis_ph[dis_ph["status"] == "AD"].copy()
if "MMSE" in dis_ph.columns:
    x = pd.to_numeric(dis_ph["MMSE"], errors="coerce")
    valid = x.notna()
    if valid.sum() >= 4:
        try:
            dis_ph.loc[valid, "MMSE_impairment_quartile"] = pd.to_numeric(
                pd.qcut(x[valid], q=4, labels=[4, 3, 2, 1], duplicates="drop"), errors="coerce"
            )
        except ValueError:
            pass
phenos = [c for c in ["Braak_stage", "ceradsc", "MMSE_impairment_quartile"] if c in dis_ph.columns]
if phenos:
    dis_ph_assoc=run_pathology_associations(dis_ph,phenos,outcome_col="AASO")
    dis_ph_assoc.to_csv(RESULT_DIR/"AD_discovery_AASO_pathology_cognition.csv",index=False)
    plot_pathology_signed_p(dis_ph_assoc,FIGURE_DIR/"AD_discovery_AASO_pathology_cognition.pdf","Discovery: AASO association with AD pathology and cognition")

if not rep_aaso.empty:
    rep_ph = attach_donor_fields(rep_aaso, rep_cells, ["Braak_stage", "CERAD score", "Cognitive Status"])
    rep_ph = rep_ph[rep_ph["status"] == "AD"].copy()
    if "Braak_stage" in rep_ph.columns:
        rep_ph["Braak_numeric"] = rep_ph["Braak_stage"].replace({
            "Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3,
            "Braak IV": 4, "Braak V": 5, "Braak VI": 6,
        })
    if "CERAD score" in rep_ph.columns:
        rep_ph["CERAD_numeric"] = rep_ph["CERAD score"].map({"Frequent": 1.0, "Moderate": 2.0, "Sparse": 3.0, "Absent": 4.0})
    if "Cognitive Status" in rep_ph.columns:
        rep_ph["Cognitive_impairment"] = rep_ph["Cognitive Status"].map({"Dementia": 1.0, "No dementia": 0.0})
    phenos = [c for c in ["Braak_numeric", "CERAD_numeric", "Cognitive_impairment"] if c in rep_ph.columns]
    if phenos:
        rep_ph_assoc=run_pathology_associations(rep_ph,phenos,outcome_col="AASO")
        rep_ph_assoc.to_csv(RESULT_DIR/"AD_replication_AASO_pathology_cognition.csv",index=False)
        plot_pathology_signed_p(rep_ph_assoc,FIGURE_DIR/"AD_replication_AASO_pathology_cognition.pdf","Replication: AASO association with AD pathology and cognition")

# =============================================================================
# 6. EXTEND AASO TO FTD, PD, FTLD AND ALS
# =============================================================================
for disease in ["FTD", "PD", "FTLD", "ALS"]:
    studies = meta.loc[meta["status"] == disease, "dataset"].dropna().unique()
    if len(studies) == 0:
        continue
    d = meta[meta["dataset"].isin(studies)].copy()
    if disease == "FTLD":
        d = d[d["status"] != "ALS"]
    if disease == "ALS":
        d = d[d["status"] != "FTLD"]
    d = d[d["status"].isin(["CT", disease])]
    aaso=compute_aaso(d,thresholds_dict,predicted_age_col=AASO_MODEL_COL,raa_col=AASO_RAA_COL,metadata_cols=AASO_META)
    aaso.to_csv(RESULT_DIR/f"{disease}_AASO.csv",index=False)
    disease_effect=run_aaso_disease_ols(aaso,disease_label=disease)
    disease_effect.to_csv(RESULT_DIR/f"{disease}_AASO_disease_effect.csv",index=False)
    disease_only=aaso[aaso["status"]==disease]
    if not disease_only.empty:
        anova,pairwise=compare_aaso_across_celltypes(disease_only)
        anova.to_csv(RESULT_DIR/f"{disease}_AASO_ANOVA.csv",index=False)
        pairwise.to_csv(RESULT_DIR/f"{disease}_AASO_pairwise_Welch.csv",index=False)
        plot_aaso_sequence(aaso,disease,FIGURE_DIR/f"{disease}_AASO_celltype_sequence.pdf")
    plot_disease_effect_forest(disease_effect,FIGURE_DIR/f"{disease}_CT_AASO_disease_effect.pdf",f"{disease} vs CT AASO effect")

# =============================================================================
# 7. INH SUBTYPE AASO (AD)
# =============================================================================
def load_subtype(path_list):
    parts = []
    for path in path_list:
        if Path(path).exists():
            parts.append(pd.read_csv(path, index_col=0))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, axis=0)
    d = prepare_fixed_raa_cells(d)
    if SUBTYPE_COL not in d.columns:
        for c in ["sub_celltype", "sub_celltype_celltypist", "sub_celltype_celltypist_coarse"]:
            if c in d.columns:
                d[SUBTYPE_COL] = d[c]; break
    return d

# AASO is computed within subtype using the Inh q75 threshold, matching the working
# subtype analysis. Each subtype remains a donor-level cellular distribution.
inh_threshold = thresholds.loc[thresholds["celltype"] == "Inh", "threshold"]
if len(inh_threshold):
    subtype_thresholds = None
    for name, d in [
        ("discovery", load_subtype([ROSMAP_INH_METADATA])),
        ("replication", load_subtype([SEAAD_INH_METADATA, GSE263468_INH_METADATA])),
    ]:
        if d.empty or SUBTYPE_COL not in d.columns:
            continue
        subtype_names = d[SUBTYPE_COL].dropna().astype(str).unique()
        subtype_thresholds = pd.DataFrame({"celltype": subtype_names, "threshold": float(inh_threshold.iloc[0])})
        tmp = d.rename(columns={SUBTYPE_COL: "analysis_subtype"})
        sub_aaso = compute_aaso(
            tmp, subtype_thresholds,
            celltype_col="analysis_subtype", predicted_age_col=AASO_MODEL_COL, raa_col=AASO_RAA_COL, metadata_cols=AASO_META,
        )
        sub_aaso.to_csv(RESULT_DIR/f"AD_{name}_Inh_subtype_AASO.csv",index=False)
        effect=run_aaso_disease_ols(sub_aaso,disease_label="AD",celltype_col="analysis_subtype")
        effect_out=effect.rename(columns={"CellType":"Inh_Subtype"})
        effect_out.to_csv(RESULT_DIR/f"AD_{name}_Inh_subtype_AASO_effect.csv",index=False)
        plot_aaso_sequence(sub_aaso,"AD",FIGURE_DIR/f"AD_{name}_Inh_subtype_AASO_sequence.pdf",celltype_col="analysis_subtype")
        plot_disease_effect_forest(effect_out,FIGURE_DIR/f"AD_{name}_Inh_subtype_AASO_effect.pdf",f"AD vs CT Inh-subtype AASO effect ({name})",celltype_col="Inh_Subtype")

# =============================================================================
# 8. OPTIONAL MOLECULAR CORRELATES OF EARLIER AASO
# =============================================================================
# Final criterion:
#   disease down + positive gene-AASO association, OR
#   disease up   + negative gene-AASO association.
# Gene-AASO regression adjusts for chronological age and sex. Genes are required
# to be associated with AASO at FDR < 0.05 and differentially expressed at
# adjusted P < 0.05, with expression in >=10% of cells in either comparison group.

def _attach_meta_to_adata(adata, metadata):
    ids = adata.obs_names.intersection(metadata.index.astype(str))
    out = adata[ids].copy()
    m = metadata.loc[ids]
    for col in m.columns:
        out.obs[col] = m[col].values
    return out


def _one_ndd_aaso_gene_set(adata, aaso_df, disease, celltype, subtype_col=None, subtype_label=None):
    a = adata.copy()
    if subtype_col is None:
        a = a[a.obs["celltype"].astype(str) == str(celltype)].copy()
        aaso_ct = aaso_df[aaso_df["celltype"].astype(str) == str(celltype)].copy()
    else:
        a = a[a.obs[subtype_col].astype(str) == str(subtype_label)].copy()
        aaso_ct = aaso_df[aaso_df["analysis_subtype"].astype(str) == str(subtype_label)].copy()
    a = a[a.obs["status"].isin(["CT", disease])].copy()
    if a.n_obs == 0 or a.obs["status"].nunique() < 2:
        return pd.DataFrame()
    a = normalize_log1p(a)

    # Disease-vs-study-matched-control Wilcoxon DE.
    sc.tl.rank_genes_groups(a, "status", groups=[disease], reference="CT", method="wilcoxon", pts=True)
    deg = sc.get.rank_genes_groups_df(a, group=disease).rename(
        columns={"names": "gene", "logfoldchanges": "Disease_log2FC", "pvals_adj": "Disease_FDR"}
    )
    pts = a.uns["rank_genes_groups"].get("pts")
    pts_rest = a.uns["rank_genes_groups"].get("pts_rest")
    if pts is not None and disease in pts.columns:
        keep = pts[disease] >= 0.10
        if pts_rest is not None and disease in pts_rest.columns:
            keep |= pts_rest[disease] >= 0.10
        deg = deg[deg["gene"].isin(pts.index[keep].astype(str))]

    # Donor pseudobulk mean log-expression.
    donor_ids, rows = [], []
    for donor, idx in a.obs.groupby("donor_uid", observed=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        block = a.X[idx]
        mean_expr = np.asarray(block.mean(axis=0)).reshape(-1) if hasattr(block, "toarray") else np.asarray(block).mean(axis=0)
        donor_ids.append(str(donor)); rows.append(mean_expr)
    if len(rows) < 5:
        return pd.DataFrame()
    expr = np.vstack(rows)

    cov = aaso_ct.dropna(subset=["AASO"]).drop_duplicates("donor_uid").copy()
    cov["donor_uid"] = cov["donor_uid"].astype(str)
    cov = cov.set_index("donor_uid").reindex(donor_ids)
    cov["Age_numeric"] = pd.to_numeric(cov["Age_at_death"], errors="coerce")
    cov["Sex_numeric"] = encode_sex(cov["Sex"])

    assoc_rows = []
    for j, gene in enumerate(a.var_names.astype(str)):
        d = cov[["AASO", "Age_numeric", "Sex_numeric"]].copy()
        d["GeneExpression"] = expr[:, j]
        d = d.dropna()
        if len(d) < 5 or d["GeneExpression"].nunique() < 2:
            continue
        fit = smf.ols("AASO ~ GeneExpression + Age_numeric + Sex_numeric", data=d).fit()
        assoc_rows.append({
            "gene": gene, "Beta_Gene": fit.params["GeneExpression"],
            "AASO_P": fit.pvalues["GeneExpression"], "N_Donors": len(d),
        })
    assoc = pd.DataFrame(assoc_rows)
    if assoc.empty:
        return pd.DataFrame()
    assoc["AASO_FDR"] = bh_fdr(assoc["AASO_P"])

    out = assoc.merge(deg[["gene", "Disease_log2FC", "Disease_FDR"]], on="gene", how="inner")
    out["earlier_entry_pattern"] = None
    out.loc[(out["Beta_Gene"] > 0) & (out["Disease_log2FC"] < 0), "earlier_entry_pattern"] = "Disease-down / AASO-positive"
    out.loc[(out["Beta_Gene"] < 0) & (out["Disease_log2FC"] > 0), "earlier_entry_pattern"] = "Disease-up / AASO-negative"
    out["Disease"] = disease
    out["CellType"] = celltype if subtype_col is None else subtype_label
    return out[
        (out["AASO_FDR"] < 0.05) &
        (out["Disease_FDR"] < 0.05) &
        out["earlier_entry_pattern"].notna()
    ].copy()


if RUN_GENE_ANALYSIS and NDD_EXPRESSION_H5AD.exists():
    expression = _attach_meta_to_adata(sc.read_h5ad(NDD_EXPRESSION_H5AD), meta)
    for disease in ["AD", "FTD", "PD", "FTLD", "ALS"]:
        aaso_path = RESULT_DIR / ("AD_discovery_AASO.csv" if disease == "AD" else f"{disease}_AASO.csv")
        effect_path = RESULT_DIR / ("AD_discovery_AASO_disease_effect.csv" if disease == "AD" else f"{disease}_AASO_disease_effect.csv")
        if not aaso_path.exists() or not effect_path.exists():
            continue
        aaso_d = pd.read_csv(aaso_path)
        effect = pd.read_csv(effect_path)
        # The Methods describe molecular follow-up for cell types with disease-associated AASO changes.
        # Use FDR<0.05 when available; retaining P<0.05 can be enabled explicitly for exploratory
        # nominal signals but is not the primary public-pipeline default.
        sig_ct = effect.loc[effect["FDR"] < 0.05, "CellType"].astype(str).tolist()
        gene_sets = []
        for ct in sig_ct:
            g = _one_ndd_aaso_gene_set(expression, aaso_d, disease, ct)
            if not g.empty:
                gene_sets.append(g)
        if gene_sets:
            out=pd.concat(gene_sets,ignore_index=True)
            out.to_csv(RESULT_DIR/f"{disease}_earlier_AASO_genes.csv",index=False)
            plot_gene_aaso_scatter(out,FIGURE_DIR/f"{disease}_earlier_AASO_gene_scatter.pdf",f"{disease}: genes associated with earlier AASO")
            go_dir=RESULT_DIR/"GO_input_gene_lists"
            go_dir.mkdir(exist_ok=True)
            for (ct, mode), d in out.groupby(["CellType", "earlier_entry_pattern"], observed=True):
                safe = str(mode).replace(" / ", "_").replace(" ", "_")
                pd.Series(sorted(d["gene"].astype(str).unique()), name="gene").to_csv(
                    go_dir / f"{disease}_{ct}_{safe}.csv", index=False
                )

    # AD Inh-LAMP5 molecular follow-up from the subtype analysis, when subtype
    # labels and raw expression are available in the discovery resource.
    subtype_meta = load_subtype([ROSMAP_INH_METADATA])
    subtype_aaso_path = RESULT_DIR / "AD_discovery_Inh_subtype_AASO.csv"
    if not subtype_meta.empty and subtype_aaso_path.exists() and SUBTYPE_COL in subtype_meta.columns:
        expr_sub = _attach_meta_to_adata(sc.read_h5ad(NDD_EXPRESSION_H5AD), subtype_meta)
        sub_aaso = pd.read_csv(subtype_aaso_path)
        target_candidates = [x for x in sub_aaso["analysis_subtype"].dropna().astype(str).unique() if "LAMP5" in x.upper()]
        if target_candidates:
            target = target_candidates[0]
            lamp5 = _one_ndd_aaso_gene_set(
                expr_sub, sub_aaso, "AD", "Inh", subtype_col=SUBTYPE_COL, subtype_label=target
            )
            lamp5.to_csv(RESULT_DIR/"AD_Inh_LAMP5_earlier_AASO_genes_discovery.csv",index=False)
            plot_gene_aaso_scatter(lamp5,FIGURE_DIR/"AD_Inh_LAMP5_earlier_AASO_gene_scatter_discovery.pdf","AD Inh-LAMP5: genes associated with earlier AASO")
print("NDD AASO analysis complete.")
