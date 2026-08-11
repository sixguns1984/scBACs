"""
Sex-dependent APOE4 effects on age at accelerated-aging state onset (AASO).

The same APOE4 x sex x AD interaction framework used for predicted cellular age is
applied to donor-level AASO. Molecular analysis is restricted to the final Methods:
sex-specific Inh gene-AASO associations (adjusted for age and PMI) intersected with
APOE4 carrier-vs-non-carrier differential expression, with discovery/replication
requirements and concordant directions.
"""

from pathlib import Path
import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.interpolate import interp1d
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downstream_analysis_utils import (
    MAIN_CELLTYPES, add_scbac_adult, encode_apoe4, encode_sex,
    run_apoe_interaction_models, run_status_within_sex_apoe_strata,
    normalize_log1p, bh_fdr,
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path('./')
DATA_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "APOE4_AASO"
FIGURE_DIR = PROJECT_ROOT / "figures" / "APOE4_AASO"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLS=["Benchmarking","Ensemble_Adult","Ensemble_Full"]
AASO_MODEL_COL="Ensemble_Adult"
AASO_RAA_COL="Ensemble_Adult_RAA"
AASO_QUANTILE=75
AASO_QUANTILE_COL="q75"
RAA_MODEL_ROOT=Path('./')
AASO_THRESHOLD_FILE=Path('./thresholds_20260602_103912.json')

EXPORT_FIXED_DONOR_RAA=True

MAIN_METADATA = DATA_DIR / "meta_human_cortex_scrna_atlas_CT_NDDs.csv"
SEAAD_METADATA = DATA_DIR / "meta_SEAAD_PFC_replication.csv"
GSE263468_METADATA = DATA_DIR / "meta_GSE263468_PFC_AD_replication.csv"
APOE_TEMPORAL_METADATA = DATA_DIR / "meta_APOE4_Temporal_cortex_replication.csv"

ROSMAP_INH_METADATA = DATA_DIR / "meta_ROSMAC_PFC_Inh_Subclass_discovery.csv"
SEAAD_INH_METADATA = DATA_DIR / "meta_SEAAD_PFC_Inh_Subclass_replication.csv"
GSE263468_INH_METADATA = DATA_DIR / "meta_GSE263468_Inh_Subclass_replication.csv"
APOE_INH_METADATA = DATA_DIR / "meta_APOE4_Temporal_cortex_Inh_Subclass_replication.csv"
SUBTYPE_COL = "sub_celltype_celltypist_raw"

RUN_GENE_ANALYSIS = False
ROSMAP_EXPRESSION = DATA_DIR / "sce_test.h5ad"
REPLICATION_EXPRESSION = DATA_DIR / "sce_APOE4_aging_replication.h5ad"

APOE_GENOTYPE_CANDIDATES = ["APOE Genotype", "APOE_genotype", "APOE", "apoe_genotype"]

# ============================================================================
# 2. FIXED PRETRAINED RAA MODELS + SAVED JSON Q75 THRESHOLD
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


thresholds_dict,_,AASO_QUANTILE_COL=load_thresholds(threshold_file=AASO_THRESHOLD_FILE,threshold_dir=RAA_MODEL_ROOT/AASO_MODEL_COL,quantile_of_interest=AASO_QUANTILE,verbose=True)
thresholds=threshold_dict_to_dataframe(thresholds_dict,AASO_QUANTILE_COL)

def _save_pdf(path,dpi=300):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(path,format="pdf",bbox_inches="tight",dpi=dpi)
    plt.close()

def _normalize_status_for_plot(series):
    s=series.astype(str)
    return s.map({"AD":"AD","CT":"non-AD","MCI":"non-AD","1":"AD","0":"non-AD"}).fillna(s)

def plot_signed_apoe_aaso_interactions(result_df,out_pdf,title="Combined discovery + replication"):
    d=result_df.copy()
    if d.empty:
        return
    if "Sex_Stratum" in d.columns:
        d=d[d["Sex_Stratum"].astype(str)=="Combined"].copy()
    term_order=["APOE4","APOE4:Status","Sex_numeric:APOE4:Status"]
    term_titles={"APOE4":"APOE4","APOE4:Status":"APOE4 × Status","Sex_numeric:APOE4:Status":"Sex × APOE4 × Status"}
    cell_order=[ct for ct in ["Ast","Exc","Inh","Oli","OPC","Mic"] if ct in d["CellType"].astype(str).unique()]
    if not cell_order:
        cell_order=sorted(d["CellType"].dropna().astype(str).unique())
    fig,axes=plt.subplots(1,3,figsize=(6.4,3.2),sharey=True,gridspec_kw={"wspace":0.12})
    threshold=-np.log10(0.05)
    for i,term in enumerate(term_order):
        ax=axes[i]
        sub=d[d["Term"].astype(str)==term].copy()
        sub["CellType"]=pd.Categorical(sub["CellType"].astype(str),categories=cell_order,ordered=True)
        sub=sub.sort_values("CellType")
        sub["signed_logP"]=-np.log10(np.clip(pd.to_numeric(sub["P_value"],errors="coerce"),1e-300,None))*np.sign(pd.to_numeric(sub["Effect_Size"],errors="coerce"))
        vals=[]
        for ct in cell_order:
            row=sub[sub["CellType"].astype(str)==ct]
            vals.append(float(row["signed_logP"].iloc[0]) if not row.empty else np.nan)
        y=np.arange(len(cell_order))
        ax.barh(y,vals,height=0.62)
        ax.axvline(threshold,linestyle=":",linewidth=0.8)
        ax.axvline(-threshold,linestyle=":",linewidth=0.8)
        ax.axvline(0,linewidth=0.6)
        ax.set_title(term_titles[term],fontsize=8,fontweight="bold")
        ax.set_xlabel("signed -log10(P)",fontsize=7)
        ax.set_yticks(y)
        if i==0:
            ax.set_yticklabels(cell_order,fontsize=7)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y",length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(title,fontsize=9,fontweight="bold",y=1.02)
    plt.tight_layout()
    _save_pdf(out_pdf,300)

def plot_aaso_sex_apoe_disease_interaction(raw_aaso,strata_stats,out_pdf):
    d=raw_aaso.copy()
    if d.empty:
        return
    d["Sex_numeric"]=encode_sex(d["Sex"])
    d["APOE4_numeric"]=pd.to_numeric(d["apoe4"],errors="coerce")
    d["Disease_Group"]=_normalize_status_for_plot(d["status"])
    d=d[d["Disease_Group"].isin(["non-AD","AD"])].copy()
    cell_order=[ct for ct in ["Ast","Exc","Inh","Oli","OPC","Mic"] if ct in d["celltype"].astype(str).unique()]
    fig,axes=plt.subplots(len(cell_order),2,figsize=(8.2,max(3.5,2.8*len(cell_order))),squeeze=False)
    for row,ct in enumerate(cell_order):
        row_data=d[d["celltype"].astype(str)==ct].copy()
        for col,(sex_code,sex_name) in enumerate([(1,"Male"),(0,"Female")]):
            ax=axes[row,col]
            sub=row_data[row_data["Sex_numeric"]==sex_code].copy()
            if sub.empty:
                ax.axis("off")
                continue
            for apoe_code,apoe_name,fmt in [(0.0,"APOE4-","o-"),(1.0,"APOE4+","s--")]:
                g=sub[sub["APOE4_numeric"]==apoe_code]
                summary=g.groupby("Disease_Group",observed=True)["AASO"].agg(["mean","sem","count"])
                xs=[]; ys=[]; errs=[]
                for xlab,xnum in [("non-AD",0),("AD",1)]:
                    if xlab in summary.index:
                        xs.append(xnum); ys.append(summary.loc[xlab,"mean"]); errs.append(summary.loc[xlab,"sem"])
                if xs:
                    ax.errorbar(xs,ys,yerr=errs,fmt=fmt,capsize=3,linewidth=1.1,markersize=4,label=apoe_name)
            legend_labels=[]
            for apoe_code,apoe_name in [(0.0,"APOE4-"),(1.0,"APOE4+")]:
                n_nonad=int(((sub["APOE4_numeric"]==apoe_code)&(sub["Disease_Group"]=="non-AD")).sum())
                n_ad=int(((sub["APOE4_numeric"]==apoe_code)&(sub["Disease_Group"]=="AD")).sum())
                stat=strata_stats[(strata_stats["CellType"].astype(str)==ct)&(strata_stats["Sex"].astype(str)==sex_name)&(strata_stats["APOE4"].astype(str)==("Carrier" if apoe_code==1 else "Non-carrier"))]
                if not stat.empty:
                    r=stat.iloc[0]
                    ptxt=f"{r['P_value']:.2e}" if r["P_value"]<0.001 else f"{r['P_value']:.3f}"
                    legend_labels.append(f"{apoe_name}: beta={r['AD_Effect_Years']:+.2f} y, P={ptxt}, AD/non-AD={n_ad}/{n_nonad}")
                else:
                    legend_labels.append(f"{apoe_name}: AD/non-AD={n_ad}/{n_nonad}")
            handles=ax.get_lines()[:2]
            ax.set_xticks([0,1],["non-AD","AD"])
            ax.set_title(f"{ct} | {sex_name}",fontsize=9,fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("AASO (years)" if col==0 else "")
            if handles:
                ax.legend(handles,legend_labels,frameon=False,fontsize=6,loc="best")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    plt.subplots_adjust(hspace=0.45,wspace=0.20)
    _save_pdf(out_pdf,300)

def plot_aaso_stratified_forest(stats_df,out_pdf):
    d=stats_df.copy()
    if d.empty:
        return
    strata=[("Female","Non-carrier"),("Female","Carrier"),("Male","Non-carrier"),("Male","Carrier")]
    cell_order=[ct for ct in ["Ast","Exc","Inh","Oli","OPC","Mic"] if ct in d["CellType"].astype(str).unique()]
    fig,axes=plt.subplots(2,2,figsize=(8,6),squeeze=False)
    for ax,(sex_name,apoe_name) in zip(axes.flatten(),strata):
        sub=d[(d["Sex"].astype(str)==sex_name)&(d["APOE4"].astype(str)==apoe_name)].copy()
        sub["CellType"]=pd.Categorical(sub["CellType"].astype(str),categories=cell_order,ordered=True)
        sub=sub.sort_values("CellType")
        y=np.arange(len(sub))
        if not sub.empty:
            eff=sub["AD_Effect_Years"].to_numpy(float)
            low=sub["CI95_Low"].to_numpy(float)
            high=sub["CI95_High"].to_numpy(float)
            ax.errorbar(eff,y,xerr=np.vstack([eff-low,high-eff]),fmt="o",capsize=3,linewidth=1.0)
            ax.set_yticks(y,sub["CellType"].astype(str))
            ax.invert_yaxis()
        ax.axvline(0,linestyle="--",linewidth=0.7)
        ax.set_title(f"{sex_name} | {apoe_name}",fontsize=9,fontweight="bold")
        ax.set_xlabel("AD effect on AASO (years)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_pdf(out_pdf,300)

def plot_gene_aaso_dis_rep(integrated_df,sex_name,out_pdf):
    d=integrated_df[integrated_df["Sex"].astype(str)==str(sex_name)].copy()
    if d.empty:
        return
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),gridspec_kw={"wspace":0.28})
    for ax,suffix,title in [(axes[0],"dis","Discovery"),(axes[1],"rep","Replication")]:
        xcol=f"APOE4_log2FC_{suffix}"
        ycol=f"AASO_Beta_{suffix}"
        if xcol not in d.columns or ycol not in d.columns:
            ax.axis("off")
            continue
        ax.scatter(d[xcol],d[ycol],s=14,alpha=0.55)
        candidates=d[d["reproducible"] & d["earlier_entry_pattern"].notna()].copy()
        if not candidates.empty:
            ax.scatter(candidates[xcol],candidates[ycol],s=24,marker="s")
            label_set=candidates.sort_values(f"AASO_P_{suffix}").head(min(12,len(candidates)))
            for _,r in label_set.iterrows():
                ax.text(r[xcol],r[ycol],str(r["gene"]),fontsize=6)
        ax.axhline(0,linestyle="--",linewidth=0.6)
        ax.axvline(0,linestyle="--",linewidth=0.6)
        ax.set_xlabel("APOE4 carrier vs non-carrier log2FC",fontsize=7)
        ax.set_ylabel("Gene association with AASO (beta)",fontsize=7)
        ax.set_title(title,fontsize=8,fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{sex_name} Inh: APOE4-associated AASO genes",fontsize=9,fontweight="bold",y=1.02)
    plt.tight_layout()
    _save_pdf(out_pdf,300)

def find_genotype_col(df):
    for col in APOE_GENOTYPE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def prepare_apoe_cells(df, resource):
    d = df.copy()
    d = d[d["celltype"].isin(MAIN_CELLTYPES)]
    d = d[d["status"].isin(["CT", "MCI", "AD"])]
    d = prepare_fixed_raa_cells(d)
    if "apoe4" not in d.columns:
        genotype = find_genotype_col(d)
        if genotype is None:
            raise KeyError("No APOE genotype column found")
        d["apoe4"] = encode_apoe4(d[genotype])
    else:
        d["apoe4"] = pd.to_numeric(d["apoe4"], errors="coerce")
    d["resource"] = resource
    d = d.dropna(subset=["apoe4", "Sex", "Age_at_death", "status"])
    return d

# Discovery = ROSMAP PFC.
meta = pd.read_csv(MAIN_METADATA, index_col=0)
dis_cells = meta[(meta["dataset"].astype(str) == "ROSMAP.MIT") & (meta["sub_tissue"].astype(str) == "Prefrontal cortex")]
dis_cells = prepare_apoe_cells(dis_cells, "Discovery")

# Replication = SEAAD + GSE263468 + GSE237718 + GSE157827.
rep_parts = [pd.read_csv(p, index_col=0) for p in [SEAAD_METADATA, GSE263468_METADATA, APOE_TEMPORAL_METADATA] if p.exists()]
rep_cells = prepare_apoe_cells(pd.concat(rep_parts, axis=0), "Replication") if rep_parts else pd.DataFrame()
rep_cells = rep_cells[rep_cells['sub_tissue'].isin(['Prefrontal cortex','Temporal cortex'])].copy()

if EXPORT_FIXED_DONOR_RAA:
    build_fixed_donor_raa_table(dis_cells).to_csv(RESULT_DIR / "APOE4_discovery_fixed_donor_RAA.csv", index=False)
    if not rep_cells.empty:
        build_fixed_donor_raa_table(rep_cells).to_csv(RESULT_DIR / "APOE4_replication_fixed_donor_RAA.csv", index=False)



AASO_META = ["status", "dataset", "sub_tissue", "Sex", "Age_at_death", "PMI", "apoe4", "resource"]
dis_aaso=compute_aaso(dis_cells,thresholds_dict,predicted_age_col=AASO_MODEL_COL,raa_col=AASO_RAA_COL,metadata_cols=AASO_META)
rep_aaso=compute_aaso(rep_cells,thresholds_dict,predicted_age_col=AASO_MODEL_COL,raa_col=AASO_RAA_COL,metadata_cols=AASO_META) if not rep_cells.empty else pd.DataFrame()
dis_aaso.to_csv(RESULT_DIR/"APOE4_AASO_discovery.csv",index=False)
if not rep_aaso.empty:
    rep_aaso.to_csv(RESULT_DIR/"APOE4_AASO_replication.csv",index=False)


    
combined_aaso=pd.concat([dis_aaso,rep_aaso],axis=0,ignore_index=True) if not rep_aaso.empty else dis_aaso.copy()
combined_aaso.to_csv(RESULT_DIR/"APOE4_AASO_combined_discovery_replication.csv",index=False)

# =============================================================================
# 3. APOE4 x SEX x AD INTERACTION ON AASO
# =============================================================================
interaction_results={}
for name,aaso in [("discovery",dis_aaso),("replication",rep_aaso),("combined_dis_rep",combined_aaso)]:
    if aaso.empty:
        continue
    for sex_stratum in [None,"Female","Male"]:
        res=run_apoe_interaction_models(aaso,outcome_col="AASO",sex_stratum=sex_stratum,min_n=5)
        suffix="combined" if sex_stratum is None else sex_stratum.lower()
        res.to_csv(RESULT_DIR/f"APOE4_AASO_interaction_{name}_{suffix}.csv",index=False)
        interaction_results[(name,suffix)]=res

combined_interaction=interaction_results.get(("combined_dis_rep","combined"),pd.DataFrame())
plot_signed_apoe_aaso_interactions(combined_interaction,FIGURE_DIR/"APOE4_AASO_interaction_combined_dis_rep_signed_logP.pdf","AASO: combined discovery + replication")

# AD effect within Female/Male x APOE4 carrier/non-carrier strata, using combined donors.
aaso_ad_strata=run_status_within_sex_apoe_strata(combined_aaso,outcome_col="AASO",celltype_col="celltype",status_col="status",sex_col="Sex",apoe_col="apoe4",age_col="Age_at_death",pmi_col="PMI",min_donors=5)
if not aaso_ad_strata.empty:
    aaso_ad_strata["FDR"]=aaso_ad_strata.groupby(["Sex","APOE4"],observed=True)["P_value"].transform(bh_fdr)


    
aaso_ad_strata.to_csv(RESULT_DIR/"APOE4_AASO_AD_effect_within_sex_APOE4_strata_combined_dis_rep.csv",index=False)
plot_aaso_sex_apoe_disease_interaction(combined_aaso,aaso_ad_strata,FIGURE_DIR/"APOE4_AASO_sex_APOE4_AD_interaction_combined_dis_rep.pdf")
plot_aaso_stratified_forest(aaso_ad_strata,FIGURE_DIR/"APOE4_AASO_AD_effect_within_sex_APOE4_strata_combined_dis_rep.pdf")

# =============================================================================
# 4. INH SUBTYPE AASO AND APOE4 INTERACTION
# =============================================================================
def load_subtype(paths, resource):
    parts = [pd.read_csv(p, index_col=0) for p in paths if Path(p).exists()]
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, axis=0)
    d = prepare_fixed_raa_cells(d)
    if "apoe4" not in d.columns:
        col = find_genotype_col(d)
        if col is None:
            return pd.DataFrame()
        d["apoe4"] = encode_apoe4(d[col])
    if SUBTYPE_COL not in d.columns:
        for c in ["sub_celltype2", "sub_celltype_celltypist", "sub_celltype_celltypist_coarse"]:
            if c in d.columns:
                d[SUBTYPE_COL] = d[c]; break
    d["resource"] = resource
    return d

inh_q75=thresholds.loc[thresholds["celltype"]=="Inh","threshold"]
subtype_aaso_parts=[]
if len(inh_q75):
    for name,cells in [("discovery",load_subtype([ROSMAP_INH_METADATA],"Discovery")),("replication",load_subtype([SEAAD_INH_METADATA,GSE263468_INH_METADATA,APOE_INH_METADATA],"Replication"))]:
        if cells.empty or SUBTYPE_COL not in cells.columns:
            continue
        tmp=cells.rename(columns={SUBTYPE_COL:"analysis_subtype"}).copy()
        subtype_names=tmp["analysis_subtype"].dropna().astype(str).unique()
        subtype_thresholds=pd.DataFrame({"celltype":subtype_names,"threshold":float(inh_q75.iloc[0])})
        sub_aaso=compute_aaso(tmp,subtype_thresholds,celltype_col="analysis_subtype",predicted_age_col=AASO_MODEL_COL,raa_col=AASO_RAA_COL,metadata_cols=AASO_META)
        sub_aaso.to_csv(RESULT_DIR/f"APOE4_Inh_subtype_AASO_{name}.csv",index=False)
        subtype_aaso_parts.append(sub_aaso)
        for sex_stratum in [None,"Female","Male"]:
            res=run_apoe_interaction_models(sub_aaso,outcome_col="AASO",celltype_col="analysis_subtype",sex_stratum=sex_stratum,min_n=5)
            suffix="combined" if sex_stratum is None else sex_stratum.lower()
            res.rename(columns={"CellType":"Inh_Subtype"}).to_csv(RESULT_DIR/f"APOE4_Inh_subtype_AASO_interaction_{name}_{suffix}.csv",index=False)
    if subtype_aaso_parts:
        combined_subtype_aaso=pd.concat(subtype_aaso_parts,axis=0,ignore_index=True)
        combined_subtype_aaso.to_csv(RESULT_DIR/"APOE4_Inh_subtype_AASO_combined_dis_rep.csv",index=False)
        for sex_stratum in [None,"Female","Male"]:
            res=run_apoe_interaction_models(combined_subtype_aaso,outcome_col="AASO",celltype_col="analysis_subtype",sex_stratum=sex_stratum,min_n=5)
            suffix="combined" if sex_stratum is None else sex_stratum.lower()
            res.rename(columns={"CellType":"Inh_Subtype"}).to_csv(RESULT_DIR/f"APOE4_Inh_subtype_AASO_interaction_combined_dis_rep_{suffix}.csv",index=False)
            if sex_stratum is None:
                plot_signed_apoe_aaso_interactions(res,FIGURE_DIR/"APOE4_Inh_subtype_AASO_interaction_combined_dis_rep_signed_logP.pdf","Inh subtype AASO: combined discovery + replication")

# =============================================================================
# 5. SEX-SPECIFIC Inh GENES ASSOCIATED WITH AASO
# =============================================================================
def attach_metadata(adata, metadata):
    ids = adata.obs_names.intersection(metadata.index.astype(str))
    a = adata[ids].copy(); m = metadata.loc[ids]
    for col in m.columns:
        a.obs[col] = m[col].values
    return a


def apoe_deg_and_gene_aaso(adata, aaso_df, sex_name, resource_name):
    """APOE4 DE and gene-AASO OLS within one sex and Inh."""
    sex_code = 1 if sex_name == "Male" else 0
    sex_numeric = encode_sex(adata.obs["Sex"])
    a = adata[(adata.obs["celltype"] == "Inh") & (sex_numeric == sex_code)].copy()
    a = a[a.obs["status"].isin(["CT", "MCI", "AD"])].copy()
    if a.n_obs == 0:
        return pd.DataFrame()
    a = normalize_log1p(a)
    a.obs["APOE4_group"] = pd.to_numeric(a.obs["apoe4"], errors="coerce").map({0.0: "Non-carrier", 1.0: "Carrier"})
    a = a[a.obs["APOE4_group"].notna()].copy()
    if a.obs["APOE4_group"].nunique() < 2:
        return pd.DataFrame()
    sc.tl.rank_genes_groups(a, "APOE4_group", groups=["Carrier"], reference="Non-carrier", method="wilcoxon", pts=True)
    deg = sc.get.rank_genes_groups_df(a, group="Carrier").rename(
        columns={"names": "gene", "logfoldchanges": "APOE4_log2FC", "pvals": "APOE4_P", "pvals_adj": "APOE4_FDR"}
    )

    donor_aaso = aaso_df.copy()
    donor_aaso["Sex_numeric"] = encode_sex(donor_aaso["Sex"])
    donor_aaso = donor_aaso[(donor_aaso["celltype"] == "Inh") & (donor_aaso["Sex_numeric"] == sex_code)].dropna(subset=["AASO"])
    donor_aaso = donor_aaso.drop_duplicates("donor_uid").set_index("donor_uid")

    # Donor-level mean log-expression.
    donor_ids = []
    expr_rows = []
    for donor, idx in a.obs.groupby("donor_uid", observed=True).indices.items():
        if donor not in donor_aaso.index:
            continue
        block = a.X[np.asarray(idx, dtype=int)]
        expr_rows.append(np.asarray(block.mean(axis=0)).reshape(-1) if sparse.issparse(block) else np.asarray(block).mean(axis=0))
        donor_ids.append(donor)
    if len(donor_ids) < 5:
        return pd.DataFrame()
    expr = np.vstack(expr_rows)
    cov = donor_aaso.loc[donor_ids].copy()
    cov["Age_numeric"] = pd.to_numeric(cov["Age_at_death"], errors="coerce")
    cov["PMI_numeric"] = pd.to_numeric(cov["PMI"], errors="coerce")

    rows = []
    for j, gene in enumerate(a.var_names.astype(str)):
        d = cov[["AASO", "Age_numeric", "PMI_numeric"]].copy()
        d["GeneExpression"] = expr[:, j]
        d = d.dropna()
        if len(d) < 5 or d["GeneExpression"].nunique() < 2:
            continue
        fit = smf.ols("AASO ~ GeneExpression + Age_numeric + PMI_numeric", data=d).fit()
        rows.append({"gene": gene, "AASO_Beta": fit.params["GeneExpression"], "AASO_P": fit.pvalues["GeneExpression"], "N_Donors": len(d)})
    assoc = pd.DataFrame(rows)
    if not assoc.empty:
        assoc["AASO_FDR"] = bh_fdr(assoc["AASO_P"])
    out = deg.merge(assoc, on="gene", how="inner")
    out["Sex"] = sex_name; out["resource"] = resource_name
    return out


def integrate_aaso_genes(dis_res, rep_res):
    m = dis_res.merge(rep_res, on=["gene", "Sex"], suffixes=("_dis", "_rep"))
    m["DE_concordant"] = np.sign(m["APOE4_log2FC_dis"]) == np.sign(m["APOE4_log2FC_rep"])
    m["AASO_concordant"] = np.sign(m["AASO_Beta_dis"]) == np.sign(m["AASO_Beta_rep"])
    m["reproducible"] = (
        (m["APOE4_FDR_dis"] < 0.05) & (m["APOE4_P_rep"] < 0.05) &
        (m["AASO_FDR_dis"] < 0.05) & (m["AASO_P_rep"] < 0.05) &
        m["DE_concordant"] & m["AASO_concordant"]
    )
    m["earlier_entry_pattern"] = np.where(
        (m["APOE4_log2FC_dis"] < 0) & (m["AASO_Beta_dis"] > 0),
        "APOE4-down / AASO-positive",
        np.where(
            (m["APOE4_log2FC_dis"] > 0) & (m["AASO_Beta_dis"] < 0),
            "APOE4-up / AASO-negative", None,
        ),
    )
    return m

if RUN_GENE_ANALYSIS and ROSMAP_EXPRESSION.exists() and REPLICATION_EXPRESSION.exists():
    ad_dis = attach_metadata(sc.read_h5ad(ROSMAP_EXPRESSION), dis_cells)
    ad_rep = attach_metadata(sc.read_h5ad(REPLICATION_EXPRESSION), rep_cells)
    dis_gene = pd.concat([apoe_deg_and_gene_aaso(ad_dis, dis_aaso, s, "Discovery") for s in ["Female", "Male"]], ignore_index=True)
    rep_gene = pd.concat([apoe_deg_and_gene_aaso(ad_rep, rep_aaso, s, "Replication") for s in ["Female", "Male"]], ignore_index=True)
    dis_gene.to_csv(RESULT_DIR / "APOE4_AASO_genes_discovery.csv", index=False)
    rep_gene.to_csv(RESULT_DIR / "APOE4_AASO_genes_replication.csv", index=False)
    integrated=integrate_aaso_genes(dis_gene,rep_gene)
    integrated[integrated["reproducible"]&integrated["earlier_entry_pattern"].notna()].to_csv(RESULT_DIR/"APOE4_AASO_genes_reproducible.csv",index=False)
    plot_gene_aaso_dis_rep(integrated,"Male",FIGURE_DIR/"APOE4_AASO_genes_Male_discovery_replication.pdf")
    plot_gene_aaso_dis_rep(integrated,"Female",FIGURE_DIR/"APOE4_AASO_genes_Female_discovery_replication.pdf")
print("APOE4 AASO analysis complete.")
