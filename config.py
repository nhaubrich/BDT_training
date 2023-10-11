import glob
import numpy as np

#config
#path = "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_22jan23/haddjobs_noSA5/"
path = "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_10Apr23_MF/haddjobs/"
files = [f for f in glob.glob(path+"/*root") if "Run" not in f]
trainOnOdd = "event%2==1"
trainOnEven = "event%2==0"

isVZ = "||".join(["(sampleIndex/100=="+str(y)+")" for y in [32,34,36,37,38]])
isVH = "sampleIndex<0"
VZVH_labeling = "({})+2*({})".format(isVZ,isVH)

#NLO only!
isZJets = "(sampleIndex/100>=182&&sampleIndex<190)"
isWJets = "((sampleIndex/100>=190&&sampleIndex/100<202)||(sampleIndex/100>=400&&sampleIndex/100<402))"
isDYJets = "((sampleIndex/100>=170&&sampleIndex/100<182)||(sampleIndex/100>=500&&sampleIndex/100<504))"

isST = "(sampleIndex>15&&sampleIndex<22)"
isTT = "(sampleIndex>=202&&sampleIndex<213)"
#isZnnJets = "sample

isVJets = "||".join([isZJets,isWJets,isDYJets])
Vj0b_udsg = "((sampleIndex%10)==0&&nGenStatus2DHadFinalAccept==0)"
Vj0b_c = "((sampleIndex%10)==0&&nGenStatus2DHadFinalAccept>0)"
Vj1b = "((sampleIndex%10)==1)"
Vj2b = "((sampleIndex%10)==2)"

HF_5bin_labeling = "({isVJets})*({V_c}+2*({V_b}))+3*({sT})+4*({tt})".format(isVJets=isVJets,V_c=Vj0b_c,V_b="(sampleIndex%10)>0",sT=isST,tt=isTT)

HF_6bin_labeling = "({isVJets})*({V_c}+2*({V_1b})+3*({V_2b}))+4*({sT})+5*({tt})".format(isVJets=isVJets,V_c=Vj0b_c,V_1b=Vj1b,V_2b=Vj2b,sT=isST,tt=isTT)
print(HF_6bin_labeling)
features_0lep = {
"H_mass": [20,50,250],
"H_pt": [10,100,300],
"V_pt": [30,100,400],
"HVdPhi": [20,0,3],
"hJets_btagWP_0": [4,0,4],
"hJets_btagWP_1": [4,0,4],
"HJ1_HJ2_dEta": [20,0,2],
"abs_HJ1_HJ2_dPhi": [20,0,2],
"hJets_leadingPt": [20,50,250],
"hJets_subleadingPt": [20,20,160],
"nAddJets302p4_puid": [5,0,5],
"otherJetsBestBtag": [20,0,1],
"otherJetsHighestPt": [20,0,300],
"minDPhiFromOtherJets": [32,0,3.2],
}

features_1lep = {
"H_mass": [20,90,150],
"H_pt": [10,100,300],
"V_mt": [20,0,200],
"V_pt": [30,100,400],
"VPtjjRatio": [25,0,2],
"HVdPhi": [20,0,3],
"hJets_btagWP_0": [4,0,4],
"hJets_btagWP_1": [4,0,4],
"hJets_leadingPt": [20,50,250],
"hJets_subleadingPt": [20,20,160],
"HJ1_HJ2_dEta": [20,0,2],
"MET_Pt": [20,0,200],
"lepMetDPhi": [20,0,2],
"Top1_mass_fromLepton_regPT_w4MET": [30,0,600],
"nAddJets302p5_puid": [5,0,5],
}
features_1lep_asym = {
"H_mass": [20,90,150],
"H_pt": [10,100,300],
"V_mt": [20,0,200],
"V_pt": [30,100,400],
"VPtjjRatio": [25,0,2],
"HVdPhi": [20,0,3],
"hJets_btagWP_0": [4,0,4],
"hJets_btagWP_1": [4,0,4],
"hJets_leadingPt": [20,50,250],
"hJets_subleadingPt": [20,20,160],
"HJ1_HJ2_dEta": [20,0,2],
"MET_Pt": [20,0,200],
"lepMetDPhi": [20,0,2],
"Top1_mass_fromLepton_regPT_w4MET": [30,0,600],
"nAddJets302p5_puid": [5,0,5],
"lepSignPt": [30,-300,300],
"lepEta": [30,-3,3],
}

features_2lep = {
"H_mass_fit_fallback": [20,90,150],
"H_mass": [10,90,150],
"H_pt_fit_fallback": [20,100,300],
"H_pt": [20,100,300],
"HVdPhi_fit_fallback": [20,0,3],
"HVdPhi": [20,0,3],
"hJets_btagWP_0": [4,0,4],
"hJets_btagWP_1": [4,0,4],
"hJets_pt_0_fit_fallback": [30,0,300],
"hJets_leadingPt": [30,0,300],
"hJets_pt_1_fit_fallback": [30,0,300],
"hJets_subleadingPt": [30,0,300],
"V_mass_fit": [20,0,200],
"V_mass": [20,0,200],
"nAddJets302p4_puid": [5,0,5],
"V_pt_fit": [30,0,600],
"V_pt": [30,0,600],
"jjVPtRatio_fit_fallback": [30,0,2],
"jjVPtRatio": [30,0,2],
"HJ1_HJ2_dEta": [20,0,2],
"HVdR_fit_fallback": [32,0,3.2],
"HVdR": [32,0,3.2],
"MET_Pt": [30,0,300],
"H_mass_sigma_fit": [20,0,5],
"n_recoil_jets_fit": [5,0,5],
"HJ1_HJ2_dR": [32,0,3.2],
}

features_1lep_nobtag = {
"H_mass": [20,90,150],
"H_pt": [10,100,300],
"V_mt": [20,0,200],
"V_pt": [30,100,400],
"VPtjjRatio": [25,0,2],
"HVdPhi": [20,0,3],
"hJets_leadingPt": [20,50,250],
"hJets_subleadingPt": [20,20,160],
"HJ1_HJ2_dEta": [20,0,2],
"MET_Pt": [20,0,200],
"lepMetDPhi": [20,0,2],
"Top1_mass_fromLepton_regPT_w4MET": [30,0,600],
"nAddJets302p5_puid": [5,0,5],
}



def VZVHmax(scores):
    pB = scores[:,0]
    pVZ = scores[:,1]
    pVH = scores[:,2]    
    return (1-pB)/2*(pVZ>pVH)+(1+pB)/2*(pVH>=pVZ)

def twoFaced(scores):
    pB = scores[:,0]
    pVZ = scores[:,1]
    pVH = scores[:,2]    
    return pVH*(pVH>=pVZ)-pVZ*(pVZ>pVH)

def getMaxInd(scores):
    return np.argmax(scores,axis=1)

def getMaxIndPlusScore(scores):
    #return np.argmax(scores,axis=1)+0.7*(np.max(scores,axis=1)>.2)
    return np.argmax(scores,axis=1)+np.max(scores,axis=1)

def getMaxIndPlusScoreGt0p3(scores):
    #return np.argmax(scores,axis=1)+0.7*(np.max(scores,axis=1)>.2)
    return np.argmax(scores,axis=1)+0.7*(np.max(scores,axis=1)>0.3)


customDists = [
    {"func": VZVHmax,"label":"VZVHmax","binning":[100,0,1]},
    {"func": twoFaced,"label":"twoFaced","binning":[100,-1,1]}
]
VZVH_params = {"booster":"gbtree","objective": "multi:softprob","max_depth": 5, "eta": 0.1, "subsample": 0.9, "colsample_bytree": 0.8, "eval_metric":"mlogloss","nTrees": 150,"num_class": 3,
        "rebinDistributions": [{"func": twoFaced,"label": "twoFaced","range": [-1.0,1.0]} ,{"func": lambda x: x[:,0], "label": "node_0"}, {"func": lambda x: x[:,1], "label": "node_1"}, {"func": lambda x: x[:,2], "label": "node_2"}] }

#params_6binHF = {"booster":"gbtree","objective": "multi:softprob","max_depth": 5, "eta": 0.1, "subsample": 0.9, "colsample_bytree": 0.8, "eval_metric":"mlogloss","nTrees": 150,"num_class": 6 }
params_6binHF = {"booster":"gbtree","objective": "multi:softprob","max_depth": 5, "eta": 0.02, "subsample": 0.75, "colsample_bytree": 0.8, "eval_metric":"mlogloss","nTrees": 250,"num_class": 6 }

VH_params = {"booster":"gbtree","objective": "binary:logistic","max_depth": 5, "eta": 0.1, "subsample": 0.9, "colsample_bytree": 0.8, "eval_metric":"logloss","nTrees": 200,
        "rebinDistributions": [{"func": lambda x: x,"label": "identity","range": [0,1.0]} ]} 


#"objective": "binary:logistic","reg:linear" #"multi:softprob"

VZVH_0lep_2017_odd = {
    "name": "VZVH_0lep_2017_odd",
    "features": features_0lep,
    "files": files,
    "selection": "((Pass_nominal&&controlSample==0&&(isZnn)))",
    "trainSelection": trainOnOdd,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}

VZVH_0lep_2017_even = {
    "name": "VZVH_0lep_2017_even",
    "features": features_0lep,
    "files": files,
    "selection": "((Pass_nominal&&controlSample==0&&(isZnn)))",
    "trainSelection": trainOnEven,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}

VZVH_1lep_2017_odd = {
    "name": "VZVH_1lep_2017_odd",
    "features": features_1lep,
    "files": files,
    "selection": "((V_pt>150&&Pass_nominal&&controlSample==0&&(isWenu||isWmunu)))",
    "trainSelection": trainOnOdd,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}
VZVH_1lep_2017_even = {
    "name": "VZVH_1lep_2017_even",
    "features": features_1lep,
    "files": files,
    "selection": "((V_pt>150&&Pass_nominal&&controlSample==0&&(isWenu||isWmunu)))",
    "trainSelection": trainOnEven,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}

VZVH_2leplow_2017_odd = {
    "name": "VZVH_2leplow_2017_odd",
    "features": features_0lep,
    "files": files,
    "selection": "((V_pt>75&&V_pt<150&&Pass_nominal&&controlSample==0&&(isZee||isZmm)))",
    "trainSelection": trainOnOdd,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}
VZVH_2leplow_2017_even = {
    "name": "VZVH_2leplow_2017_even",
    "features": features_0lep,
    "files": files,
    "selection": "((V_pt>75&&V_pt<150&&Pass_nominal&&controlSample==0&&(isZee||isZmm)))",
    "trainSelection": trainOnEven,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}

VZVH_2lephigh_2017_odd = {
    "name": "VZVH_2lephigh_2017_odd",
    "features": features_0lep,
    "files": files,
    "selection": "((V_pt>150&&Pass_nominal&&controlSample==0&&(isZee||isZmm)))",
    "trainSelection": trainOnOdd,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}

VZVH_2lephigh_2017_even = {
    "name": "VZVH_2lephigh_2017_even",
    "features": features_0lep,
    "files": files,
    "selection": "((V_pt>150&&Pass_nominal&&controlSample==0&&(isZee||isZmm)))",
    "trainSelection": trainOnEven,
    "labeling": VZVH_labeling,
    "param": VZVH_params
}
VH_1lep_2017_noBtag = {
    "name": "VH_1lep_2017_nobtag",
    "features": features_1lep_nobtag,
    "files": files,
    #"files": "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_22jan23/haddjobs_noSA5/*WplusH*",
    "selection": "((V_pt>150&&Pass_nominal&&controlSample==0&&(isWenu||isWmunu)))",
    "trainSelection": trainOnOdd,
    "labeling": isVH,
    "param": VH_params
}

testFiles = [path+"/TT_SingleLep_3.root", path+"/WJets_2J_nlo_5.root"]
VH_1lep_med_HF_2017_asym = {
    "name": "VH_1lep_med_HF_2017_asym",
    "features": features_1lep_asym,
    "files": files,
    #"files": "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_22jan23/haddjobs_noSA5/*WplusH*",
    "selection": "((150<V_pt&&V_pt<250&&Pass_nominal&&controlSample==13&&(isWenu||isWmunu)))",
    "trainSelection": trainOnOdd,
    "labeling": HF_6bin_labeling,
    "catNames":  ["V+LF","V+c","V+1b","V+2b","ST","TT"],
    "param": params_6binHF,
    "defs": {"lepSignPt":"float lepSignPt = 0.0; if(isWenu) lepSignPt=Electron_charge[lepInd1]*Electron_pt[lepInd1]; if(isWmunu) lepSignPt=Muon_charge[lepInd1]*Muon_pt[lepInd1]; return lepSignPt", "lepEta": "float lepEta = 0.0; if(isWenu) lepEta=Electron_eta[lepInd1]; if(isWmunu) lepEta=Muon_eta[lepInd1]; return lepEta;"},
    "custom": [{"func": getMaxInd,"label": "maxInd","binning": [6,0,6]}] #,{"func": getMaxIndPlusScore,"label": "maxIndPlusScore","binning": [12,0,6]}],
}
VH_1lep_med_HF_2017 = {
    "name": "VH_1lep_med_HF_2017",
    "features": features_1lep,
    "files": files,
    #"files": "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_22jan23/haddjobs_noSA5/*WplusH*",
    "selection": "((150<V_pt&&V_pt<250&&Pass_nominal&&controlSample==13&&(isWenu||isWmunu)))",
    "trainSelection": trainOnOdd,
    "labeling": HF_6bin_labeling,
    "catNames":  ["V+LF","V+c","V+1b","V+2b","ST","TT"],
    "param": params_6binHF,
    "defs": [],
    "custom": [{"func": getMaxInd,"label": "maxInd","binning": [6,0,6]},
        {"func": getMaxIndPlusScore,"label": "maxIndPlusScore","binning": [30,0,6]},
        {"func": getMaxIndPlusScoreGt0p3 ,"label": "maxIndPlusScoreGt0p3","binning": [12,0,6]}
        ],
}
VH_1lep_med_MF_2017 = {
    "name": "VH_1lep_med_MF_2017",
    "features": features_1lep,
    "files": files,
    #"files": "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_22jan23/haddjobs_noSA5/*WplusH*",
    "selection": "((150<V_pt&&V_pt<250&&Pass_nominal&&controlSample==16&&90<H_mass&&H_mass<150&&(isWenu||isWmunu)))",
    "trainSelection": trainOnOdd,
    "labeling": HF_6bin_labeling,
    "catNames":  ["V+LF","V+c","V+1b","V+2b","ST","TT"],
    "param": params_6binHF,
    "defs": [],
    "custom": [{"func": getMaxInd,"label": "maxInd","binning": [6,0,6]},
        {"func": getMaxIndPlusScoreGt0p3 ,"label": "maxIndPlusScoreGt0p3","binning": [12,0,6]}
    ],
}
VH_1lep_high_MF_2017 = {
    "name": "VH_1lep_high_MF_2017",
    "features": features_1lep,
    "files": files,
    #"files": "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbb_Sep2022_syncs/vhbb_2017_22jan23/haddjobs_noSA5/*WplusH*",
    "selection": "((V_pt>250&&Pass_nominal&&controlSample==16&&90<H_mass&&H_mass<150&&(isWenu||isWmunu)))",
    "trainSelection": trainOnOdd,
    "labeling": HF_6bin_labeling,
    "catNames":  ["V+LF","V+c","V+1b","V+2b","ST","TT"],
    "param": params_6binHF,
    "defs": [],
    "custom": [{"func": getMaxInd,"label": "maxInd","binning": [6,0,6]},
        {"func": getMaxIndPlusScoreGt0p3 ,"label": "maxIndPlusScoreGt0p3","binning": [12,0,6]}
    ],
}


configs = [VH_1lep_high_MF_2017]

