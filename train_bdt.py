import ROOT
import numpy as np
import pdb
import xgboost as xgb
import matplotlib.pyplot as plt
import glob
import os
from combine import SFmodel

from config import configs

print("XGBoost version {}".format(xgb.__version__))

def to_numpy(df,cut,features,label):
    columns = df.GetColumnNames()
    if label not in columns:
        df = df.Define("label",label)
    else:
        print("label {} already defined".format(label))
    for feature_wdef in features: 
        feature_split = feature_wdef.split(":")
        if len(feature_split)==1:
            feature = feature_split[0]
            feature_def = None
        else:
            feature = feature_split[0]
            feature_def = feature_split[1]

        if feature not in columns:
            print("Missing {}, setting to -99".format(feature))
            df = df.Define(feature,-99)
    #df = df.Define("weight_noStitch","weight/VJetsNLOStitchingWeight")
    print("filtering") 
    array_dict=df.Filter(cut).AsNumpy(list(features.keys())+["weight"]+["label"])
    #labels = np.zeros(len(array_dict[list(features.keys())[0]]))
    #labels[:] = labelk
    print("stacking")
    return np.vstack([array_dict[var] for var in features.keys()+["label"]]).T,array_dict["weight"].T


#def makeDMatrixFile(name,files,selection,trainSelection,features,labeling,defs):
def makeDMatrixFile(config):
    name=config["name"]
    files=config["files"]
    selection=config["selection"]
    trainSelection=config["trainSelection"]
    features=config["features"]
    labeling=config["labeling"]
    defs=config["defs"]


    ROOT.EnableImplicitMT(4)
    #print(files)
    print(selection)
    print(trainSelection)
    print(labeling)
    df = ROOT.RDataFrame("Events",files)
    for definition in defs:
        df = df.Define(definition,defs[definition])

    train,train_w = to_numpy(df,"({})&&({})".format(selection,trainSelection),features,labeling)
    test,test_w = to_numpy(df,"({})&&!({})".format(selection,trainSelection),features,labeling)
    
    print("shuffling training data")
    np.random.seed(42)
    np.random.shuffle(train)
    np.random.seed(42)
    np.random.shuffle(train_w)

    
    dtrain = xgb.DMatrix(train[:,:-1],train[:,-1],feature_names=features,weight=abs(train_w))
    dtest = xgb.DMatrix(test[:,:-1],test[:,-1],feature_names=features,weight=abs(test_w))
    cats = list(set(dtrain.get_label()))

    for i,feature in enumerate(features):
        binning=np.linspace(features[feature][1],features[feature][2],features[feature][0]+1)
        plt = plotTestAndTrain(name,feature, train[:,i].astype("float32"),dtrain,test[:,i].astype("float32"),dtest,binning,cats,returnPlot=True)
        plt.xlabel(feature)
        plt.yscale("linear")
        plt.savefig("{}/{}.png".format(name,feature))
        plt.clf()

    #dtrain = xgb.DMatrix(train[:,:-1],train[:,-1],feature_names=features,weight=train_w)
    #dtest = xgb.DMatrix(test[:,:-1],test[:,-1],feature_names=features,weight=test_w)
    dtrain.save_binary(name+"/train.dmx")
    dtest.save_binary(name+"/test.dmx")

def computeAsimov(cats,dmx,scores,binning,bkgLabel):
    tmp=ROOT.RooStats.AsimovSignificance(1,1,1)
    hists = {}
    for cat in cats:
        hists[cat] = {}
        #trainIsCat = dtrain.get_label()==cat
        testIsCat = dmx.get_label()==cat
        hists[cat]["w"],binning = np.histogram(scores[testIsCat],weights=dmx.get_weight()[testIsCat],bins=binning)
        hists[cat]["w2"],binning = np.histogram(scores[testIsCat],weights=dmx.get_weight()[testIsCat]**2,bins=binning)
    
    catSignificances = [] 
    for cat in cats:
        if cat!=bkgLabel:
            bkg = hists[bkgLabel]["w"]
            bkgErr = hists[cat]["w2"]**0.5
            s=hists[cat]["w"]
            sErr=hists[cat]["w2"]**0.5
             
            catSignificance = sum([ROOT.RooStats.AsimovSignificance(s_i,b_i,bErr_i)**2 for s_i,b_i,bErr_i in zip(s,bkg,bkgErr)])**0.5
            catSignificance = np.nan_to_num(catSignificance)
            catSignificances.append(catSignificance)
    
    #get bkg effective entries per bin
    if any(hists[bkgLabel]["w2"]<=0) or any(np.nan_to_num(hists[bkgLabel]["w"]**2/hists[bkgLabel]["w2"])<5):
        return -1
    else:
        return catSignificances

def plotTestAndTrain(name,plot_name,train_scores,dtrain,test_scores,dtest,binning,cats,transform=lambda x:x,returnPlot=False):  
    
    trainDist = transform(train_scores)
    testDist = transform(test_scores)
    for cat,color in zip(cats,["blue","orange","green","fuchsia","cyan","gold"]):
        trainIsCat = dtrain.get_label()==cat
        testIsCat = dtest.get_label()==cat
        plt.hist(trainDist[trainIsCat],weights=dtrain.get_weight()[trainIsCat],bins=binning,label="train {}".format(cat),histtype='stepfilled',density=False,alpha=0.3,color=color)
        plt.hist(testDist[testIsCat],weights=dtest.get_weight()[testIsCat],bins=binning,label="test {}".format(cat),histtype='step',density=False,color=color,linestyle='dashed')
        #plt.yscale("log")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    #plt.legend(loc="upper right")
    if not returnPlot:
        plt.savefig("{}/{}.png".format(name,plot_name))
        plt.clf()
    else:
        return plt


def confusionMatrix(name,plot_name,predictedLabels,trueLabels,weights,labels):
    ax = plt.subplot()

    hist,xbins,ybins,im = ax.hist2d(predictedLabels,trueLabels,weights=weights,bins=len(labels),range=[[0,len(labels)],[0,len(labels)]])
    plt.clf()
    ax = plt.subplot()
    #normalize by row (i.e. what percent of ttbar is classified as...)
    normalizedHist = (hist/np.sum(hist,axis=0)).T
    ax.imshow(normalizedHist,origin='lower')

    #pdb.set_trace()
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            #ax.text(xbins[j]+0.5,ybins[i]+0.5, "{:.2f}%".format(hist.T[i,j]),color="w", ha="center", va="center", fontweight="bold")
            ax.text(xbins[j],ybins[i], "{:.1f}%".format(100.0*normalizedHist[i,j]),color="w", ha="center", va="center", fontweight="bold")

    ax.set_xticks([x for x in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_yticks([x for x in range(len(labels))])
    ax.set_yticklabels(labels)
    #ax.xticks(ticks=[.5+x for x in range(len(labels)-1)],labels=labels)
    plt.xlabel("predicted")
    plt.ylabel("target")

    plt.savefig("{}/{}.png".format(name,plot_name))
    plt.clf()

#def trainModel(name,param):
def trainModel(config):
    name=config["name"]
    param=config["param"]
    print("training")
    print("loading train.dmx")
    dtrain = xgb.DMatrix(name+"/train.dmx")
    
    #balance classes
    print("balancing classes")
    weights = dtrain.get_weight()
    cats = list(set(dtrain.get_label()))
    labels = dtrain.get_label()
    class_biases = param["class_bias"] if "class_bias" in param else [1.0]*len(labels)
    
    for i,cat in enumerate(cats):
        integral = sum(weights[labels==cat])

        scale = 1.0/integral
        weights[labels==cat] = scale*weights[labels==cat]*1000*class_biases[i]
        #weights[labels==cat] = 1e-8
    dtrain.set_weight(weights)

    watchlist = [(dtrain,"train")]
    #pdb.set_trace()
    bdt = xgb.train(param,dtrain,num_boost_round=param["nTrees"],evals=watchlist)
    bdt.save_model("{}/{}.bin".format(name,name))

    return bdt

def validate(bdt,config):
    name=config["name"]
    #bdt=config["bdt"]
    param=config["param"]
    features=config["features"]
    selection=config["selection"]
    trainSelection=config["trainSelection"]
    labeling=config["labeling"]
    files=config["files"]
    defs=config["defs"]

    print("validating")
    dtrain = xgb.DMatrix(name+"/train.dmx")
    dtest = xgb.DMatrix(name+"/test.dmx")

    train_scores = bdt.predict(dtrain)
    test_scores = bdt.predict(dtest)
    dtrain.feature_names = features.keys()
    dtest.feature_names = features.keys()
    bdt.feature_names = features.keys()

    cats = list(set(dtrain.get_label()))
    
    if train_scores.ndim==1:
        xgb.plot_importance(bdt,grid=False)
        plt.savefig(name+"/feature_ranking.png")
        plt.clf()
        plotTestAndTrain(name,"node",train_scores,dtrain,test_scores,dtest,np.linspace(0,1,51),cats)
                    
    else:
        num_outputs = train_scores.shape[1]
        for i_output in range(num_outputs):
            plotTestAndTrain(name,"node_{}".format(i_output),train_scores[:,i_output],dtrain,test_scores[:,i_output],dtest,np.linspace(0,1,51),cats)
        confusionMatrix(name,"confusion",np.argmax(test_scores,axis=1),dtest.get_label(),dtest.get_weight(),config["catNames"])
    #pdb.set_trace()
    if "custom" in config and config["custom"]:
        for customDist in config["custom"]:
            bins,xmin,xmax=customDist["binning"]
            plotTestAndTrain(name,customDist["label"],train_scores,dtrain,test_scores,dtest,np.linspace(xmin,xmax,bins+1),cats,transform=customDist["func"])
            print(customDist["label"]) 
            #asimov significance of each process


            model=SFmodel(name)
            for cat,catname in zip(cats,config["catNames"]):
                testIsCat=dtest.get_label()==cat
                transformedScores = customDist["func"](test_scores)
                
                sig = transformedScores[testIsCat]
                bkg = transformedScores[~testIsCat]
                h_sig,binning = np.histogram(transformedScores[testIsCat],weights=dtest.get_weight()[testIsCat],bins=np.linspace(xmin,xmax,bins+1))
                h_sig_w2,binning = np.histogram(transformedScores[testIsCat],weights=dtest.get_weight()[testIsCat]**2,bins=np.linspace(xmin,xmax,bins+1))
                h_bkg,binning = np.histogram(transformedScores[~testIsCat],weights=dtest.get_weight()[~testIsCat],bins=np.linspace(xmin,xmax,bins+1))
                h_bkg_w2,binning = np.histogram(transformedScores[~testIsCat],weights=dtest.get_weight()[~testIsCat]**2,bins=np.linspace(xmin,xmax,bins+1))
                #print("{} asimov: {:.2f}".format(cat,sum(h_sig**2/h_bkg)**0.5))
                
                catSignificance = sum([ROOT.RooStats.AsimovSignificance(s_i,b_i,bErr_i)**2 for s_i,b_i,bErr_i in zip(h_sig,h_bkg,(h_bkg_w2+h_sig_w2)**0.5)])**0.5

                model.AddSample(str(catname),h_sig,h_sig_w2)
            model.Save(name+"/"+customDist["label"]+".json")
                

    if "rebinDistributions" in param:
        for rebinDistribution in config["rebinDistributions"]:
            #split bins in half until significance stops improving or below 10 effective entries in bkg
            binning = [0,1.0]
            if "range" in rebinDistribution:
                binning = rebinDistribution["range"]
            
            asimovs = np.array([1e-7]*(len(cats)-1))
            #assume largest class is bkg, rest is signal         
            bkgLabel = cats[0]
            for cat in cats:
                trainIsCat = dtrain.get_label()==cat
                trainIsBkg = dtrain.get_label()==bkgLabel
                if dtrain.get_weight()[trainIsCat].sum()>dtrain.get_weight()[trainIsBkg].sum():
                    bkgLabel=cat
            print("treating {} as bkg, rest as signal".format(bkgLabel))
            
            scores = rebinDistribution["func"](test_scores)

            updatedBinning=True
            while updatedBinning: 
                updatedBinning=False

                for bin_idx in range(len(binning)-1):
                    newBinning = np.insert(binning,[bin_idx+1],(binning[bin_idx]+binning[bin_idx+1])/2.0)
                        
                    newAsimovs = np.array(computeAsimov(cats,dtest,scores,newBinning,bkgLabel))
                    asimovs=np.array(asimovs)
                    if all(newAsimovs>=asimovs) and any(newAsimovs/asimovs>=1.01):
                        binning=newBinning
                        asimovs=newAsimovs
                        updatedBinning=True
            print(name,rebinDistribution["label"],binning,asimovs)

            plot = plotTestAndTrain(name,"rebinned",train_scores,dtrain,test_scores,dtest,binning,cats,transform=rebinDistribution["func"],returnPlot=True)
            plot.title(name+" " + ",".join(["{:.2f}".format(asimov) for asimov in asimovs]))
            plot.xlabel("bdt score")
            plot.savefig("{}/{}_rebinned.png".format(name,rebinDistribution["label"]))
            plot.clf()
    
    if "rwtVarswBDT" in config:
        rwtVars = config["rwtVarswBDT"]

        #bdtRwtWeights = test_scores/(1.0-test_scores)
        #if no numpy file w/aux vars, make it
        if not os.path.isfile("{}/testAux.csv".format(name)) or any(rwtVar not in np.genfromtxt(name+"/testAux.csv",names=True).dtype.names for rwtVar in rwtVars):
            print("making testAux.csv")
            df = ROOT.RDataFrame("Events",files)
            for definition in defs:
                df = df.Define(definition,defs[definition])
            allVars = {}
            allVars.update(features)
            allVars.update(rwtVars)
            testAux,testAux_w = to_numpy(df,"({})&&!({})".format(selection,trainSelection),allVars,labeling)
            testAux_w.shape = (-1,1)
            testAux = np.hstack([testAux,testAux_w]) 
            np.savetxt(name+"/testAux.csv",testAux,header=" ".join(list(allVars)+["label"]+["weight"]))
       
            


        testAux = np.genfromtxt(name+"/testAux.csv",names=True)
        feature_array = np.vstack([testAux[feature] for feature in bdt.feature_names]).T
        #pdb.set_trace()
        dtestaux = xgb.DMatrix(feature_array,testAux["label"],feature_names=bdt.feature_names,weight=abs(testAux["weight"]))
        rwt_scores = bdt.predict(dtestaux)
        bdtRwtWeights = rwt_scores/(1.0-rwt_scores)

        for rwtVar in rwtVars:
            binning = np.linspace(rwtVars[rwtVar][1],rwtVars[rwtVar][2],rwtVars[rwtVar][0]+1)
            for i,cat in enumerate(cats):
                #assume first is the category to be reweighted
                
                testIsCat = dtestaux.get_label()==cat
                plt.hist(testAux[rwtVar][testIsCat],weights=dtestaux.get_weight()[testIsCat],bins=binning,label="{}".format(cat),histtype='step',density=True)
                if i==0: 
                #    print(i,cat)
                    plt.hist(testAux[rwtVar][testIsCat],weights=dtestaux.get_weight()[testIsCat]*bdtRwtWeights[testIsCat],bins=binning,label="{} rwt'd".format(cat),histtype='step',density=True)
            plt.xlabel(rwtVar)
            plt.legend()
            plt.savefig(name+"/"+rwtVar+"_reweighted.png")
            plt.clf()

            



def train_bdt(config):
    name = config["name"]
    files = config["files"]
    selection = config["selection"]
    trainSelection = config["trainSelection"]
    features = config["features"]
    labeling = config["labeling"]
    param = config["param"]
    if "defs" in config:
        defs = config["defs"]
    else:
        defs = {}

    try:
        os.mkdir(name)
    except:
        pass

    if not (os.path.isfile(name+"/train.dmx") and os.path.isfile(name+"/test.dmx")):
        print("skimming data")
        #makeDMatrixFile(name,files,selection,trainSelection,features,labeling,defs)
        makeDMatrixFile(config)
    
    if not os.path.isfile("{}/{}.bin".format(name,name)):
        #bdt = trainModel(name,param)
        bdt = trainModel(config)
    else:
        bdt = xgb.Booster()
        bdt.load_model("{}/{}.bin".format(name,name))

    #validate(name,bdt,param,features)
    validate(bdt,config)


for config in configs:
    train_bdt(config)

