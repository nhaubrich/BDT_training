import json
import pdb
import numpy as np

class SFmodel:
    def __init__(self,channel):
        self.model = {}
        SFmeasurement = {
            "name": "SFs", "config": {
            "poi": "lumi",
            "parameters": [
                {"name": "lumi", "auxdata": [1.0], "bounds": [ [.9,1.1]], "inits": [1.0], "sigmas": [.03] }
            ]
            }
        }
        self.model["measurements"] = [SFmeasurement]
        self.model["channels"] = [{ "name": channel, "samples": []} ]
        self.model["version"]= "1.0.0"
        return


    def AddSample(self,name,w,w2,scale=1):
        weights=np.array(w)
        weights_sq=np.array(w2)
        weights=weights*scale
        err=np.sqrt(weights_sq)*scale

        sample={
            "name": name,
            "data": weights.tolist(),
            "modifiers": [
                {"name": "lumi", "type": "lumi", "data": None},
                {"name": "SF_"+name,  "type": "normfactor", "data": None, },
                {"name": "MCStat", "type": "staterror", "data": err.tolist()}
            ]
        }
        self.model["channels"][0]["samples"].append(sample)
        return

    def Save(self,fout):
        expected = [  sample["data"] for sample in self.model["channels"][0]["samples"] ]
        expected = np.array(expected).sum(axis=0).tolist()

        obs = {"name": self.model["channels"][0]["name"], "data": expected }
        self.model["observations"]=[obs]

        with open(fout,'w') as f:
            json.dump(self.model, f)#, ensure_ascii=False, indent=4)

        #compute asimov
        return

if __name__=="__main__":
    #model = SFmodel()
    #model.AddSample("TT",[10,4],[3,2])
    #model.Save("model.json")

    import json
    import sys
    import pyhf

    backend_name = "numpy"
    #pyhf.set_backend(backend_name)
    pyhf.set_backend("numpy", "minuit")
    #print(f"Tensor Lib: {pyhf.tensorlib}")
    #print(f"Optimizer:  {pyhf.optimizer}")

    #with open("model.json") as input_file:
    if len(sys.argv)==2:
        with open(sys.argv[1]) as input_file:
            workspace = pyhf.Workspace(json.load(input_file))
    
    elif len(sys.argv)>2:
        with open(sys.argv[1]) as input_file:
            ws1 = pyhf.Workspace(json.load(input_file))
        with open(sys.argv[2]) as input_file:
            ws2 = pyhf.Workspace(json.load(input_file))
        workspace = pyhf.Workspace.combine(ws1,ws2,join="outer")
    
    model = workspace.model(measurement_name="SFs")
    data = workspace.data(model)

    result = pyhf.infer.mle.fit(data, model, return_uncertainties=True)
    bestfit, errors = result.T

    pars = model.config.par_order
    if "MCStat" in pars:
        pars=pars[:-1]+["MCStat"+str(i) for i in range(1+len(bestfit)-len(pars))]

    for par,val,err in zip(pars,bestfit,errors):
        if "MCStat" not in par:
            print("{}: {:.2f}+/-{:.2f}".format(par,val,err))
