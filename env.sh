#for use with combine.py to approximate asimov scale factors
#first, scl enable rh-python38 bash (for lxplus)
#then install and enter environment with script
#then, python combine.py [bdt]/[output].json

python3 -m venv pyhfenv
. pyhfenv/bin/activate
python3 -m pip install --upgrade pip
#python3 -m pip install --force-reinstall -v "xgboost==0.82"
python3 -m pip install pyhf
python3 -m pip install iminuit
##python3 -m pip install --force-reinstall -v "pyhf>=0.7.0"
python3 -m pip install matplotlib


