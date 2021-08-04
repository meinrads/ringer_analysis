from Gaugi import load
file = load('/home/mschefer/outputs/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et0_eta4.v1/tunedDiscr.jobID_0000.pic.gz')
print( file.keys() )
print(file['tunedData'][0].keys() )