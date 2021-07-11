import det.detSobel as ds 
import stoch.stochWrapper as sw
import glob
import numpy as np

fileList = glob.glob("DEFINE YOUR PATH TO PICTURES TO PROCESS HERE")

print("Deterministic processing")
for file in fileList:
   print("Processing: ",file)
   src,edges_det = ds.detectAndShow(file,errRate=0.00,show=False)
   sw.saveHex("./results/det/noErr/"+file.split('/')[-1][:-4]+".txt",edges_det)

print("Stochastic processing")
for file in fileList:
   print("Processing: ",file)
   src,edges_stoch = sw.detectAndShow(file,errRate=0.00,show=False)
   sw.saveHex("./results/stoch/noErr/"+file.split('/')[-1][:-4]+".txt",edges_stoch)

## Calculate metrics and write in files
for proc_type in ('det', 'stoch'):
   resultList = glob.glob("./results/"+proc_type+"/*.txt")
   resultList = list(np.sort(resultList))
   meanAbsError = []
   meanSquaredError = []
   for errRate in ('0_01', '0_02', '0_05'):
      for result in resultList:
         reference = np.float64(sw.loadHex(result))
         afterError = np.float64(sw.loadHex(result.replace(proc_type,""+proc_type+"/errRate/"+errRate)))
         meanAbsError.append(np.abs(np.subtract(reference,afterError)).mean())
         meanSquaredError.append(np.square(np.subtract(reference,afterError)).mean())

      for errMetric in ('MAE', 'MSE') :
         f = open("./results/result"+proc_type+errRate+errMetric".txt","w")

         f.write(errMetric+":\n")
         if errMetric == 'MAE':
            for r in meanAbsError:
               f.write(str(r)+"\n")
            f.write("Mean:\n" + str(np.mean(meanAbsError))+"\n")

         if errMetric == 'MSE':
            f.write("MSE:\n")
            for r in meanSquaredError:
               f.write(str(r)+"\n")
            f.write("Mean:\n" + str(np.mean(meanSquaredError))+"\n")

         f.close()

# import code
# code.interact(local=locals())