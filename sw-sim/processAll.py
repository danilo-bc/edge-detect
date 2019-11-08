import det.detSobel as ds 
import stoch.stochWrapper as sw
import glob
import numpy as np
import ray

ray.init()

fileList = glob.glob("/mnt/hdd1tera/databaseDanilo/IEEE-CURE-OR/10_grayscale_no_challenge/texture1/LG/*.jpg")

print("Parte deterministica")
for file in fileList:
   print("Processando: ",file)
   src,edges_det = ds.detectAndShow(file,errRate=0.00,show=False)
   sw.saveHex("./results/det/noErr/"+file.split('/')[-1][:-4]+".txt",edges_det)

print("Parte estocastica")
for file in fileList:
   print("Processando: ",file)
   src,edges_stoch = sw.rayDetectAndShow(file,errRate=0.00,show=False)
   sw.saveHex("./results/stoch/NoErr/"+file.split('/')[-1][:-4]+".txt",edges_stoch)
   # sw.saveHex("./results/stoch/errRate/"+file.split('/')[-1][:-4]+"0_05.txt",edges_stoch)


# resultList = glob.glob("./results/det/*.txt")
# resultList = list(np.sort(resultList))
# meanAbsError = []
# for result in resultList:
#    detR = np.float64(sw.loadHex(result))
#    stochR = np.float64(sw.loadHex(result.replace("det","stoch")))
#    meanAbsError.append(np.abs(np.subtract(detR,stochR)).mean())

# resultList = glob.glob("./results/det/*.txt")
# resultList = list(np.sort(resultList))
# meanAbsError = []
# meanSquaredError = []
# for result in resultList:
#    reference = np.float64(sw.loadHex(result))
#    afterError = np.float64(sw.loadHex(result.replace("det","det/errRate/0_05")))
#    meanAbsError.append(np.abs(np.subtract(reference,afterError)).mean())
#    meanSquaredError.append(np.square(np.subtract(reference,afterError)).mean())


# f = open("./results/resultDet0_05.txt","w")

# f.write("MAE:\n")
# for r in meanAbsError:
#    f.write(str(r)+"\n")
# f.write("Média:\n" + str(np.mean(meanAbsError))+"\n")

# f.write("MSE:\n")
# for r in meanSquaredError:
#    f.write(str(r)+"\n")
# f.write("Média:\n" + str(np.mean(meanSquaredError))+"\n")
# f.close()

# resultList = glob.glob("./results/stoch/*.txt")
# resultList = list(np.sort(resultList))
# meanAbsError = []
# meanSquaredError = []
# for result in resultList:
#    reference = np.float64(sw.loadHex(result))
#    afterError = np.float64(sw.loadHex(result.replace("stoch","stoch/errRate/0_05")))
#    meanAbsError.append(np.abs(np.subtract(reference,afterError)).mean())
#    meanSquaredError.append(np.square(np.subtract(reference,afterError)).mean())


# f = open("./results/resultStoch0_05.txt","w")
# f.write("MAE:\n")
# for r in meanAbsError:
#    f.write(str(r)+"\n")
# f.write("Média:\n" + str(np.mean(meanAbsError))+"\n")

# f.write("MSE:\n")
# for r in meanSquaredError:
#    f.write(str(r)+"\n")
# f.write("Média:\n" + str(np.mean(meanSquaredError))+"\n")

# f.close()


# import code
# code.interact(local=locals())