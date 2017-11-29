import matplotlib.pyplot as plt
import numpy as np

looptime = np.zeros((4))
normlooptime = np.zeros((4))
problooptime = np.zeros((4))
looptimezero = np.zeros((4))
normlooptimezero = np.zeros((4))

loopone = np.array([171.576041,168.30132,180.485116,178.624507])
normloopone = np.array([50.970324,48.803164,53.069511,52.57946])
probloopone = np.array([112.979108,112.033939,119.249789,118.045243])
looptime[0] = np.average(loopone)
normlooptime[0] = np.average(normloopone)
problooptime[0] = np.average(probloopone)
looptimezero[0] = np.average(loopone)
normlooptimezero[0] = np.average(normloopone)

looptwo = np.array([101.389064,102.864322,102.90186,100.259693]) #0 much faster than 1
normlooptwo = np.array([35.327562,36.474008,36.243468,35.347569]) #0 much faster than 1
problooptwo = np.array([63.868159,60.536947,63.202502,60.707815,63.702863,61.134497,63.474063,59.214592]) #1 much faster than 0
looptwozero = np.array([98.109085,98.283958,98.376026,98.15348])
normlooptwozero = np.array([29.568583,30.137208,29.894946,30.13126])
looptime[1] = np.average(looptwo)
normlooptime[1] = np.average(normlooptwo)
problooptime[1] = np.average(problooptwo)
looptimezero[1] = np.average(looptwozero)
normlooptimezero[1] = np.average(normlooptwozero)

loopthree = np.array([77.667288,78.372585,76.933384,79.058278,78.366923,78.417778,77.184592,79.239451]) #0 much faster
normloopthree = np.array([29.709709,28.495583,28.165809,29.246459,28.708691,28.880266,28.233007,29.166226]) #0 much faster
probloopthree = np.array([43.83827,45.060523,44.050468,44.723389,45.692008,45.324111,45.414656,45.221228,47.084372,46.434274,44.9065,45.464886])
loopthreezero = np.array([69.949015,74.002972,75.639292,74.416145])
normloopthreezero = np.array([22.406116,24.369925,24.682312,24.158219])
looptime[2] = np.average(loopthree)
normlooptime[2] = np.average(normloopthree)
problooptime[2] = np.average(probloopthree)
looptimezero[2] = np.average(loopthreezero)
normlooptimezero[2] = np.average(normloopthreezero)

loopfour = np.array([68.580242,69.222209,69.385667,70.350422,70.482519,70.657469,70.369468,70.487006,70.601161,69.971224,70.329015,70.897949]) #0 much faster
normloopfour = np.array([27.588094,27.495467,27.698215,28.597373,28.099909,27.848239,27.972763,28.207945,28.000857,27.600605,28.72634,27.68006]) #0 much faster
probloopfour = np.array([36.865823,37.69657,38.276321,36.887782,37.835028,38.015401,38.56841,38.263841,38.439203,38.603008,37.864236,38.021619,38.109517,39.18051,37.779361,38.507137])
loopfourzero = np.array([64.324248,65.430555,65.640778,66.807095])
normloopfourzero = np.array([23.053919,23.171928,23.336687,23.989217])
looptime[3] = np.average(loopfour)
normlooptime[3] = np.average(normloopfour)
problooptime[3] = np.average(probloopfour)
looptimezero[3] = np.average(loopfourzero)
normlooptimezero[3] = np.average(normloopfourzero)

nproc = np.array([1,2,3,4])
looptime = looptime/looptime[0]
normlooptime = normlooptime/normlooptime[0]
problooptime = problooptime/problooptime[0]
looptimezero = looptimezero/looptimezero[0]
normlooptimezero = normlooptimezero/normlooptimezero[0]
linear = np.array([1,0.75,0.5,0.25])

plt.title('Decrease in simulation time')
plt.ylabel('Simulation time')
plt.xlabel('Number of processes')
plt.plot(nproc,looptime,'r',label='Entire loop')
plt.plot(nproc,looptimezero,'r--',label='Entire loop for process 0')
plt.plot(nproc,normlooptime,'b',label='Calculation of norm')
plt.plot(nproc,normlooptimezero,'b--',label='Calculation of norm for process 0')
plt.plot(nproc,problooptime,'g',label='Calculation of probabilities')
plt.plot(nproc,linear,'k',label='Linear decrease (for normalized graphs)')
plt.legend()
#plt.axis([1,4,0,1])
plt.show()
