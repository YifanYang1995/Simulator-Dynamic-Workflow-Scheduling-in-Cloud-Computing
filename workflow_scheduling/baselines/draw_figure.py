import matplotlib.pylab as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def trainCurve(logbook):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    # 方fig,式2：通过logbook
    gen = logbook.select("gen")
    fits=logbook.chapters["fitness"].select("min")
    
    
    axs.plot(gen, fits)

    plt.title("Covergence Curves on Training Instances")
    plt.savefig("./Saved_Results_/train.pdf",bbox_inches="tight")


def testCurve(Results):
    # Results.shape = sceNum, runNum, rule_gen, test_evalNum
    p = list(Results.shape)

    if p[0]==1:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    else:
        n = int(np.ceil(p[0]/2))
        fig, axs = plt.subplots(nrows=2, ncols= n , figsize=(4*n, 6))

    
    if p[2] ==1:  # choose the best individual in each generation 
        Results = Results.reshape((p[0],p[1],p[3]))
        gen = np.arange(p[1])
        gens = np.tile(A=gen, reps=(p[3], 1)).T

        # dataRe=None
        for i in range(p[0]):
            fits = Results[i]
            df = pd.DataFrame({'gen': np.hstack(gens), 'fit': np.hstack(fits)})
            df['method'] = ['Scenario_'+str(i) for _ in range(df.shape[0])]
            # dataRe = pd.concat([dataRe, df])

            sns.lineplot(x="gen", y="fit", hue='method', data=df,  ax = axs[i//axs.shape[1], i%axs.shape[1]])
            # axs[i//axs.shape[1], i%axs.shape[1]].get_legend(fontsize=font_size)
    

    fig.suptitle("Test performance of training stage")
    plt.savefig("./Saved_Results_/test.pdf",bbox_inches="tight")    
    # plt.show()