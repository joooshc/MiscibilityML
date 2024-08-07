import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

unscaled = pd.read_csv("Datasets/master_unscaled.csv")
scaled_Train = pd.read_csv("Datasets/TrainTestData/NoDragonYx/QuantileTrain.csv")
scaled_Test = pd.read_csv("Datasets/TrainTestData/NoDragonYx/QuantileTest.csv")
scaled = pd.concat([scaled_Train, scaled_Test])

unscaled_features = ["MoleFraction"]
scaled_features = ["MoleFraction"]

plt.style.use("seaborn-v0_8-colorblind")

def plotter(features, isScaled):
    for feature in features:
        plt.figure(figsize=(16, 7))

        if isScaled:
            plt.hist(scaled[f"{feature}"], bins=20, label=f"{feature} (Scaled)")
            plt.title(f"Log Scaled {feature} Distribution", fontsize=35)
        else:
            plt.hist(unscaled[f"{feature}"], bins=30, label=f"{feature} (Unscaled)")
            plt.title(f"Unscaled {feature} Distribution", fontsize=35)

        plt.xlabel(feature, fontsize=25)
        plt.ylabel("Frequency Density", fontsize=25)
        plt.yticks(fontsize=20); plt.xticks(fontsize=20)
        plt.tight_layout()

        plt.savefig("Results/PlotsForReport/Hist" + feature + "Scaled.png")
        plt.show()

# plotter(unscaled_features, False)
# plotter(scaled_features, True)

myGrey = "#3b4649"
myCyan = "#01e99c"

fig, ax1, = plt.subplots(layout="constrained", figsize=(16, 9), dpi = 300)
comp1 = unscaled["MoleFraction"]
comp1 = comp1[comp1 < 1]
comp1 = comp1[comp1 > 0]

comp2 = scaled["MoleFraction"]
ax1.hist(comp1, bins=20, label="Mole Fraction (Unscaled)", color=myCyan, alpha=1)
ax1.hist(comp1, bins=20, color= "black", alpha=1, histtype="step")

ax1.set_xlabel("Mole Fraction", fontsize=25)
ax1.set_ylabel("Frequency Density", fontsize=25)
ax1.tick_params(axis="both", labelsize=15)
ax1l = mpatches.Patch(color=myCyan, alpha=1, label="Mole Fraction (Unscaled)")

ax2 = ax1.twiny()
ax2.hist(comp2, bins=20, label="Mole Fraction (Log Scaled)", color=myGrey, alpha=0.8)
ax2.hist(comp2, bins=20, color="black", alpha=1, histtype="step")

ax2.set_xlabel("Mole Fraction (Log Scaled)", fontsize=25)
ax2.tick_params(axis="x", labelsize=15)
ax2l = mpatches.Patch(color=myGrey, alpha=0.8, label="Mole Fraction (Log Scaled)")

plt.title("Mole Fraction Distribution", fontsize=35)
plt.legend(handles=[ax1l, ax2l], fontsize=20)
plt.savefig("Results/PlotsForReport/DualMoleFractionHist2.png")
plt.show()