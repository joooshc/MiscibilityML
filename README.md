# Modelling Miscibility using Machine Learning

_Joshua Cheung, Daniel Nathan Tomás Cançado, Thasmia Mohamed Aleem Basha, Joanna Grundy, Samantha Pearman-Kanza and Jeremy Frey_

This work was funded by EPSRC through grants EP/X032701/1, EP/X032663/1 and EP/W032252/1 – PSDI (Physical Sciences Data Infrastructure) (PSDI) <https://www.psdi.ac.uk/>. Additionally we would like to thank Colin Bird for helpful discussions.

---

Miscibility is an important component of both solubility, and many wider applications of scientific research such as drug design, flow chemistry and a range of chemical processes. However, despite the significant volume of work that has been conducted in the solubility sphere, in particular the aqueous solubility of drug-like compounds, miscibility modelling by comparison has received very little attention. This is mainly due to the lack of available data on miscibility, and the complexity of modelling this data. To the authors knowledge this is the first attempt at modelling using machine learning for small molecule miscibility. This was achieved by building an enriched dataset from the IUPAC Solubility Data Series volumes and augmenting it with data from AqSolDB and the Dortmund Data Bank Liquid-Vapour Equilibrium dataset. A number of preparatory methods were trialled on this enriched dataset, including feature engineering and scaling methods, in addition to exploring the use of Principal Component Analysis in dimensionality reduction. Machine learning techniques were then employed on the resulting dataset, including ensemble based algorithms such as Random Forest, gradient-boosted decision trees, Support Vector Regression and Neural Network algorithms. This paper presents the initial results from this work, in addition to the database, as a useful resource for other researchers in this area. The resulting predictions were used to build phase diagrams, resulting in better predictions where there is good miscibility, and poorer predictions where behaviour is further from ideal.

---

Python 3.11.2 was used for this project, and the required libraries are listed in `requirements.txt`.