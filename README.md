Tabular deep learning package aims to make multi-task deep learning with tabular data for genome-wide prediction over real-world genomic data: mice data, pig data, wheat data, 14-cancer microarray data, and Loblolly pine data.

There are totally eight models for the evaluation: LassoNet, TabR, TabNet, NODE, TabTransformer, FT-Transformer, AutoInt, Gandalf,SAINT. These models have been built on Pytorch successfully. Different parameter space that we have used for multi-trait GWP has been invovled in the Word document. '

(1) For the LassoNet, in colab envrionemn:


step 1: install bayesian-optimization
        %pip install bayesian-optimization omegaconf

step 2: from your own drive, commands are
        from google.colab import drive
        import os
        drive.mount()
        os.chdir()

step 3: run the file named entry.py


In other environment, please ensure you hava had the parallel computing condition, and run the files.

(1) For the LightGBM, NODE and TabNet, these projects can be runed in Anaconda or pytorch directly.

(2) For the other codes, run the main files in Anaconda or pytorch directly. 
      


BTW: 
Because the pig data is bigger than other four datasets, it is not possible for us to put all the datasets together. If you need, you can access this data by the link: https://github.com/angelYHF/Pig-data.
