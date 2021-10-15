# PhoneGraphs

This repository contains code for in-progress work on using graph-representations of phoneme distributions to learn phonological class systems, and then evaluating those phonological class systems using a new variant of Maximum Entropy phonotactic grammars that fits the MaxEnt Grammar to an n-gram model. 

This repository is made public to be used by other researchers, but is still very much in progress. This is very much research code and is not optimized for either legibility or efficiency - a more efficient and user friendly version will be posted in the near future. 

After cloning the repository, a minimal working example can be run with:

`Run_Template.sh`

This template file uses relative paths, so it will run from the `expts` directory. The example learns a class system from the token frequencies of the English Onset corpus used by Hayes-Wilson (2008), builds and fits a MaxEnt n-gram phonotactic grammar to the same corpus, and then uses the resulting grammar to make well-formedness judgements of the test forms from Daland et al. (2011)'s human experiments. It then reports the Kendall's Tau and Pearson's r between the model's judgments and Daland et al.'s aggregated human well-formedness scores. The specifics of the class learning algorithm and phonotactic model used in the example can be seen in `expts/confs/English_SC_onsets.config`. This example uses spectral clustering on the phonological environment graph. Kendall's Tau with human judgements should range between 0.70 and 0.75, and Pearson's r should range between 0.775 and 0.825 in this example. 



