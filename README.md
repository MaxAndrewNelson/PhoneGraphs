# PhoneGraphs

This repository contains code for in-progress work on using graph-representations of phoneme distributions to learn phonological class systems, and then evaluating those phonological class systems using a new variant of Maximum Entropy phonotactic grammars that fits the MaxEnt Grammar to an n-gram model. 

This repository is made public to be used by other researchers, but is still in-progress. This is very much research code and is not optimized for either legibility or efficiency - a more efficient and user friendly version will be posted in the near future. 

After cloning the repository, a minimal working example using relative filepaths can be run from the `expts` directory with `Run_Template.sh`.

The example learns a class system from the token frequencies of the English Onset corpus used by Hayes-Wilson (2008), builds and fits a MaxEnt n-gram phonotactic grammar to the same corpus, and then uses the resulting grammar to make well-formedness judgements of the test forms from Daland et al. (2011)'s human experiments. It then reports the Kendall's Tau and Pearson's r between the model's judgments and Daland et al.'s aggregated human well-formedness scores. The specifics of the class learning algorithm and phonotactic model used in the example can be seen in `expts/confs/English_SC_onsets.config`. This example uses spectral clustering on the phonological environment graph. Kendall's Tau with human judgements should range between 0.70 and 0.75, and Pearson's r should range between 0.775 and 0.825. 

More detailed results of the example can be seen in the subdirectories of `example`. `example/Communities/` should contain a file which shows the learned class system. `example/Grammars/` should contain a file which contains the resulting phonotactic grammar, where each line takes the form `<constraint as regex>  <weight>`. `example/Judgements/` should contain a file which lists the predicted harmony scores for each of the forms in the test file. 

You can run your own languages by modifying the filepaths in the template. Note that the `Communities`, `Judgements`, `Grammars` directories, as well as the `_phones` file that appears in `example/` are created by the `.sh` and do not need to be made in advance. 



