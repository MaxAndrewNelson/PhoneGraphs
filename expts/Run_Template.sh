name="English_Template"
config="../expts/confs/English_SC_onsets.config"
premade_classes=""
cluster_on="contexts"
input_path="../example/Data/onset_tokens_arpa.txt"
experiment_dir="../example/"
test_file="../example/Data/Daland_et_al_arpa.txt"

PhoneGraphs="../src" 

echo "0. Making output directories and phone file"

### Make output folders if needed ###
if [[ ! -e ${experiment_dir}/Communities/ ]]; then
    mkdir ${experiment_dir}/Communities/
fi

if [[ ! -e ${experiment_dir}/Grammars/ ]]; then
    mkdir ${experiment_dir}/Grammars/
fi

if [[ ! -e ${experiment_dir}/Judgements/ ]]; then
    mkdir ${experiment_dir}/Judgements/
fi

python ${PhoneGraphs}/make_phones.py ${input_path} > ${experiment_dir}/${name}_phones.txt #make phones file

### Run class discovery ###
if [ -z "${premade_classes}" ]; then
    echo "1. Running class discovery algorithm on ${cluster_on}"
    if [ ${cluster_on} == "phones" ]; then
        python ${PhoneGraphs}/phoneme_clustering.py ${input_path} ${experiment_dir}/Communities/${name}_${i} ${config}
    else
        python ${PhoneGraphs}/context_clustering.py ${input_path} ${experiment_dir}/Communities/${name}_${i} ${config}
    fi
else
    echo "1. Skipping class discovery, using classes in ${premade_classes}"
    cp ${premade_classes} ${experiment_dir}/Communities/${name}
fi
    
### Fit a MaxEnt model ###

echo "2. Fitting phonotactic grammar"

python ${PhoneGraphs}/ng_phonotactic.py ${input_path} ${experiment_dir}/Communities/${name}_${i} ${test_file} ${experiment_dir}/${name}_phones.txt ${experiment_dir} ${name}_${i} ${config}

### Test correlations with Daland Et Al judgements ###
echo "3. Testing Daland Et Al correlations"

python ${experiment_dir}/Scripts/daland_eval.py ${experiment_dir}/Judgements/${name}_${i} ${experiment_dir}/Scripts/Daland_etal_2011__AverageScores.csv 

