# DOME - Deidentification of Medical Entities

With the help of this framework, you can de-identify senstive data from your documents. It was developed at the University Hospital Essen by the [SHIP.AI team](https://ship-ai.ikim.nrw/).

These are the PHI categories and types that are annotated:

|PHI category|PHI type|Description|
| -------  | -------| ------ |
|Age | Age | Age in years
|Contact|Email|Email address|
|Contact|Fax | Fax number |
|Contact|IPAddress|IP address|
|Contact|Phone|Telephone number|
|Contact|URL|URL of a webpage|
|ID|PatientID|Medical record number or other patient identifier|
|ID|StudyID|Study title or name of a study|
|ID|Other|Other identification number|
|Location|Organization|Name of an organization|
|Location|Hospital|Name of a hospital, ward or other medical facility|
|Location|State|Name of a state|
|Location|Country|Name of a country|
|Location|City|Name of a city or city district|
|Location|Street|Street name including street number|
|Location|ZIP|ZIP-code |
|Location|Other|Other location|
|Name|Patient|Name of a patient
|Name|Staff|Name of a doctor or medical staff member|
|Name|Other|Name of a family member or other related person|
|Profession|Profession|Job title|
|Profession|Status|Employment status
|Other||PHI not covered by other types|

We are planning to release fine-tuned models on pseudonymized data. We will update this page as soon as the models are publicly available. 

## Inference

### Build docker image

Make sure the model you are using is placed in the `model` directory. The model should be loadable with the huggingface transformers library. It is highly recommended, that the model is fine-tuned on a de-identification task. Otherwise it probably won't do correct predictions.

You can build the docker image with

```bash
bash generate_version.sh
docker build -t ship-ai/dome .
```

### Run docker image

Set the variables `$INPUT_FOLDER` and `$OUTPUT_FOLDER` to the folder where the original documents are stored and the folder where the deidentified documents should be stored. 

```bash
docker run \
    --rm \
    -v $INPUT_FOLDER:/input\
    -v $OUTPUT_FOLDER:/output \
    --runtime=nvidia \
    --network host \
    --user $(id -u):$(id -g) \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/sh \
    ship-ai/dome \
    -c \
    "python -m dome.cli --input-folder /input --output-folder /output --verbose"
```
### Configuration
The decision which PHI are removed, replaced or shifted are stored in the `config/default-config.yaml`. You can change these values for each PHI according to your needs. Allowed values are
- redact
- replace
- shift 
- keep

 `redact` removes the PHI and replaces each character with an `X`. 
 `replace` replaces the PHI with the PHI-tag that contains the PHI category and type. `shift` shifts date or age values for a specific value. The value for the shift are also stored in the config-file under `OFFSET`. If you just want to leave a PHI untouched because it is not considered sensitive by your institution, you can set the value to `keep`. By default, each PHI is replaced with a PHI tag. 

 ## Training
 For fine-tuning an transformer model on a clinical de-identification task, you need to make sure that you have an annotated dataset that is compatible with the UIMA Cas Typesystem defined in `config/TypeSystem.xml`. Also you need to set the correct path to your dataset in an environment file like in `env/train-example.env`. In this file you can also set hyperparameters used for training. After you correctly set all necessary environment variables, you can execute

```bash
 poetry run python -m dome.bert \
                --env envs/train_example.env
```

## Evaluation
In order to evaluate the results of your model, there is a eval module in this framework. You can execute

```bash
poetry run python -m dome.eval \
       --typesystem config/TypeSystem.xml \
       --ground-truth path_to/ground_truth \
       --prediction path_to/predictions \
       --eval-name example_evaluation
```
You need to specify the CAS typesystem you are using, the path to the ground truth dataset, the path to the predictions of your model and a name for the evaluation. The results are written into a `results` folder in your cwd as a csv table. 

