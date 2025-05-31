# NOtices-To-AirMen
# Semantics-aware prompting for translating NOtices To AirMen


This project focuses on translating and clustering NOTAMs (Notice to Airmen) using semantic techniques. It leverages BERT embeddings and KMeans clustering for analysis and translation of aviation-related messages.

## Project Structure

-   `apron.pdf`, `ficon.pdf`, `obst.pdf`: These files likely contain rulebooks or guidelines used in the translation process, possibly related to apron conditions, field conditions, and obstacles, respectively.
-   `bert_embeddings.py`: This script implements the [`BertEmbeddingsExtractor`]which is responsible for extracting BERT embeddings from text.
-   `create_NOTAM_embeddings.py`: This script processes NOTAM data, extracts embeddings using BERT, and clusters them using pre-trained KMeans models.
-   `one_shot_translation_cluster_opensource.py`: This script performs one-shot translation of NOTAMs using opensource language model, incorporating rulebooks and cluster prompts for context.
-   `prepocessed_NOTAM_pairs.csv`: This CSV file contains NOTAM pairs, including the original NOTAM, a human translation, NOTAM type, label, modified NOTAM, and final NOTAM.
-   `Requirements.txt`: This file lists the Python packages required to run the project.
-   `updated_clusterprompts.json`: This JSON file contains prompts used for clustering NOTAMs, organized by NOTAM type and label.

## Requirements

The project requires the following Python packages, as specified in [Requirements.txt](Requirements.txt):

```txt
transformers==4.47.1
torch==1.13.1
numpy==1.25.2
scikit-learn==1.2.2
pandas==1.5.3
openai==1.59.6
fitz==1.23.2
```

## Data
The project uses the prepocessed_NOTAM_pairs.csv file as the main data source. This file contains NOTAM pairs, including the original NOTAM, a human translation, NOTAM type, label, modified NOTAM, and final NOTAM.

## Configuration
The project uses the following configuration files:

updated_clusterprompts.json: This file contains example translation pairs for each cluster of a given NOTAM type. These pairs serve as prompts for the translation model.
rulebook_paths in one_shot_translation_cluster_opensource.py: This dictionary maps keywords to PDF files containing rulebooks or guidelines used in the translation process.
NOTAM_number.txt: This file lists unique NOTAM numbers for downloading data from the FAA.

## Usage
Install the required packages using pip:

```
pip install -r Requirements.txt
```

Download a pre-trained BERT model and specify the path in create_NOTAM_embeddings.py.

Update the paths to the data and model files in create_NOTAM_embeddings.py and one_shot_translation_cluster_opensource.py.

Run the scripts:

python create_NOTAM_embeddings.py
python one_shot_translation_cluster_opensource.py

## Notes 
The bert_model_path variable in create_NOTAM_embeddings.py needs to be updated to the actual path of your BERT model.
The OpenAI API key in one_shot_translation_cluster_opensource.py is set to lm-studio and the base URL is set to http://localhost:1234/v1. This configuration is intended for use with a local inference server like LM Studio.
The model parameter in the client.chat.completions.create method in one_shot_translation_cluster_opensource.py is set to "TheBloke/Mistral-7B-Instruct-v0.1-GGUF". Ensure this is correct for your setup and that the model is available on your local inference server.
The rulebooks (apron.pdf, ficon.pdf, obst.pdf) can be updated to reflect the latest NOTAM documentation.

