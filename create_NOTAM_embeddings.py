from bert_embeddings import BertEmbeddingsExtractor
from sklearn.preprocessing import normalize
import os
import pandas as pd
import joblib
import torch

def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def process_notams(notam_data_path, model_base_dir, notam_type_to_model_dir, bert_model_path, output_csv_path):
    """
    Processes NOTAM data, extracts embeddings using BERT, and clusters them using pre-trained KMeans models.

    Args:
        notam_data_path (str): Path to the CSV file containing NOTAM data.
        model_base_dir (str): Base directory containing NOTAM type-specific model subdirectories.
        notam_type_to_model_dir (dict): Dictionary mapping NOTAM types to their corresponding model directories.
        bert_model_path (str): Path to the pre-trained BERT model.
        output_csv_path (str): Path to save the results (NOTAMs with cluster labels) to a CSV file.
    """
    # Initialize BERT embeddings extractor
    bert_extractor = BertEmbeddingsExtractor(bert_model_path)
    
    
    
    # Read the first file
    # df = pd.read_csv(file1_path)
    notam_df = pd.read_csv(notam_data_path, encoding='latin1')


    # Initialize a list to store the results
    results = []

    # Iterate over each NOTAM type and process accordingly
    for notam_type, model_dir in notam_type_to_model_dir.items():
        # Filter data by Notam Type
        filtered_data = notam_df[notam_df['Notam Type '] == notam_type]
        filtered_data=filtered_data[(filtered_data['id'] == 12)]
        # Load the KMeans model from the directory corresponding to the Notam Type
        model_path = os.path.join(model_dir, 'kmeans_model.pkl')
        #model_path = os.path.join(model_dir, 'kmeans_model_normalize.pkl')
         
        if not os.path.exists(model_path):
            print(f"Model for {notam_type} not found at {model_path}. Skipping...")
            continue
        
        kmeans = joblib.load(model_path)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        pca = joblib.load(os.path.join(model_dir, 'pca.pkl'))
        
       
        # Iterate through the filtered data
        for index, row in filtered_data.iterrows():
            notam_text = row['NOTAM ']
            print('notam---',index,notam_text)
            

            
            #notam='!SYR 02/042 SYR RWY 15/33 CLSD 2202050119-2202062100'
            text_split_required=notam_text.split()
            if is_num(text_split_required[-2]):
                j=-2
            
            else:
                if len(text_split_required[-2])==21 or  len(text_split_required[-2])==22:
                    j=-2
                else:
                    j=-1

        

    
            NOTAM_TAXIWAY_DESIGNATOR=text_split_required[3:j]
            if text_split_required[j-1]=="AT" and text_split_required[j-2]=="OBS":
                NOTAM_TAXIWAY_DESIGNATOR.append ('datetime')
                #NOTAM_TAXIWAY_DESIGNATOR.append(".")
                NOTAM_TAXIWAY_DESIGNATOR.append(".\n")
            else:
                if text_split_required[j] !=".":
                    
                    #NOTAM_TAXIWAY_DESIGNATOR.append(".")
                    NOTAM_TAXIWAY_DESIGNATOR.append(".\n")
            # Extract BERT embeddings for the NOTAM
            notam_text=" ".join(NOTAM_TAXIWAY_DESIGNATOR)
            sentence_id = 0  # Assuming each NOTAM corresponds to a single sentence; adjust if needed
            embeddings, labels, tokens = bert_extractor.get_bert_concate_embeddings(notam_text, sentence_id)
            
            # Use the embeddings as the feature vector for clustering
            word_embeddings_tensor = torch.tensor(embeddings)

            print('Before ...word_embeddings_tensor',word_embeddings_tensor)
            print('Before ...word_embeddings_tensor',word_embeddings_tensor.shape)
            # Calculate the average embedding
            word_embeddings_tensor = torch.mean(word_embeddings_tensor, dim=0).unsqueeze(0)
            #word_embeddings_tensor = torch.mean(word_embeddings_tensor, dim=0)
            print('after ...word_embeddings_tensor',word_embeddings_tensor)

            kmeans = joblib.load(model_path)
            scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            pca = joblib.load(os.path.join(model_dir, 'pca.pkl'))


            embeddings_scaled = scaler.transform(word_embeddings_tensor)
            #embeddings_normalize=normalize(embeddings)

            X_pca = pca.transform(embeddings_scaled)
            #X_pca=embeddings_scaled
            #X_pca = pca.transform(embeddings_normalize)
            # Get the cluster assignment
            cluster_label = kmeans.predict(X_pca )[0]
            print ('NOTAM------', notam_text,'Cluster Label-----',cluster_label)

            # Append the NOTAM and its cluster label to the results
            results.append({'NOTAM': notam_text, 'Cluster Label': cluster_label})

    # Convert the results to a DataFrame and save to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

# Example usage
notam_data_path = 'output_file.csv'
model_base_dir = "."
notam_type_to_model_dir = {
    'AIRSPACE': os.path.join(model_base_dir, 'AIRSPACE'),
    }
# change the path to your BERT model directory
bert_model_path = r'your bert model path'  # Replace with your actual BERT model path
output_dir=os.path.join(model_base_dir, 'AIRSPACE')
output_csv_path = os.path.join(output_dir, 'notam_cluster_results.csv')


process_notams(notam_data_path, model_base_dir, notam_type_to_model_dir, bert_model_path, output_csv_path)
