# AI-Legal-Contract-Analyzer
This is a flask based application that allows the user to upload a text file that contains the contract and give small summary about the contract , the type of contract it is and  the red and green flags and the type of clause it is 
To use this code you need to have llama 3.2 1B instruct model and facebook/bart-large-cnn. llama 3.2 1B instruct is used for flag detection and clause prediction and bart-large-cnn is used for contract summarization. You can get llama 3.2 1B from huggingface 
Once you have all the models ready make sure that you fine tune llama 3.2 1B with the simplified_contract_dataset for flag classification and clause prediction purposes.
After that add teh code in your environmentand download all teh necessary libraries like flask and make template folder and add teh html file in teh templates folder.
Run the code after all the process 
