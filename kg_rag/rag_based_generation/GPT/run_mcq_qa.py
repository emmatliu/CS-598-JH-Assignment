'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys
import re


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


MODE = "0"
### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 
### MODE 4: Filter context                      ### 

# For mode 2 / 3
prior_domain_knowledge = """Related diseases often share gene associations. Variants follow the disease patterns of their genes. Ontology IDs are identifiers only. Disease–disease relations imply shared biology. Symptoms and provenance do not affect gene associations. Always return exactly one answer. Use context evidence first; only when every candidate has zero context evidence should you rely on prior biological knowledge to choose the most plausible candidate."""

# For mode 4
filter_prompt = """
You are given a JSON string containing disease context. Filter it so that it keeps only information
that is useful for answering the question. Keep entries that directly mention any candidate answer
or that could help infer a relation to a candidate (for example, associations like "Disease X is
associated with gene Y" may support a candidate gene Z if Y and Z are related). Remove clearly
irrelevant or uninformative entries. Return only the filtered JSON string.

Input JSON:
{json_str}

Candidates:
{candidate_list}

Filtered JSON:
"""


def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    
    answer_list = []
    detailed_answer_list = []  # just for recording Mode 4 outputs
    
    for index, row in tqdm(question_df.iterrows(), total=306):
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ### 
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: jsonlize the context from KG search ### 
                ### Please implement the first strategy here    ###
                context = retrieve_context_jsonize(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Question: "+ question + "\nContext: "+ context
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ### 
                ### Please implement the second strategy here   ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "\nQuestion: "+ question + "\n" + prior_domain_knowledge + "\nContext: "+ context # no "guess", Q -> prior -> context.
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            
            if MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ### 
                ### Please implement the third strategy here    ###
                context = retrieve_context_jsonize(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "\nQuestion: "+ question + "\n" + prior_domain_knowledge + "\nContext: "+ context # no "guess", Q -> prior -> context.
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "4":
                ### MODE 4: Mode 3 + a second LLM filter        ###
                context = retrieve_context_jsonize(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                candidate_list = re.search(r'Given list is:\s*(.*)$', question).group(1)
                filtered_context = get_Gemini_response(filter_prompt.format(json_str=context, candidate_list=candidate_list), SYSTEM_PROMPT, temperature=TEMPERATURE)
                enriched_prompt = "\nQuestion: "+ question + "\n" + prior_domain_knowledge + "\nContext: "+ filtered_context
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
                detailed_answer_list.append((row["text"], row["correct_node"], output, context, filtered_context))

            answer_list.append((row["text"], row["correct_node"], output))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True) 
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

    # to look at the shortened contexts
    if MODE == "4" and len(detailed_answer_list) > 0:
        save_for_context = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}_contexts.csv"
        output_file_mode4 = os.path.join(SAVE_PATH, f"{save_for_context}".format(mode=MODE))
        detailed_df = pd.DataFrame(
            detailed_answer_list,
            columns=["question", "correct_answer", "llm_answer", "original_context", "shortened_context"]
        )
        detailed_df.to_csv(output_file_mode4, index=False, header=True)

        
        
if __name__ == "__main__":
    main()


