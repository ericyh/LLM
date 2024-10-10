from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 50

documents = SimpleDirectoryReader("Archive").load_data()

index = VectorStoreIndex.from_documents(documents)

top_k = 3
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)


tokenizer = AutoTokenizer.from_pretrained("./Wizard-Vicuna-30B-Uncensored-GPTQ", device_map='auto')
model = AutoModelForCausalLM.from_pretrained("./Wizard-Vicuna-30B-Uncensored-GPTQ", device_map='auto')

system = "You are a psychology professor who uses the context provided to eagerly answer the users questions"
input_text = "SYSTEM: " + system
in_length = 0

def inference(input_text):
    global in_length
    inputs = tokenizer(input_text, return_tensors="pt")
    in_length = len(input_text)
    outputs = model.generate(**inputs, max_length = 2000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def output(generated_text):
    global in_length
    print("ASSISTANT: " + generated_text[in_length+1:])
    print("\n")

def get_query(query):
    response = query_engine.query(query)
    context = "Here are some context.\n"
    for i in range(top_k):
        context = context + response.source_nodes[i].text + "\n\n"
    return(context)


print("\n" * 50)    

while True:
    user = input("USER: ")
    context = get_query(user)
    print(context)
    print("\n")
    input_text += " USER:" + context + "\n" + user + " ASSISTANT: "
    generated_text = inference(input_text)
    output(generated_text)
    input_text = generated_text


