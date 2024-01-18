from utils.utils import init_argument_parser
from utils.openai_utils import build_chat_model
from utils.rag_utils import build_scientific_papers_loader, build_documents_retriever, format_docs
from dotenv import dotenv_values
import numpy as np
import random
import os
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser


def execute_task(opt, env):
    #fix seed if it is not None
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)

    with open(opt.template_file) as f:
        template = f.read()
    with open(opt.rag_template_file) as f:
        template_rag = template +"\n" + f.read()

    model, embeddings = build_chat_model(opt, env)
    prompt_parameters = dict()
    with open(opt.task_file) as f:
        prompt_parameters["input"] = f.read()

    
    os.makedirs(opt.output_folder, exist_ok=True)

    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])
    basic_query = prompt.format(**prompt_parameters)
    chain =  prompt | model | StrOutputParser() 
    print("running basic query")
    
    response = chain.invoke(prompt_parameters)
    basic_output_folder = os.path.join(opt.output_folder, "basic")
    os.makedirs(basic_output_folder, exist_ok=True)
    #write the basic query on txt file
    with open(os.path.join(basic_output_folder, "query.txt"), "w") as f:
        f.write(basic_query)

    #write the basic response on txt file
    with open(os.path.join(basic_output_folder, "response.txt"), "w") as f:
        f.write(response)
    
    print("loading papers")
    docs =  build_scientific_papers_loader(opt.papers_folder)
    retriever = build_documents_retriever(docs, db_persist_path=opt.db_persist_path, chunk_size=opt.chunk_size, chunk_overlap=opt.chunk_overlap, embeddings=embeddings)
    prompt_rag = ChatPromptTemplate.from_messages([("system", template_rag), ("human", "{input}")])
    context = (retriever | format_docs).invoke(prompt_parameters['input'])
    prompt_parameters['context'] = context
    rag_output_folder = os.path.join(opt.output_folder, "rag")
    os.makedirs(rag_output_folder, exist_ok=True)
    #save context on txt file
    with open(os.path.join(rag_output_folder, "context.txt"), "w") as f:
        f.write(context)
    rag_query = prompt_rag.format(**prompt_parameters)
    chain_rag =  prompt_rag | model | StrOutputParser()
    print("running rag query")
    response_rag = chain_rag.invoke(prompt_parameters)
    
    #write the rag query on txt file
    with open(os.path.join(rag_output_folder, "query.txt"), "w") as f:
        f.write(rag_query)
    
    #write the rag response on txt file
    with open(os.path.join(rag_output_folder, "response.txt"), "w") as f:
        f.write(response_rag)



def add_parse_arguments(parser):
    #model parameters
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0613', help='name of the model')
    parser.add_argument('--temperature', type=float, default=0.0, help='model temperature')

    #task parameters
    parser.add_argument('--task_file', type=str, default='data/tasks/pcir.txt', help='input task')
    parser.add_argument('--template_file', type=str, default='data/templates/write_paragraph_template.txt', help='template for the prompt')
  
    parser.add_argument('--seed', type=int, default=156, help='seed for reproducibility')

    #rag parameters
    parser.add_argument('--rag_template_file', type=str, default='data/rag_templates/basic_rag_suffix.txt', help='template for the prompt with RAG')
    parser.add_argument('--papers_folder', type=str, default='data/papers', help='folder with scientific papers')
    parser.add_argument('--db_persist_path', type=str, default='data/db/chroma', help='path to the db')
    parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='chunk overlap')

    #output
    parser.add_argument('--output_folder', type=str, default='output', help='output folder')
    return parser


def main():
    opt = init_argument_parser(add_parse_arguments)
    env = dotenv_values()
    execute_task(opt, env)

if __name__ == '__main__':
    main()