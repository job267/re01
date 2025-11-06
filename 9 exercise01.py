import os
import numpy as np
from dotenv import load_dotenv
#启动环境变量

load_dotenv()
#启动deepseek
api_key = os.getenv("DEEPSEEK_API_KEY")
from openai import OpenAI
deepseek_client = OpenAI(
    api_key = api_key,
    base_url = "https://api.deepseek.com/v1" #基地址
)
#导入模型模块
from pymilvus import model as milvus_model
embedding_model = milvus_model.DefaultEmbeddingFunction()
#归一化函数
def normalize_batch(vectors):
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms
#获取嵌入维度
test_embedding = embedding_model.encode_queries(["this is a test"]) [0]
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10]) #测试文件
#初始化客户端
from pymilvus import MilvusClient
milvus_client = MilvusClient(uri = "./milvus_demo.db",timeout=600)
collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type='IP',
    consistent_level = "strong"
)

from tqdm import tqdm
data=[]
#text_line 提供知识传递
text_lines = [
"Milvus is a vector database designed specifically for AI applications and machine learning workloads.",
    "Data in Milvus is organized into collections, which are similar to tables in relational databases.",
    "Each collection contains multiple entities, and each entity consists of fields that can store various data types.",
    "Vector data in Milvus is stored as dense vectors in specialized vector indexes for efficient similarity search.",
    "Milvus supports multiple index types including IVF_FLAT, IVF_SQ8, HNSW, and Annoy for optimized vector retrieval.",
    "The storage architecture of Milvus separates compute from storage, using object storage for persistence and memory for caching.",
    "Milvus supports both structured and unstructured data, allowing you to store metadata alongside vector embeddings.",
    "Data persistence in Milvus is achieved through write-ahead logging (WAL) and periodic snapshots to prevent data loss.",
    "Milvus uses a distributed architecture that can scale horizontally to handle billions of vectors across multiple nodes.",
    "Collections in Milvus can be partitioned to improve query performance and manage data more efficiently.",
    "Each vector in Milvus is associated with a unique ID and can be retrieved using vector similarity search or exact match by ID.",
    "Milvus supports both CPU and GPU acceleration for vector operations, depending on the deployment configuration.",
    "The data model in Milvus allows for dynamic schemas, enabling flexible field definitions within collections.",
    "Milvus provides data consistency guarantees through its storage engine and supports both strong and eventual consistency models.",
    "Vector indexes in Milvus are built asynchronously, allowing for real-time insertion and search operations.",
    "Data compression techniques are employed in Milvus to reduce storage footprint while maintaining search performance.",
    "Milvus integrates with popular machine learning frameworks and can store embeddings from models like BERT, ResNet, and GPT.",
    "The query language in Milvus supports boolean expressions for filtering based on scalar fields alongside vector similarity search.",
    "Data backup and recovery in Milvus can be performed through built-in tools that export and import collections.",
    "Milvus supports multiple distance metrics including L2, IP, and cosine similarity for measuring vector similarity."
]
doc_embedding = embedding_model.encode_documents(text_lines)
doc_embedding = normalize_batch(doc_embedding) #数据库进行归一化操作
for i,line in enumerate(tqdm(text_lines,desc="Creating embeddings")):
    data.append({"id": i,"vector": doc_embedding[i],"text":line})
milvus_client.insert(collection_name=collection_name,data=data)

question = "how is data stored in milvus?"
#生成并将查询向量归一化
query_vec = embedding_model.encode_queries([question])
query_vec = normalize_batch(query_vec)

search_res = milvus_client.search(
    collection_name=collection_name,
    data=query_vec,
    limit=3,
    search_params={"metric_type":"IP","params":{}},
    output_fields=["text"]
)

import json
retrieved_lines_with_distances = [
    (res["entity"]["text"],res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances,indent=4))
context = "\n".join(
   [line_width_distance[0] for line_width_distance in retrieved_lines_with_distances]
)
SYSTEM_PROMPT = "human: 你是一个AI助手，你能够从提供的上下文段落中找到问题的答案。不要编造内容，如果无法确定答案，请说“根据提供的信息无法确定”。" #对提示词进行优化，使其基于知识库进行检索
USER_PROMPT =  '''
请使用以下用<context>标签括起来的信息片段来回答用<question>标签括起来的问题，最后追加原始回答的中文翻译，并用<translated>和</translated>
<context>
{context}
</context>
<question>
{question}
</question>
<translated>
</translated>
'''

response = deepseek_client.chat.completions.create(
    model = "deepseek-chat",
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":USER_PROMPT.format(context=context, question=question)}
    ]
)
print(response.choices[0].message.content)



