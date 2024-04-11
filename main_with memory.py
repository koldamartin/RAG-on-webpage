
# https://python.langchain.com/docs/use_cases/question_answering/chat_history/
# https://usescraper.com/blog/langchain-chatgpt-rag-with-your-website-content

from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma, FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader, UnstructuredMarkdownLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# 1. Create Markdown loader for scraped web page (made by Usescraper dashboard)
markdown_path = "scrape_https___addai.life_7YE4VY7W8QPP3KHQSSPRREN3Y7.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()

# 2. Split the data into chunks
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(data[0].page_content)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=100,
                                               length_function=len)
all_splits = text_splitter.split_documents(md_header_splits)

# 3. Create embeddings
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0",
                              cohere_api_key=cohere_api_key)

# 4. Set up a model
llm = ChatCohere(model='command-r',
                 temperature=0.1,
                 cohere_api_key=cohere_api_key)

# 5. Set up a Vector Store
# vectorstore = DocArrayInMemorySearch.from_documents(all_splits, embeddings)
# vectorstore = Chroma.from_documents(all_splits, embeddings, persist_directory='docs/chroma')
if not os.path.exists('docs/faiss-index'):
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    vectorstore.save_local('docs/faiss-index')
    print("Creating new vectorstore and saving it.")

else:
    vectorstore = FAISS.load_local('docs/faiss-index', embeddings, allow_dangerous_deserialization=True)
    print("Loading local vectorstore.")


# 6. Set up a retriever
retriever = vectorstore.as_retriever(search_type="mmr", ) #search_kwargs={"k": 1}
retriever_from_llm = MultiQueryRetriever.from_llm(llm=llm, retriever=retriever)

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever_from_llm, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


### Conversation loop ###
print("Pro ukončení konverzace napište exit.")
while True:
    query_input = input("\nPoložte otázku: ")
    if query_input == "exit":
        print(f"\nZde je přepis historie konverzace:\n{store['complete_history']}")
        break

    else:
        response = conversational_rag_chain.invoke({"input": query_input},
                                                   config={"configurable": {"session_id": "complete_history"}})

        print(f"Odpověď: {response['answer']}")
        print("")
        # Get the first relevant source text, works only for Command-R-Plus model
        print(f"Zdroj: {response['context'][0].page_content}")

# Get all the relevant source:
        #for document in response['context']:
            #print(f"Zdroj: {document[0]}\n")