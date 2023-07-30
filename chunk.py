class chunking:
    def __init__(self,text):
        self.text = text

    def yt_data(self):
        from llama_index import download_loader
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from transformers import GPT2TokenizerFast
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI
        from langchain.chains import ConversationalRetrievalChain
        # Step 3: Create function to count tokens
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text = self.text
        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=512,
            chunk_overlap=52,
            length_function=count_tokens,
        )

        chunks = text_splitter.create_documents([text])
        # Quick data visualization to ensure chunking was successful

        # Create a list of token counts
        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

        # Create a DataFrame from the token counts
        df = pd.DataFrame({'Token Count': token_counts})

        # Create a histogram of the token count distribution
        df.hist(bins=40, )

        # Show the plot
        plt.show()
        return chunks