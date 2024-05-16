meta_data_extraction_template="""Use the following pieces of context to accurately classify the documents based on the schema passed. Output should follow the pattern defined in schema.
    No verbose should be present. Output should follow the pattern defined in schema and the output should be in json format only so that it can be directly used with json.loads():
    {context}
    schema: {schema}
    //Note: Don't change the schema
"""

meta_data_filtration_template = """You are Database Admin, given the following MetaStore return all the `collection_name`  which might have some information on the about the `user_query`
    
    MetaStore
    -----------
    {metastore}
    
    output Format
    -------------
    [collection1, collection2]
    
    user_query:{query}
    //Note: Don't change the output Format, dont add anything else into it
"""

reranking_template='''Relevance measures how well the Context addresses the main aspects of the question, based on the context. Consider whether all and only the important aspects are contained in the Context when evaluating relevance. Given the context and question, score the relevance between one to five stars using the following rating scale: 

            One star: the answer completely lacks relevance 

            Two stars: the answer mostly lacks relevance 

            Three stars: the answer is partially relevant 

            Four stars: the answer is mostly relevant 

            Five stars: the answer has perfect relevance 

            This rating value should always be an integer between 1 and 5. So the rating produced should be 1 or 2 or 3 or 4 or 5. 
            
            Output should follow the pattern defined in schema and the output should be in json format only so that it can be directly used with json.loads():
             
            
            Schema
            -------------
            {schema}
            
            Previous Conversations
            -------------
            {chat_history}
            
            
            Context
            -------------
            {context}
            
            User Query
            --------------
            {user_query}
            
            // Note :
'''


rag_template=system_prompt='''Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            
            Previous Conversations
            -------------
            {chat_history}
            
            Context
            -------------
            {context}
            
            User Query
            --------------
            {user_query}
'''